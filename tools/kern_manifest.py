"""Manifest post-pass shared by the generators: turn an explicit, verbose
manifest into the normalized wire form (schema_version 3).

A generator writes everything out longhand — every launch with its full
ABI and wiring, every call with every ABI scalar the mined kernel takes,
each launch naming its artifact inline as ``{"cubin": ..., "sha256": ...}``.
`normalize` is the linker pass that makes the manifest minimal without
changing what runs:

1. **hoist modules** — inline ``cubin``/``sha256`` pairs become entries of
   the top-level ``modules`` table (the manifest's dependency list); each
   launch keeps only ``"module": <name>``.
2. **fold constants** — an interface scalar that every call of an op passes
   as the same literal is not part of the contract, it is the impl's ABI
   constant (a mined kernel's strides, flags, eps). It leaves the interface
   and becomes a literal in each launch's wiring; every call drops it.
3. **default the identity** — a launch whose ABI equals the op's params
   omits ``params``; wiring that forwards the params in order omits ``args``.
4. **extern launches have no geometry** — ``block``/``grid``/``shared_mem``
   are dropped from ``extern:`` entries.

Keys are emitted in a canonical order so diffs stay readable.
"""

import copy
import re

SCHEMA_VERSION = 3
SCALARS = ("i32", "i64", "f32", "u8")

_TOP = ["schema_version", "model", "spec", "vars", "states", "buffers", "modules", "ops", "programs"]
_BUFFER = ["dtype", "shape", "kind", "domain"]
_LAUNCH = ["module", "entry", "params", "block", "grid", "shared_mem", "args"]
_CALL = ["label", "op", "args"]


def _order(d, keys):
    return {k: d[k] for k in keys if k in d} | {k: v for k, v in d.items() if k not in keys}


def _is_literal(arg):
    return isinstance(arg, dict) and len(arg) == 1 and next(iter(arg)) in SCALARS


def module_name(source):
    """Human name for a module from its source: the repo of a registry ref,
    else the file stem minus any ``-<sha12>`` suffix."""
    if source.startswith("hf:"):
        return source[3:].split("@")[0].split("/")[1]
    stem = source.rsplit("/", 1)[-1]
    stem = stem[: -len(".cubin")] if stem.endswith(".cubin") else stem.rsplit(".", 1)[0]
    return re.sub(r"-[0-9a-f]{12}$", "", stem)


def hoist_modules(m):
    modules = dict(m.get("modules", {}))
    by_sha = {v["sha256"]: k for k, v in modules.items()}
    for op in m["ops"].values():
        for launch in op["impl"]["launches"]:
            cubin, sha = launch.pop("cubin", None), launch.pop("sha256", None)
            if cubin is None:
                assert sha is None, f"{launch['entry']}: sha256 without cubin"
                continue
            assert sha, f"{launch['entry']}: cubin `{cubin}` without sha256"
            if sha not in by_sha:
                name = module_name(cubin)
                if name in modules:
                    name = f"{name}-{sha[:8]}"
                assert name not in modules
                modules[name] = {"source": cubin, "sha256": sha}
                by_sha[sha] = name
            launch["module"] = by_sha[sha]
    m["modules"] = dict(sorted(modules.items()))


def _materialize(op):
    """Make every launch's params/args explicit against the op's interface."""
    for launch in op["impl"]["launches"]:
        launch.setdefault("params", list(op["params"]))
        launch.setdefault("args", [{"param": i} for i in range(len(op["params"]))])


def fold_constants(m):
    calls_by_op = {}
    for calls in m["programs"].values():
        for c in calls:
            calls_by_op.setdefault(c["op"], []).append(c)
    for oname, op in m["ops"].items():
        calls = calls_by_op.get(oname)
        if not calls:
            continue
        params = op["params"]
        folded = {}
        for i, p in enumerate(params):
            if p not in SCALARS:
                continue
            first = calls[0]["args"][i]
            if _is_literal(first) and all(c["args"][i] == first for c in calls):
                folded[i] = first
        if not folded:
            continue
        _materialize(op)
        keep = [i for i in range(len(params)) if i not in folded]
        renumber = {old: new for new, old in enumerate(keep)}
        for launch in op["impl"]["launches"]:
            launch["args"] = [
                folded[a["param"]] if "param" in a and a["param"] in folded
                else {"param": renumber[a["param"]]} if "param" in a
                else a
                for a in launch["args"]
            ]
        op["params"] = [params[i] for i in keep]
        for c in calls:
            c["args"] = [c["args"][i] for i in keep]


def default_identity(m):
    for op in m["ops"].values():
        n = len(op["params"])
        for launch in op["impl"]["launches"]:
            if launch.get("params") == op["params"]:
                del launch["params"]
            if launch.get("args") == [{"param": i} for i in range(n)]:
                del launch["args"]


def normalize(m):
    m = copy.deepcopy(m)
    assert m.get("schema_version") == SCHEMA_VERSION, m.get("schema_version")
    for op in m["ops"].values():
        for launch in op["impl"]["launches"]:
            for a in launch.get("args", []):
                if "scratch" in a:
                    assert a.pop("offset", 0) == 0, "scratch offsets are gone; declare two scratches"
    hoist_modules(m)
    fold_constants(m)
    default_identity(m)
    for op in m["ops"].values():
        launches = []
        for launch in op["impl"]["launches"]:
            if launch["entry"].startswith("extern:"):
                for k in ("module", "block", "grid", "shared_mem"):
                    launch.pop(k, None)
            launches.append(_order(launch, _LAUNCH))
        op["impl"]["launches"] = launches
    m["buffers"] = {k: _order(v, _BUFFER) for k, v in m["buffers"].items()}
    m["programs"] = {k: [_order(c, _CALL) for c in v] for k, v in m["programs"].items()}
    return _order(m, _TOP)
