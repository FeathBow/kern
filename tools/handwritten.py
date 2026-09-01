"""Handwritten kernels (tools/kernels-src/*.cu): build once, pin by sha256.

A generator writes `**hw("gemm8")` into a step and gets
`{"cubin": "gemm8.cubin", "sha256": "<hash of the current build>"}` — the
display name plus the identity the runtime resolves. The build happens at
most once per process (tools/build_kernels.sh, into target/cubins/).
"""
import functools
import hashlib
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def build_dir() -> pathlib.Path:
    out = REPO / "target" / "cubins"
    subprocess.run([str(REPO / "tools" / "build_kernels.sh"), str(out)], check=True)
    return out


def hw(name: str) -> dict:
    """`cubin` + `sha256` fields for tools/kernels-src/<name>.cu as built now."""
    cb = build_dir() / f"{name}.cubin"
    return {"cubin": f"{name}.cubin", "sha256": hashlib.sha256(cb.read_bytes()).hexdigest()}
