//! Manifest schema (format 3). Parsing is already strict: unknown fields,
//! duplicate names and malformed type strings are rejected at
//! deserialization time. Semantic checks (references, dtypes, dataflow,
//! bounds) live in [`crate::verify`].
//!
//! Vocabulary, one word per level so nothing collides:
//!
//! ```text
//! programs.<name>[]          a *call* of an op            {"op": "attn", "args": [...]}
//! ops.<name>                 an op: interface + impl      {"params": [...], "impl": {...}}
//! ops.<name>.impl.launches[] a *launch* of a module entry {"module": "argmax", "entry": "kern_argmax_partial"}
//! modules.<name>             an artifact the launches pin {"source": "argmax.cubin", "sha256": "..."}
//! vars.<name>                a per-call scalar the caller supplies, bounded
//! states.<name>              opaque persistent memory, sized by the runtime
//! buffers.<name>             typed tensors: input / output / weight / workspace / carry
//! ```

use serde::de::{Error as DeError, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

/// The one format this crate reads and writes.
pub const SCHEMA_VERSION: u32 = 3;

/// Deserialize a JSON object into a map, rejecting duplicate keys (plain
/// serde silently keeps the last one).
fn unique_map<'de, D, V>(deserializer: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct UniqueMap<V>(PhantomData<V>);

    impl<'de, V: Deserialize<'de>> Visitor<'de> for UniqueMap<V> {
        type Value = BTreeMap<String, V>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a map with unique keys")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
            let mut map = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, V>()? {
                if map.insert(key.clone(), value).is_some() {
                    return Err(A::Error::custom(format!("duplicate name `{key}`")));
                }
            }
            Ok(map)
        }
    }

    deserializer.deserialize_map(UniqueMap(PhantomData))
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Wire-format version. Only [`SCHEMA_VERSION`] is accepted.
    pub schema_version: u32,
    /// Free-form label; the runtime assigns no meaning to it.
    pub model: String,
    /// Caller contract of a speculative-decoding manifest (absent for plain
    /// prefill/decode ones). The runtime assigns no meaning to it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<Spec>,
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub vars: BTreeMap<String, Var>,
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub states: BTreeMap<String, State>,
    #[serde(deserialize_with = "unique_map")]
    pub buffers: BTreeMap<String, Buffer>,
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub modules: BTreeMap<String, Module>,
    #[serde(deserialize_with = "unique_map")]
    pub ops: BTreeMap<String, Op>,
    /// Each program is a straight-line list of op calls.
    #[serde(deserialize_with = "unique_map")]
    pub programs: BTreeMap<String, Vec<Call>>,
}

impl Manifest {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest serialization cannot fail")
    }
}

/// How a driver stages the `draft` / `verify` programs of a speculative
/// manifest: `draft` runs over `block` rows = the anchor token followed by
/// `block - 1` copies of `mask_token`, `verify` over `block` rows = the
/// anchor followed by the `block - 1` drafted tokens.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub block: u64,
    pub mask_token: i64,
}

/// A per-call scalar the caller supplies (e.g. the token count this step),
/// bounded `1..=max`. Vars are the only thing that may size a shape or a
/// launch grid; any other caller-supplied number is data and goes through an
/// input buffer. Verification runs at the bounds; the runtime rejects an
/// out-of-range value at call time.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Var {
    pub max: u64,
}

impl Var {
    /// Lower bound of every var.
    pub const MIN: u64 = 1;
}

/// Opaque persistent memory. The runtime knows only how many bytes to
/// provision — per token slot (`bytes_per_token`, scaled by the capacity:
/// paged KV) or one fixed block regardless of capacity (`bytes`:
/// per-sequence recurrent state such as a Mamba/GDN conv + SSM state).
/// Exactly one of the two is non-zero. The internal layout belongs to the
/// provider's kernels, which receive the base pointer as a `state` param.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct State {
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes_per_token: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes: u64,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    pub dtype: DType,
    pub shape: Vec<Dim>,
    pub kind: BufferKind,
    /// Optional prior on the buffer's *contents*. Never required: a manifest
    /// without domains runs exactly the same. With one, the runtime rejects
    /// out-of-domain input writes, and attestation can synthesize valid
    /// values for the buffer (and check produced values against it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<Domain>,
}

/// A prior on buffer contents, declared by whoever wires the model — the
/// only party that knows a `buffer<i32>` is a page table and not an
/// activation. It is unary (one buffer, no relations between buffers) and
/// says nothing about kernel behaviour: a kernel that misbehaves on
/// in-domain input is a kernel bug the harness can now provoke.
///
/// Two forms, mutually exclusive:
/// - `min`/`max`: inclusive bounds, integers, floats or a var expression;
///   either may be omitted. Float buffers without a domain are implicitly
///   "any finite value".
/// - `index_into`: every element indexes a row of the named buffer or a
///   token slot of the named state, `stride` rows/tokens per index (a paged
///   KV block table indexes 16 tokens at a time).
///
/// `monotone` additionally requires a non-decreasing sequence (prefix sums
/// such as `cu_seqlens`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Domain {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<Bound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<Bound>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_into: Option<String>,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub stride: u64,
    #[serde(default, skip_serializing_if = "is_false")]
    pub monotone: bool,
}

fn one() -> u64 {
    1
}
fn is_one(v: &u64) -> bool {
    *v == 1
}
fn is_false(v: &bool) -> bool {
    !*v
}

/// A domain bound: a literal integer, a literal float, or a var expression
/// (evaluated at the caller's var values).
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Bound {
    Int(i64),
    Float(f64),
    Expr(Expr),
}

impl Bound {
    pub fn eval(&self, env: &BTreeMap<String, u64>) -> Result<f64, EvalError> {
        Ok(match self {
            Bound::Int(v) => *v as f64,
            Bound::Float(v) => *v,
            Bound::Expr(e) => e.eval(env)? as f64,
        })
    }
}

/// A domain with its bounds evaluated for one var environment and one
/// state capacity. `lo`/`hi` are inclusive; `None` is unbounded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedDomain {
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub monotone: bool,
}

impl ResolvedDomain {
    pub fn contains(&self, v: f64) -> bool {
        if v.is_nan() {
            return false;
        }
        self.lo.is_none_or(|lo| v >= lo) && self.hi.is_none_or(|hi| v <= hi)
    }
}

impl Domain {
    /// Row count of `index_into`'s target at `env`, or its token capacity for
    /// a state; `None` when the name resolves to nothing.
    fn target_rows(
        &self,
        m: &Manifest,
        env: &BTreeMap<String, u64>,
        state_capacity_tokens: u64,
    ) -> Result<Option<u64>, EvalError> {
        let Some(t) = &self.index_into else { return Ok(None) };
        if let Some(b) = m.buffers.get(t) {
            return Ok(Some(match b.shape.first() {
                Some(Dim::Const(c)) => *c,
                Some(Dim::Var(s)) => {
                    *env.get(s).ok_or_else(|| EvalError::UnknownVar(s.clone()))?
                }
                None => 0,
            }));
        }
        if m.states.contains_key(t) {
            return Ok(Some(state_capacity_tokens));
        }
        Ok(None)
    }

    /// Evaluate the bounds. Verification guarantees the references resolve;
    /// an unknown `index_into` target here yields an unbounded domain.
    pub fn resolve(
        &self,
        m: &Manifest,
        env: &BTreeMap<String, u64>,
        state_capacity_tokens: u64,
    ) -> Result<ResolvedDomain, EvalError> {
        if self.index_into.is_some() {
            let rows = self.target_rows(m, env, state_capacity_tokens)?;
            let hi = rows.map(|r| (r / self.stride.max(1)).saturating_sub(1) as f64);
            return Ok(ResolvedDomain { lo: Some(0.0), hi, monotone: self.monotone });
        }
        Ok(ResolvedDomain {
            lo: self.min.as_ref().map(|b| b.eval(env)).transpose()?,
            hi: self.max.as_ref().map(|b| b.eval(env)).transpose()?,
            monotone: self.monotone,
        })
    }
}

/// One shape extent: a constant or the name of a var.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Dim {
    Const(u64),
    Var(String),
}

/// Who provides a buffer and how long its contents live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BufferKind {
    /// Written by the runtime before program execution.
    Input,
    /// Read back by the runtime after program execution.
    Output,
    /// Bound by name from the weight artifact at load time.
    Weight,
    /// Planned and owned by the runtime; contents dead across program runs.
    Workspace,
    /// Written by one program, read by another; contents persist across
    /// program runs. Which program runs first is the caller's contract —
    /// the verifier only requires that *some* program writes it.
    Carry,
}

impl fmt::Display for BufferKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BufferKind::Input => "input",
            BufferKind::Output => "output",
            BufferKind::Weight => "weight",
            BufferKind::Workspace => "workspace",
            BufferKind::Carry => "carry",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Bf16,
    F16,
    F32,
    Fp8E4m3,
    I32,
    U32,
    I64,
    U8,
}

impl DType {
    pub fn bytes(self) -> u64 {
        match self {
            DType::Bf16 | DType::F16 => 2,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::I64 => 8,
            DType::Fp8E4m3 | DType::U8 => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DType::Bf16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::Fp8E4m3 => "fp8e4m3",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I64 => "i64",
            DType::U8 => "u8",
        }
    }
}

impl FromStr for DType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        Ok(match s {
            "bf16" => DType::Bf16,
            "f16" => DType::F16,
            "f32" => DType::F32,
            "fp8e4m3" => DType::Fp8E4m3,
            "i32" => DType::I32,
            "u32" => DType::U32,
            "i64" => DType::I64,
            "u8" => DType::U8,
            _ => return Err(format!("unknown dtype `{s}`")),
        })
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl schemars::JsonSchema for DType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "DType".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "Element type of a buffer or scratch declaration.",
            "type": "string",
            "enum": ["bf16", "f16", "f32", "fp8e4m3", "i32", "u32", "i64", "u8"],
        })
    }
}

impl Serialize for DType {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(D::Error::custom)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    In,
    Out,
    InOut,
}

impl fmt::Display for Dir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Dir::In => "in",
            Dir::Out => "out",
            Dir::InOut => "inout",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    I64,
    F32,
    /// Single-byte scalar (bool flags in mined vLLM kernel ABIs stage as
    /// one-byte params).
    U8,
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ScalarType::I32 => "i32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
            ScalarType::U8 => "u8",
        })
    }
}

/// One parameter, written as a string in the manifest: `"in buffer<bf16>"`,
/// `"out buffer<fp8e4m3>"`, `"inout state"`, `"i32"`. Buffers and states
/// require an explicit direction; scalars are by-value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Buf { dtype: DType, dir: Dir },
    /// An opaque state base pointer (only a `states` entry can bind here).
    State { dir: Dir },
    Scalar(ScalarType),
}

impl ParamType {
    /// Size of the param slot in the kernel ABI, cross-checked against
    /// `cuFuncGetParamInfo` when the module is loaded.
    pub fn size_bytes(self) -> u64 {
        match self {
            ParamType::Buf { .. } | ParamType::State { .. } => 8,
            ParamType::Scalar(ScalarType::I64) => 8,
            ParamType::Scalar(ScalarType::U8) => 1,
            ParamType::Scalar(_) => 4,
        }
    }

    /// Direction of a pointer param; `None` for scalars.
    pub fn dir(self) -> Option<Dir> {
        match self {
            ParamType::Buf { dir, .. } | ParamType::State { dir } => Some(dir),
            ParamType::Scalar(_) => None,
        }
    }
}

impl FromStr for ParamType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        match s {
            "i32" => return Ok(ParamType::Scalar(ScalarType::I32)),
            "i64" => return Ok(ParamType::Scalar(ScalarType::I64)),
            "f32" => return Ok(ParamType::Scalar(ScalarType::F32)),
            "u8" => return Ok(ParamType::Scalar(ScalarType::U8)),
            _ => {}
        }
        let (dir_s, rest) = s
            .split_once(' ')
            .ok_or_else(|| format!("invalid param type `{s}`"))?;
        let dir = match dir_s {
            "in" => Dir::In,
            "out" => Dir::Out,
            "inout" => Dir::InOut,
            _ => return Err(format!("invalid direction `{dir_s}` in param type `{s}`")),
        };
        let rest = rest.trim();
        if rest == "state" {
            return Ok(ParamType::State { dir });
        }
        if let Some(dt) = rest.strip_prefix("buffer<").and_then(|r| r.strip_suffix('>')) {
            let dtype = dt.parse::<DType>().map_err(|e| format!("{e} in param type `{s}`"))?;
            return Ok(ParamType::Buf { dtype, dir });
        }
        Err(format!("invalid param type `{s}`"))
    }
}

impl fmt::Display for ParamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParamType::Buf { dtype, dir } => write!(f, "{dir} buffer<{dtype}>"),
            ParamType::State { dir } => write!(f, "{dir} state"),
            ParamType::Scalar(st) => write!(f, "{st}"),
        }
    }
}

impl schemars::JsonSchema for ParamType {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "ParamType".into()
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "description": "One parameter: a by-value scalar (`\"i32\"`, `\"i64\"`, \
                `\"f32\"`, `\"u8\"`) or a directional pointer — `\"in\"`/`\"out\"`/\
                `\"inout\"` followed by `\"state\"` (opaque state base) or \
                `\"buffer<dtype>\"`.",
            "type": "string",
            "pattern": "^(i32|i64|f32|u8|(in|out|inout) (state|buffer<(bf16|f16|f32|fp8e4m3|i32|u32|i64|u8)>))$",
        })
    }
}

impl Serialize for ParamType {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ParamType {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(D::Error::custom)
    }
}

/// A code artifact the launches pin: a local file name (`argmax.cubin`) or a
/// registry ref (`hf:org/repo/path[@rev]`). The runtime never resolves by
/// `source`: it loads every artifact in the kernel dir (plus fetched
/// registry refs), hashes each, and matches by `sha256` — so one dir can hold
/// every version of a kernel, and a rebuilt artifact with different bytes
/// is a different module. `source` is a label and, for registry refs, a URL.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Module {
    pub source: String,
    pub sha256: String,
}

/// An op is an **interface** (the typed params a call passes — the only
/// thing a call site knows) plus an **implementation**: how those args are
/// lowered onto actual launches. Swapping the implementation — a faster
/// kernel from elsewhere with the same interface — touches only `impl`;
/// every call stays untouched.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Op {
    /// The interface: typed, directional params. Call-site semantics
    /// (what each position means) are the contract implementations must
    /// honor; the verifier can only check shape, not meaning.
    pub params: Vec<ParamType>,
    #[serde(rename = "impl")]
    pub imp: Impl,
}

/// An implementation is a micro-program: private scratch buffers (sized by
/// var expressions, provisioned by the runtime, dead outside one call) and
/// one or more launches. Launch geometry lives here, not at the call site —
/// it belongs to the implementation.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Impl {
    #[serde(
        default,
        deserialize_with = "unique_map",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub scratch: BTreeMap<String, Scratch>,
    pub launches: Vec<Launch>,
}

/// One scratch buffer declaration: like a workspace buffer, but private to
/// the implementation.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Scratch {
    pub dtype: DType,
    pub shape: Vec<Dim>,
}

/// One launch inside an implementation. The common single-launch op whose
/// ABI *is* the interface writes only `module`, `entry`, `block`, `grid`:
/// `params` defaults to the op's params and `args` to forwarding them in
/// order.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Launch {
    /// Name of the `modules` entry this entry point lives in. Absent: the
    /// runtime searches every loaded module (disambiguating by param
    /// layout) — mined kernels that only know their symbol. `extern:`
    /// entries have no module.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// Entry point in the module, or `extern:<name>` for a runtime built-in
    /// (no module, no geometry).
    pub entry: String,
    /// This launch's own ABI (what the entry actually takes). Absent: the
    /// op's `params`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Vec<ParamType>>,
    /// Required unless `entry` is `extern:`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<[u32; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grid: Option<[Expr; 3]>,
    /// Dynamic shared memory in bytes, if the launch needs any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mem: Option<Expr>,
    /// Wiring: where each launch param comes from — a forwarded interface
    /// param, a scratch buffer, or an implementation-private literal.
    /// Absent: `[{"param": 0}, {"param": 1}, ...]` over the op's params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<LaunchArg>>,
}

impl Launch {
    pub fn is_extern(&self) -> bool {
        self.entry.starts_with("extern:")
    }

    /// The launch ABI, defaulting to the op's interface.
    pub fn params_of<'a>(&'a self, op: &'a Op) -> &'a [ParamType] {
        self.params.as_deref().unwrap_or(&op.params)
    }

    /// The wiring, defaulting to forwarding the op's params in order.
    pub fn args_of(&self, op: &Op) -> Cow<'_, [LaunchArg]> {
        match &self.args {
            Some(a) => Cow::Borrowed(a),
            None => Cow::Owned((0..op.params.len()).map(|param| LaunchArg::Param { param }).collect()),
        }
    }
}

/// A launch argument. `{"param": 0}` forwards the call's arg #0 verbatim;
/// `{"scratch": "pmax"}` passes a private scratch pointer; scalar literals
/// are implementation constants the interface never sees (a partial-count
/// baked into a two-stage reduction, a mined ABI's strides and flags).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum LaunchArg {
    Param { param: usize },
    Scratch { scratch: String },
    I32 { i32: i32 },
    I64 { i64: i64 },
    F32 { f32: f32 },
    U8 { u8: u8 },
}

impl fmt::Display for LaunchArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchArg::Param { param } => write!(f, "interface param #{param}"),
            LaunchArg::Scratch { scratch } => write!(f, "scratch `{scratch}`"),
            LaunchArg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            LaunchArg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            LaunchArg::F32 { f32: v } => write!(f, "f32 literal {v}"),
            LaunchArg::U8 { u8: v } => write!(f, "u8 literal {v}"),
        }
    }
}

/// A remote artifact reference in a module's `source`:
/// `hf:<org>/<repo>/<path>[@<revision>]` (revision defaults to `main`).
/// The runtime materializes it into a content-addressed local cache at load
/// time; the module's `sha256` is the artifact's identity, so the transport
/// needs no trust. Names are just URLs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryRef {
    pub org: String,
    pub repo: String,
    pub path: String,
    pub revision: String,
}

impl RegistryRef {
    /// `None` if `s` is a plain local file name (no registry prefix);
    /// otherwise the parsed ref or why it is malformed.
    pub fn parse(s: &str) -> Option<Result<RegistryRef, String>> {
        let rest = s.strip_prefix("hf:")?;
        let malformed =
            || format!("invalid registry ref `{s}`: expected hf:<org>/<repo>/<path>[@revision]");
        let (rest, revision) = match rest.rsplit_once('@') {
            Some((r, rev)) if !rev.is_empty() && !rev.contains('/') => (r, rev),
            Some(_) => return Some(Err(malformed())),
            None => (rest, "main"),
        };
        let Some((org, rest)) = rest.split_once('/') else {
            return Some(Err(malformed()));
        };
        let Some((repo, path)) = rest.split_once('/') else {
            return Some(Err(malformed()));
        };
        if org.is_empty()
            || repo.is_empty()
            || path.is_empty()
            || path.split('/').any(|seg| seg.is_empty() || seg == "." || seg == "..")
        {
            return Some(Err(malformed()));
        }
        Some(Ok(RegistryRef {
            org: org.to_string(),
            repo: repo.to_string(),
            path: path.to_string(),
            revision: revision.to_string(),
        }))
    }
}

/// The closed scalar-expression set: a constant, a var name (`"tokens"`),
/// `{"ceil_div": [e, c]}`, `{"mul": [e, c]}`. This is deliberately not a
/// language: grid geometry, dynamic shared memory and var-derived scalar
/// args are the only runtime-dependent numbers a provider may compute, and
/// only with these forms.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Expr {
    Const(u64),
    /// A var by name — the same bare-string form a shape uses.
    Var(String),
    CeilDiv { ceil_div: (Box<Expr>, u64) },
    Mul { mul: (Box<Expr>, u64) },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("unknown var `{0}`")]
    UnknownVar(String),
    #[error("arithmetic overflow")]
    Overflow,
    #[error("division by zero")]
    DivByZero,
}

impl Expr {
    pub fn eval(&self, env: &BTreeMap<String, u64>) -> Result<u64, EvalError> {
        match self {
            Expr::Const(c) => Ok(*c),
            Expr::Var(var) => env
                .get(var)
                .copied()
                .ok_or_else(|| EvalError::UnknownVar(var.clone())),
            Expr::CeilDiv { ceil_div: (inner, c) } => {
                if *c == 0 {
                    return Err(EvalError::DivByZero);
                }
                let x = inner.eval(env)?;
                Ok(x.checked_add(c - 1).ok_or(EvalError::Overflow)? / c)
            }
            Expr::Mul { mul: (inner, c) } => {
                inner.eval(env)?.checked_mul(*c).ok_or(EvalError::Overflow)
            }
        }
    }
}

/// A call site: which op, with which args. No geometry — grid/block belong
/// to the op's implementation. `label` is for humans and diagnostics only.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Call {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub op: String,
    pub args: Vec<Arg>,
}

/// A call argument. Explicitly tagged so a buffer name can never be
/// mistaken for a var name: `{"buf": "hidden"}`, `{"state": "kv"}`,
/// `{"var": "tokens"}`, `{"i32": 2560}`, `{"i64": 4096}`, `{"f32": 1e-6}`.
///
/// Buffer and state args carry an optional byte `offset` (default 0): the
/// kernel receives base + offset. This is how a provider addresses a view
/// inside a fused buffer (q/k/v slices of a merged qkv projection) or a
/// per-layer region inside an opaque state — the offset is a literal from
/// the provider's own layout arithmetic, the runtime just adds it.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Arg {
    Buf {
        buf: String,
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
    },
    State {
        state: String,
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
    },
    Var { var: String },
    /// Var-derived integer scalar, e.g. `{"expr": {"mul": ["tokens", 32]}}`.
    /// Same closed expression set as launch grids; f32 params cannot bind one.
    Expr { expr: Expr },
    I32 { i32: i32 },
    I64 { i64: i64 },
    F32 { f32: f32 },
    U8 { u8: u8 },
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Buf { buf, offset: 0 } => write!(f, "buffer `{buf}`"),
            Arg::Buf { buf, offset } => write!(f, "buffer `{buf}`+{offset}"),
            Arg::State { state, offset: 0 } => write!(f, "state `{state}`"),
            Arg::State { state, offset } => write!(f, "state `{state}`+{offset}"),
            Arg::Var { var } => write!(f, "var `{var}`"),
            Arg::Expr { .. } => write!(f, "expression"),
            Arg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            Arg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            Arg::F32 { f32: v } => write!(f, "f32 literal {v}"),
            Arg::U8 { u8: v } => write!(f, "u8 literal {v}"),
        }
    }
}
