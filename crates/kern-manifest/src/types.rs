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

/// The whole contract a model ships as: one JSON file naming its vars, states, buffers, modules, ops and programs.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Wire-format version; must be `3`.
    pub schema_version: u32,
    /// Free-form model label, e.g. `"qwen3-4b"`.
    pub model: String,
    /// Speculative-decoding caller contract; absent for plain prefill/decode manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<Spec>,
    /// Per-call scalars the caller supplies, e.g. `{"tokens": {"max": 2048}}`.
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub vars: BTreeMap<String, Var>,
    /// Opaque persistent memory the runtime provisions by size, e.g. a paged KV cache.
    #[serde(default, deserialize_with = "unique_map", skip_serializing_if = "BTreeMap::is_empty")]
    pub states: BTreeMap<String, State>,
    /// Typed tensors: inputs, outputs, weights, workspace and carries.
    #[serde(deserialize_with = "unique_map")]
    pub buffers: BTreeMap<String, Buffer>,
    /// Code artifacts the launches pin, by name, e.g. `{"argmax": {"source": "argmax.cubin", "sha256": "2537…"}}`.
    #[serde(deserialize_with = "unique_map")]
    pub modules: BTreeMap<String, Module>,
    /// Operators: a typed interface plus the launches that implement it.
    #[serde(deserialize_with = "unique_map")]
    pub ops: BTreeMap<String, Op>,
    /// Named straight-line call lists, e.g. `"decode": [{"op": "embedding", "args": [...]}, ...]`.
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

/// Speculative-decoding contract: `draft` and `verify` run over `block` rows — the anchor token followed by `block - 1` mask tokens (draft) or drafted tokens (verify).
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    /// Rows per draft/verify call, e.g. `8`.
    pub block: u64,
    /// Token id filling the undrafted rows, e.g. `248070`.
    pub mask_token: i64,
}

/// A per-call scalar the caller supplies, bounded `1..=max`; the only kind of number that may size a shape or a grid.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Var {
    /// Upper bound, e.g. `2048` for the token count of a prefill chunk.
    pub max: u64,
}

impl Var {
    /// Lower bound of every var.
    pub const MIN: u64 = 1;
}

/// Opaque persistent memory; the runtime provisions the bytes and hands the base pointer to `inout state` params.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct State {
    /// Bytes per token slot, scaled by the capacity — a paged KV cache, e.g. `147456`; `0` for a fixed-size state.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes_per_token: u64,
    /// Fixed byte count independent of capacity — a recurrent conv/SSM state, e.g. `1409286144`; `0` for a per-token state.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub bytes: u64,
}

fn is_zero(v: &u64) -> bool {
    *v == 0
}

/// A typed tensor, e.g. `{"dtype": "bf16", "shape": ["tokens", 2560], "kind": "workspace"}`.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    /// Element type, e.g. `"bf16"`.
    pub dtype: DType,
    /// Extents, constants or var names, e.g. `["tokens", 2560]`.
    pub shape: Vec<Dim>,
    /// Who provides the buffer and how long its contents live.
    pub kind: BufferKind,
    /// Optional prior on the contents; the runtime rejects out-of-domain input writes and `kern test` synthesizes values from it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<Domain>,
}

/// A prior on a buffer's contents: bounds (`{"min": 0, "max": "tokens"}`) or an index into a buffer/state (`{"index_into": "kv", "stride": 16}`).
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Domain {
    /// Inclusive lower bound, e.g. `0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<Bound>,
    /// Inclusive upper bound, a literal or a var expression, e.g. `"tokens"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<Bound>,
    /// Buffer or state whose rows / token slots the elements index, e.g. `"model.embed_tokens.weight"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_into: Option<String>,
    /// Rows or token slots per index (default `1`), e.g. `16` for a paged KV block table.
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub stride: u64,
    /// Require a non-decreasing sequence, e.g. `cu_seqlens`.
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

/// A domain bound: an integer, a float, or a var expression, e.g. `0`, `1e-6`, `"tokens"`.
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

/// One shape extent: a constant or a var name, e.g. `2560` or `"tokens"`.
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
    /// Written by the runtime before each run, e.g. `token_ids`.
    Input,
    /// Read back by the runtime after each run, e.g. `next_token`.
    Output,
    /// Bound by name from the weights file at load time, e.g. `model.embed_tokens.weight`.
    Weight,
    /// Runtime-owned scratch, dead between runs, e.g. `hidden`.
    Workspace,
    /// Written by one program and read by another, kept between runs, e.g. the `fc_out` hidden states a draft reads.
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

/// Element type: `bf16`, `f16`, `f32`, `fp8e4m3`, `i32`, `u32`, `i64`, `u8`.
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
            "description": "Element type of a buffer or scratch, e.g. `\"bf16\"`.",
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

/// Direction of a pointer param: `in`, `out`, `inout`.
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

/// A by-value scalar param: `i32`, `i64`, `f32`, `u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    I32,
    I64,
    F32,
    /// One-byte scalar, e.g. a bool flag.
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

/// One op or launch parameter, written as a string: `"in buffer<bf16>"`, `"inout state"`, `"i32"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    /// A buffer pointer with its element type and direction, e.g. `"out buffer<bf16>"`.
    Buf { dtype: DType, dir: Dir },
    /// An opaque state base pointer, e.g. `"inout state"`.
    State { dir: Dir },
    /// A by-value scalar, e.g. `"i32"`.
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
            "description": "One parameter: a scalar (`\"i32\"`, `\"i64\"`, `\"f32\"`, `\"u8\"`) \
                or a directional pointer (`\"in buffer<bf16>\"`, `\"out buffer<f32>\"`, \
                `\"inout state\"`).",
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

/// A code artifact, e.g. `{"source": "hf:kernels-community/activation/build/.../_activation.abi3.so", "sha256": "73748b54..."}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Module {
    /// Local file name (`argmax.cubin`) or registry ref (`hf:org/repo/path[@rev]`); a label, not the identity.
    pub source: String,
    /// Hex sha256 of the artifact bytes; the runtime matches modules by this.
    pub sha256: String,
}

/// An operator: the typed interface a call binds, plus the launches that implement it.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Op {
    /// Interface params in call order, e.g. `["out buffer<bf16>", "in buffer<bf16>"]`.
    pub params: Vec<ParamType>,
    /// The implementation; swapping it leaves every call untouched.
    #[serde(rename = "impl")]
    pub imp: Impl,
}

/// An op implementation: private scratch buffers and one or more launches in order.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Impl {
    #[serde(
        default,
        deserialize_with = "unique_map",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    /// Implementation-private buffers, e.g. `{"pmax": {"dtype": "f32", "shape": [1, 64]}}`.
    pub scratch: BTreeMap<String, Scratch>,
    /// Launches in execution order.
    pub launches: Vec<Launch>,
}

/// A private buffer of one implementation, sized like a workspace buffer.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Scratch {
    /// Element type, e.g. `"f32"`.
    pub dtype: DType,
    /// Extents, constants or var names, e.g. `["tokens", 8]`.
    pub shape: Vec<Dim>,
}

/// One launch of an implementation: a kernel entry in a pinned module, or a runtime built-in.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Launch {
    Kernel(KernelLaunch),
    Extern(ExternLaunch),
}

/// A device kernel launch, e.g. `{"module": "argmax", "entry": "kern_argmax_partial_bf16", "block": [1024, 1, 1], "grid": [1, 64, 1]}`.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KernelLaunch {
    /// Name of a `modules` entry, e.g. `"argmax"`.
    pub module: String,
    /// Kernel symbol in the module, e.g. `"kern_argmax_partial_bf16"`.
    pub entry: String,
    /// This launch's own ABI when it differs from the op's params, e.g. `["in buffer<bf16>", "out buffer<f32>", "i32"]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Vec<ParamType>>,
    /// Threads per block, e.g. `[1024, 1, 1]`.
    pub block: [u32; 3],
    /// Blocks per launch, as expressions, e.g. `[{"ceil_div": ["tokens", 128]}, 1, 1]`.
    pub grid: [Expr; 3],
    /// Dynamic shared memory in bytes, e.g. `{"mul": ["tokens", 512]}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mem: Option<Expr>,
    /// Where each launch param comes from (default: the op's params in order), e.g. `[{"param": 0}, {"scratch": "pmax"}, {"i32": 64}]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<LaunchArg>>,
}

/// A runtime built-in launch, e.g. `{"entry": "extern:cublaslt_bf16_tn"}`.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExternLaunch {
    /// `extern:<name>`: `cublaslt_bf16_tn` (C = A·Wᵀ) or `cublaslt_bf16_tn_acc` (C += A·Wᵀ).
    pub entry: String,
    /// This launch's own ABI when it differs from the op's params.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Vec<ParamType>>,
    /// Where each launch param comes from (default: the op's params in order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<LaunchArg>>,
}

impl Launch {
    pub fn is_extern(&self) -> bool {
        matches!(self, Launch::Extern(_))
    }

    pub fn entry(&self) -> &str {
        match self {
            Launch::Kernel(k) => &k.entry,
            Launch::Extern(e) => &e.entry,
        }
    }

    /// The pinned module, for a kernel launch.
    pub fn module(&self) -> Option<&str> {
        match self {
            Launch::Kernel(k) => Some(&k.module),
            Launch::Extern(_) => None,
        }
    }

    pub fn kernel(&self) -> Option<&KernelLaunch> {
        match self {
            Launch::Kernel(k) => Some(k),
            Launch::Extern(_) => None,
        }
    }

    /// The launch ABI, defaulting to the op's interface.
    pub fn params_of<'a>(&'a self, op: &'a Op) -> &'a [ParamType] {
        let own = match self {
            Launch::Kernel(k) => &k.params,
            Launch::Extern(e) => &e.params,
        };
        own.as_deref().unwrap_or(&op.params)
    }

    /// The wiring, defaulting to forwarding the op's params in order.
    pub fn args_of(&self, op: &Op) -> Cow<'_, [LaunchArg]> {
        let own = match self {
            Launch::Kernel(k) => &k.args,
            Launch::Extern(e) => &e.args,
        };
        match own {
            Some(a) => Cow::Borrowed(a),
            None => Cow::Owned((0..op.params.len()).map(|param| LaunchArg::Param { param }).collect()),
        }
    }
}

/// A launch argument: a forwarded op param, a scratch buffer, or a literal, e.g. `{"param": 0}`, `{"scratch": "pmax"}`, `{"i32": 64}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum LaunchArg {
    /// Forward the op's param at this index.
    Param { param: usize },
    /// Pass the named scratch buffer.
    Scratch { scratch: String },
    /// A literal `i32`.
    I32 { i32: i32 },
    /// A literal `i64`.
    I64 { i64: i64 },
    /// A literal `f32`.
    F32 { f32: f32 },
    /// A literal `u8`.
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

/// A `source` of the form `hf:<org>/<repo>/<path>[@<revision>]` (revision defaults to `main`), fetched into a content-addressed cache at load time.
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

/// A scalar expression: a constant, a var name, `{"ceil_div": [e, c]}` or `{"mul": [e, c]}`, e.g. `{"ceil_div": ["tokens", 128]}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum Expr {
    /// A constant, e.g. `64`.
    Const(u64),
    /// A var by name, e.g. `"tokens"`.
    Var(String),
    /// `ceil(e / c)`, e.g. `{"ceil_div": ["tokens", 128]}`.
    CeilDiv { ceil_div: (Box<Expr>, u64) },
    /// `e * c`, e.g. `{"mul": ["tokens", 32]}`.
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

/// One op call, e.g. `{"label": "l0.attn", "op": "attn", "args": [{"buf": "q"}, {"state": "kv"}, ...]}`.
#[derive(Debug, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Call {
    /// Human-readable name for diagnostics, e.g. `"l0.attn"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Name of the op called, e.g. `"attn"`.
    pub op: String,
    /// One argument per interface param, in order.
    pub args: Vec<Arg>,
}

/// A call argument, e.g. `{"buf": "hidden"}`, `{"state": "kv", "offset": 65536}`, `{"var": "tokens"}`, `{"i32": 2560}`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged, deny_unknown_fields)]
pub enum Arg {
    /// A buffer plus an optional byte offset into it, e.g. the v slice of a fused qkv buffer.
    Buf {
        /// Buffer name, e.g. `"qkv"`.
        buf: String,
        /// Byte offset added to the base pointer (default `0`), e.g. `4096`.
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
    },
    /// A state plus an optional byte offset into it, e.g. one layer's region of a KV cache.
    State {
        /// State name, e.g. `"kv"`.
        state: String,
        /// Byte offset added to the base pointer (default `0`), e.g. `65536`.
        #[serde(default, skip_serializing_if = "is_zero")]
        offset: u64,
    },
    /// The current value of a var, e.g. `{"var": "tokens"}`.
    Var { var: String },
    /// The value of a var expression, e.g. `{"expr": {"mul": ["tokens", 32]}}`.
    Expr { expr: Expr },
    /// A literal `i32`.
    I32 { i32: i32 },
    /// A literal `i64`.
    I64 { i64: i64 },
    /// A literal `f32`.
    F32 { f32: f32 },
    /// A literal `u8`.
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
