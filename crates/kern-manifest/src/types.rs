//! Manifest schema. Parsing is already strict: unknown fields, duplicate
//! names and malformed type strings are rejected at deserialization time.
//! Semantic checks (references, dtypes, dataflow, bounds) live in
//! [`crate::verify`].

use serde::de::{Error as DeError, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub meta: Meta,
    #[serde(default, deserialize_with = "unique_map")]
    pub symbols: BTreeMap<String, Symbol>,
    #[serde(default, deserialize_with = "unique_map")]
    pub states: BTreeMap<String, State>,
    #[serde(deserialize_with = "unique_map")]
    pub buffers: BTreeMap<String, Buffer>,
    #[serde(deserialize_with = "unique_map")]
    pub kernels: BTreeMap<String, Kernel>,
    #[serde(deserialize_with = "unique_map")]
    pub programs: BTreeMap<String, Program>,
}

impl Manifest {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("manifest serialization cannot fail")
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Meta {
    /// Manifest format version. Only 2 is accepted.
    pub version: u32,
    /// Free-form label; the runtime assigns no meaning to it.
    pub model: String,
}

/// A runtime-provided scalar (e.g. token count this step). The declared
/// bounds are what verification is performed against; the runtime rejects
/// out-of-bounds values at dispatch time.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Symbol {
    pub max: u64,
    #[serde(default = "default_symbol_min")]
    pub min: u64,
}

fn default_symbol_min() -> u64 {
    1
}

/// Opaque persistent state. The runtime knows only how many bytes to
/// provision per token; the internal layout belongs to the provider's
/// kernels, which receive the base pointer as an untyped `ptr` param.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub bytes_per_token: u64,
    #[serde(default = "default_align")]
    pub align: u64,
}

fn default_align() -> u64 {
    256
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer {
    pub dtype: DType,
    pub shape: Vec<Dim>,
    pub class: BufferClass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    Const(u64),
    Sym(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BufferClass {
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

impl fmt::Display for BufferClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BufferClass::Input => "input",
            BufferClass::Output => "output",
            BufferClass::Weight => "weight",
            BufferClass::Workspace => "workspace",
            BufferClass::Carry => "carry",
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
    U32,
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
            ScalarType::U32 => "u32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
            ScalarType::U8 => "u8",
        })
    }
}

/// One kernel parameter, written as a string in the manifest:
/// `"in buffer<bf16>"`, `"out buffer<fp8e4m3>"`, `"inout ptr"`, `"i32"`.
/// Buffers and ptrs require an explicit direction; scalars are by-value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamType {
    Buf { dtype: DType, dir: Dir },
    Ptr { dir: Dir },
    Scalar(ScalarType),
}

impl ParamType {
    /// Size of the param slot in the kernel ABI, for cross-checking against
    /// `cuKernelGetParamInfo` when the cubin is loaded.
    pub fn size_bytes(self) -> u64 {
        match self {
            ParamType::Buf { .. } | ParamType::Ptr { .. } => 8,
            ParamType::Scalar(ScalarType::I64) => 8,
            ParamType::Scalar(ScalarType::U8) => 1,
            ParamType::Scalar(_) => 4,
        }
    }
}

impl FromStr for ParamType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let s = s.trim();
        match s {
            "i32" => return Ok(ParamType::Scalar(ScalarType::I32)),
            "u32" => return Ok(ParamType::Scalar(ScalarType::U32)),
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
        if rest == "ptr" {
            return Ok(ParamType::Ptr { dir });
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
            ParamType::Ptr { dir } => write!(f, "{dir} ptr"),
            ParamType::Scalar(st) => write!(f, "{st}"),
        }
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

/// A kernel is an **interface** (the typed params a dispatch passes — the
/// only thing a call site knows) plus an **implementation**: how those args
/// are lowered onto actual launches. Swapping the implementation — a faster
/// kernel from elsewhere with the same interface — touches only `impl`;
/// every dispatch stays untouched.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    /// The interface: typed, directional params. Call-site semantics
    /// (what each position means) are the contract implementations must
    /// honor; the verifier can only check shape, not meaning.
    pub params: Vec<ParamType>,
    #[serde(rename = "impl")]
    pub imp: Impl,
}

/// An implementation is a micro-program: private scratch buffers (sized by
/// expressions over the interface's symbols, provisioned by the runtime,
/// dead outside one dispatch) and one or more launch steps. Launch geometry
/// lives here, not at the call site — it belongs to the implementation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Impl {
    #[serde(
        default,
        deserialize_with = "unique_map",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub scratch: BTreeMap<String, Scratch>,
    pub steps: Vec<Step>,
}

/// One scratch buffer declaration: like a workspace buffer, but private to
/// the implementation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scratch {
    pub dtype: DType,
    pub shape: Vec<Dim>,
}

/// One launch inside an implementation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    /// Cubin file (relative to the kernel artifact dir) this step's symbol
    /// must come from. Absent: the runtime searches every loaded module
    /// (disambiguating by param layout). `extern:` symbols have no cubin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cubin: Option<String>,
    /// sha256 of that cubin file, checked by the runtime when both are set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    /// Entry symbol, or `extern:<op>` for runtime built-ins.
    pub symbol: String,
    /// This step's own launch ABI (what the cubin function actually takes).
    pub params: Vec<ParamType>,
    pub block: [u32; 3],
    pub grid: [Expr; 3],
    /// Dynamic shared memory in bytes, if the step needs any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mem: Option<Expr>,
    /// Wiring: where each step param comes from — a forwarded interface
    /// arg, a scratch buffer, or an implementation-private literal.
    pub args: Vec<StepArg>,
}

/// A step argument. `{"arg": 0}` forwards the dispatch's arg #0 verbatim;
/// `{"scratch": "pmax"}` passes a private scratch pointer (optional byte
/// `offset`); scalar literals are implementation constants the interface
/// never sees (e.g. a partial-count baked into a two-stage reduction).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StepArg {
    Arg {
        arg: usize,
    },
    Scratch {
        scratch: String,
        #[serde(default, skip_serializing_if = "offset_is_zero")]
        offset: u64,
    },
    I32 { i32: i32 },
    U32 { u32: u32 },
    I64 { i64: i64 },
    F32 { f32: f32 },
    U8 { u8: u8 },
}

impl fmt::Display for StepArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StepArg::Arg { arg } => write!(f, "interface arg #{arg}"),
            StepArg::Scratch { scratch, offset: 0 } => write!(f, "scratch `{scratch}`"),
            StepArg::Scratch { scratch, offset } => write!(f, "scratch `{scratch}`+{offset}"),
            StepArg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            StepArg::U32 { u32: v } => write!(f, "u32 literal {v}"),
            StepArg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            StepArg::F32 { f32: v } => write!(f, "f32 literal {v}"),
            StepArg::U8 { u8: v } => write!(f, "u8 literal {v}"),
        }
    }
}

/// A remote cubin reference in a step's `cubin` field:
/// `hf:<org>/<repo>/<path>[@<revision>]` (revision defaults to `main`).
/// The runtime materializes it into a content-addressed local cache at load
/// time; the step's `sha256` — mandatory for registry refs — is the artifact's
/// identity, so the transport needs no trust. Names are just URLs.
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

/// The closed scalar-expression set. This is deliberately not a language:
/// grid geometry and dynamic shared memory are the only runtime-dependent
/// numbers a provider may compute, and only with these forms.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Expr {
    Const(u64),
    Sym { sym: String },
    CeilDiv { ceil_div: (Box<Expr>, u64) },
    Mul { mul: (Box<Expr>, u64) },
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EvalError {
    #[error("unknown symbol `{0}`")]
    UnknownSymbol(String),
    #[error("arithmetic overflow")]
    Overflow,
    #[error("division by zero")]
    DivByZero,
}

impl Expr {
    pub fn eval(&self, env: &BTreeMap<String, u64>) -> Result<u64, EvalError> {
        match self {
            Expr::Const(c) => Ok(*c),
            Expr::Sym { sym } => env
                .get(sym)
                .copied()
                .ok_or_else(|| EvalError::UnknownSymbol(sym.clone())),
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    pub dispatches: Vec<Dispatch>,
}

/// A call site: which kernel interface, with which args. No geometry —
/// grid/block belong to the kernel's implementation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dispatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub kernel: String,
    pub args: Vec<Arg>,
}

/// A dispatch argument. Explicitly tagged so a buffer name can never be
/// mistaken for a symbol name: `{"buf": "hidden"}`, `{"state": "kv"}`,
/// `{"sym": "tokens"}`, `{"i32": 2560}`, `{"i64": 4096}`, `{"f32": 1e-6}`.
///
/// Buffer and state args carry an optional byte `offset` (default 0): the
/// kernel receives base + offset. This is how a provider addresses a view
/// inside a fused buffer (q/k/v slices of a merged qkv projection) or a
/// per-layer region inside an opaque state — the offset is a literal from
/// the provider's own layout arithmetic, the runtime just adds it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Arg {
    Buf {
        buf: String,
        #[serde(default, skip_serializing_if = "offset_is_zero")]
        offset: u64,
    },
    State {
        state: String,
        #[serde(default, skip_serializing_if = "offset_is_zero")]
        offset: u64,
    },
    Sym { sym: String },
    /// Symbol-derived integer scalar, e.g. `{"expr": {"mul": ["tokens", 32]}}`.
    /// Same closed expression language as launch grids; f32 params cannot
    /// bind one.
    Expr { expr: Expr },
    I32 { i32: i32 },
    U32 { u32: u32 },
    I64 { i64: i64 },
    F32 { f32: f32 },
    U8 { u8: u8 },
}

fn offset_is_zero(v: &u64) -> bool {
    *v == 0
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Buf { buf, offset: 0 } => write!(f, "buffer `{buf}`"),
            Arg::Buf { buf, offset } => write!(f, "buffer `{buf}`+{offset}"),
            Arg::State { state, offset: 0 } => write!(f, "state `{state}`"),
            Arg::State { state, offset } => write!(f, "state `{state}`+{offset}"),
            Arg::Sym { sym } => write!(f, "symbol `{sym}`"),
            Arg::Expr { .. } => write!(f, "expression"),
            Arg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            Arg::U32 { u32: v } => write!(f, "u32 literal {v}"),
            Arg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            Arg::F32 { f32: v } => write!(f, "f32 literal {v}"),
            Arg::U8 { u8: v } => write!(f, "u8 literal {v}"),
        }
    }
}
