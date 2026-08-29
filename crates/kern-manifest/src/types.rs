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
    /// Manifest format version. Only 1 is accepted.
    pub version: u32,
    /// Free-form label; the runtime assigns no meaning to it.
    pub model: String,
    pub cubin: Artifact,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub file: String,
    pub sha256: String,
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
}

impl fmt::Display for BufferClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BufferClass::Input => "input",
            BufferClass::Output => "output",
            BufferClass::Weight => "weight",
            BufferClass::Workspace => "workspace",
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
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ScalarType::I32 => "i32",
            ScalarType::U32 => "u32",
            ScalarType::I64 => "i64",
            ScalarType::F32 => "f32",
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    /// Entry symbol inside the cubin.
    pub symbol: String,
    pub params: Vec<ParamType>,
    pub block: [u32; 3],
    /// Dynamic shared memory in bytes, if the kernel needs any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_mem: Option<Expr>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    UnknownSymbol(String),
    Overflow,
    DivByZero,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::UnknownSymbol(s) => write!(f, "unknown symbol `{s}`"),
            EvalError::Overflow => f.write_str("arithmetic overflow"),
            EvalError::DivByZero => f.write_str("division by zero"),
        }
    }
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

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dispatch {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub kernel: String,
    pub grid: [Expr; 3],
    pub args: Vec<Arg>,
}

/// A dispatch argument. Explicitly tagged so a buffer name can never be
/// mistaken for a symbol name: `{"buf": "hidden"}`, `{"state": "kv"}`,
/// `{"sym": "tokens"}`, `{"i32": 2560}`, `{"i64": 4096}`, `{"f32": 1e-6}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Arg {
    Buf { buf: String },
    State { state: String },
    Sym { sym: String },
    I32 { i32: i32 },
    U32 { u32: u32 },
    I64 { i64: i64 },
    F32 { f32: f32 },
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arg::Buf { buf } => write!(f, "buffer `{buf}`"),
            Arg::State { state } => write!(f, "state `{state}`"),
            Arg::Sym { sym } => write!(f, "symbol `{sym}`"),
            Arg::I32 { i32: v } => write!(f, "i32 literal {v}"),
            Arg::U32 { u32: v } => write!(f, "u32 literal {v}"),
            Arg::I64 { i64: v } => write!(f, "i64 literal {v}"),
            Arg::F32 { f32: v } => write!(f, "f32 literal {v}"),
        }
    }
}
