//! Load-time lowering. Everything name-shaped in a program dies here:
//! op launches resolve to CUDA functions, buffer/state/scratch references
//! to device addresses (static once allocated), var names to indices into
//! the dense env. What execution replays is a flat launch list whose slots
//! are either finished values or var-indexed expressions — no name lookups,
//! no wiring, and no panics left for the hot path.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{result as cu, sys, CudaStream};
use kern_manifest::types::{Arg, Call, Dim, Expr, LaunchArg, Manifest, Op, ParamType};

use crate::cubin::{param_sizes, LoadedModule};
use crate::device::{alloc, DeviceBuf};
use crate::error::{bail, cuda_check, Error, Result};

/// A scalar expression with var names resolved to indices into the dense
/// env (manifest var order). Division by zero is rejected at compile
/// time; overflow stays a runtime error (value-dependent).
#[derive(Clone)]
pub(crate) enum CExpr {
    Const(u64),
    Var(usize),
    CeilDiv(Box<CExpr>, u64),
    Mul(Box<CExpr>, u64),
}

impl CExpr {
    pub(crate) fn eval(&self, env: &[u64]) -> Result<u64> {
        match self {
            CExpr::Const(c) => Ok(*c),
            CExpr::Var(i) => Ok(env[*i]),
            CExpr::CeilDiv(e, c) => Ok(e.eval(env)?.checked_add(c - 1).ok_or_else(overflow)? / c),
            CExpr::Mul(e, c) => e.eval(env)?.checked_mul(*c).ok_or_else(overflow),
        }
    }
}

fn overflow() -> Error {
    Error::Manifest("expression eval: arithmetic overflow".into())
}

/// A launch value: the low bytes of `val` are what the param slot receives;
/// `bytes` is the remaining buffer size for pointer args (0 for scalars),
/// used by the extern gemm.
#[derive(Clone, Copy)]
pub(crate) struct RVal {
    pub(crate) val: u64,
    pub(crate) bytes: u64,
}

/// One launch parameter, lowered.
#[derive(Clone)]
pub(crate) enum Slot {
    /// Known at load time: a device pointer (base + offset) or a literal.
    Const(RVal),
    /// Var-dependent scalar, evaluated against the dense env per run.
    Expr(CExpr),
}

pub(crate) enum LaunchKind {
    Cubin {
        func: sys::CUfunction,
        block: [u32; 3],
        grid: [CExpr; 3],
        shared_mem: Option<CExpr>,
    },
    /// `extern:cublaslt_bf16_tn` / `..._acc` (beta 0.0 / 1.0).
    Gemm { beta: f32 },
}

pub(crate) struct Launch {
    pub(crate) kind: LaunchKind,
    pub(crate) slots: Vec<Slot>,
    /// Error context: which call and impl launch this came from.
    pub(crate) ctx: String,
}

pub(crate) struct CompiledProgram {
    pub(crate) launches: Vec<Launch>,
    /// Launch index range `[lo, hi)` of every call, in call order (a
    /// multi-launch impl contributes several launches).
    pub(crate) call_ranges: Vec<(usize, usize)>,
}

/// One launch of an op implementation, resolved against the loaded modules.
enum LaunchImpl {
    Cubin { func: sys::CUfunction, module: String },
    GemmBf16Tn { beta: f32 },
}

/// An op implementation, resolved: one entry per launch, plus the private
/// scratch buffers the impl declared (allocated once at var max, reused
/// every call — contents are dead outside a single call).
pub(crate) struct ResolvedOp {
    launches: Vec<LaunchImpl>,
    pub(crate) scratch: BTreeMap<String, DeviceBuf>,
}

impl ResolvedOp {
    /// The module each launch resolved to, in launch order (introspection).
    pub(crate) fn launch_modules(&self) -> Vec<String> {
        self.launches
            .iter()
            .map(|s| match s {
                LaunchImpl::Cubin { module, .. } => module.clone(),
                LaunchImpl::GemmBf16Tn { .. } => "runtime built-in (cublasLt)".into(),
            })
            .collect()
    }
}

/// Byte size of a shaped declaration at var upper bounds.
pub(crate) fn shaped_bytes(
    what: &str,
    shape: &[Dim],
    dtype_bytes: u64,
    max_env: &BTreeMap<String, u64>,
) -> Result<u64> {
    let mut elems = 1u64;
    for d in shape {
        let n = match d {
            Dim::Const(c) => *c,
            Dim::Var(s) => *max_env
                .get(s)
                .ok_or_else(|| Error::Manifest(format!("{what}: unknown var `{s}` in shape")))?,
        };
        elems = elems
            .checked_mul(n)
            .ok_or_else(|| Error::Manifest(format!("{what}: size overflow")))?;
    }
    elems
        .checked_mul(dtype_bytes)
        .ok_or_else(|| Error::Manifest(format!("{what}: size overflow")))
}

/// Resolve every op: match each launch's entry + declared param layout
/// against the loaded modules (or an extern built-in), pin the module's
/// hash, opt into >48KB dynamic shared memory, allocate scratch.
pub(crate) fn resolve_ops(
    manifest: &Manifest,
    modules: &[LoadedModule],
    kernels_dir: &Path,
    stream: &Arc<CudaStream>,
    max_env: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, ResolvedOp>> {
    let mut ops = BTreeMap::new();
    for (name, op) in &manifest.ops {
        let mut launches = Vec::new();
        for (li, l) in op.imp.launches.iter().enumerate() {
            if let Some(ext) = l.entry.strip_prefix("extern:") {
                match ext {
                    "cublaslt_bf16_tn" => launches.push(LaunchImpl::GemmBf16Tn { beta: 0.0 }),
                    "cublaslt_bf16_tn_acc" => launches.push(LaunchImpl::GemmBf16Tn { beta: 1.0 }),
                    _ => bail!(Manifest, "op `{name}` launch #{li}: unsupported extern `{ext}`"),
                }
                continue;
            }
            // Identity: a launch that names its module pins that module's
            // sha256 (the verifier resolved the name); the source is a
            // label. Only loaded modules with that hash are candidates.
            // Unpinned launches search every loaded module and
            // disambiguate by param layout.
            let pinned = l.module.as_deref().and_then(|m| manifest.modules.get(m).map(|md| (m, md)));
            let want_sha = pinned.map(|(_, md)| md.sha256.to_lowercase());
            if let (Some((mname, md)), Some(sha)) = (pinned, &want_sha) {
                if !modules.iter().any(|m| &m.sha == sha) {
                    bail!(
                        KernelArtifact,
                        "op `{name}` launch #{li}: module `{mname}` ({} @{}) is not among the {} \
                         artifacts loaded from {} — the source is a label, the hash is the identity; \
                         put an artifact with that sha256 there (tools/extract_kernels.sh <manifest> \
                         <dump dirs> {})",
                        md.source,
                        &sha[..12.min(sha.len())],
                        modules.len(),
                        kernels_dir.display(),
                        kernels_dir.display()
                    );
                }
            }
            let want: Vec<usize> = l.params_of(op).iter().map(|p| p.size_bytes() as usize).collect();
            let entry = CString::new(l.entry.as_str())
                .map_err(|e| Error::Manifest(format!("op `{name}` entry: {e}")))?;
            let mut resolved = None;
            let mut seen = Vec::new();
            for m in modules {
                if let Some(sha) = &want_sha {
                    if &m.sha != sha {
                        continue;
                    }
                }
                let Ok(func) = (unsafe { cu::module::get_function(m.module, entry.clone()) }) else {
                    continue;
                };
                let got = param_sizes(func)?;
                if got == want {
                    resolved = Some(LaunchImpl::Cubin { func, module: format!("{}@{}", m.label, &m.sha[..8]) });
                    break;
                }
                seen.push(format!("{}@{}: {got:?}", m.label, &m.sha[..8]));
            }
            let Some(r) = resolved else {
                bail!(
                    KernelArtifact,
                    "op `{name}` launch #{li} ({}): no loaded instance matches declared \
                     param layout {want:?} (module {:?} sha {:?}); saw {seen:?}",
                    l.entry,
                    l.module,
                    want_sha.as_deref().map(|s| &s[..12.min(s.len())])
                );
            };
            // Opt in to >48KB dynamic shared memory where the launch needs it.
            if let (LaunchImpl::Cubin { func, .. }, Some(sm)) = (&r, &l.shared_mem) {
                let bytes = sm
                    .eval(max_env)
                    .map_err(|e| Error::Manifest(format!("op `{name}`: {e}")))?;
                if bytes > 48 * 1024 {
                    cuda_check(
                        unsafe {
                            sys::cuFuncSetAttribute(
                                *func,
                                sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                bytes as i32,
                            )
                        },
                        "cuFuncSetAttribute",
                    )?;
                }
            }
            launches.push(r);
        }
        let mut scratch = BTreeMap::new();
        for (sname, sd) in &op.imp.scratch {
            let bytes = shaped_bytes(
                &format!("op `{name}` scratch `{sname}`"),
                &sd.shape,
                sd.dtype.bytes(),
                max_env,
            )?;
            scratch.insert(sname.clone(), alloc(stream, bytes)?);
        }
        ops.insert(name.clone(), ResolvedOp { launches, scratch });
    }
    Ok(ops)
}

/// Lower every program's call list into a flat launch list.
pub(crate) fn compile_programs(
    manifest: &Manifest,
    ops: &BTreeMap<String, ResolvedOp>,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
) -> Result<BTreeMap<String, CompiledProgram>> {
    let vars: BTreeMap<&str, usize> =
        manifest.vars.keys().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let mut programs = BTreeMap::new();
    for (pname, calls) in &manifest.programs {
        let mut launches = Vec::new();
        let mut call_ranges = Vec::with_capacity(calls.len());
        for (ci, c) in calls.iter().enumerate() {
            let cctx = call_ctx(ci, c);
            let (Some(op), Some(rop)) = (manifest.ops.get(&c.op), ops.get(&c.op)) else {
                bail!(Manifest, "program `{pname}` {cctx}: unknown op");
            };
            let lo = launches.len();
            compile_call(c, op, rop, &cctx, buffers, states, &vars, &mut launches).map_err(|e| {
                Error::Call { context: format!("program `{pname}` {cctx}"), source: Box::new(e) }
            })?;
            call_ranges.push((lo, launches.len()));
        }
        programs.insert(pname.clone(), CompiledProgram { launches, call_ranges });
    }
    Ok(programs)
}

/// Error context locating one entry of a program's call list.
fn call_ctx(i: usize, c: &Call) -> String {
    match &c.label {
        Some(l) => format!("call #{i} `{l}` (op `{}`)", c.op),
        None => format!("call #{i} (op `{}`)", c.op),
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_call(
    c: &Call,
    op: &Op,
    rop: &ResolvedOp,
    cctx: &str,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
    vars: &BTreeMap<&str, usize>,
    launches: &mut Vec<Launch>,
) -> Result<()> {
    if c.args.len() != op.params.len() {
        bail!(Manifest, "op takes {} args, call passes {}", op.params.len(), c.args.len());
    }
    // Lower the interface args once; each launch then wires its own params
    // from these, its scratch, and its private literals.
    let mut vals = Vec::with_capacity(c.args.len());
    for (arg, pty) in c.args.iter().zip(&op.params) {
        vals.push(match pty {
            ParamType::Buf { .. } | ParamType::State { .. } => {
                Slot::Const(pointer_arg(arg, buffers, states)?)
            }
            ParamType::Scalar(_) => scalar_arg(arg, vars)?,
        });
    }
    for (li, (l, imp)) in op.imp.launches.iter().zip(&rop.launches).enumerate() {
        let wiring = l.args_of(op);
        let mut slots = Vec::with_capacity(wiring.len());
        for la in wiring.iter() {
            slots.push(match la {
                LaunchArg::Param { param } => vals.get(*param).cloned().ok_or_else(|| {
                    Error::Manifest(format!("launch #{li}: forwarded param #{param} out of range"))
                })?,
                LaunchArg::Scratch { scratch } => {
                    let Some(b) = rop.scratch.get(scratch) else {
                        bail!(Manifest, "launch #{li}: unknown scratch `{scratch}`");
                    };
                    Slot::Const(RVal { val: b.ptr, bytes: b.bytes })
                }
                LaunchArg::I32 { i32: v } => lit(*v as u32 as u64),
                LaunchArg::I64 { i64: v } => lit(*v as u64),
                LaunchArg::U8 { u8: v } => lit(*v as u64),
                LaunchArg::F32 { f32: v } => lit(v.to_bits() as u64),
            });
        }
        let kind = match imp {
            LaunchImpl::GemmBf16Tn { beta } => {
                if slots.len() != 6 {
                    bail!(Manifest, "launch #{li}: extern gemm takes 6 args, got {}", slots.len());
                }
                LaunchKind::Gemm { beta: *beta }
            }
            LaunchImpl::Cubin { func, .. } => {
                let (Some(block), Some(grid)) = (l.block, &l.grid) else {
                    bail!(Manifest, "launch #{li}: missing block/grid");
                };
                LaunchKind::Cubin {
                    func: *func,
                    block,
                    grid: [
                        compile_expr(&grid[0], vars)?,
                        compile_expr(&grid[1], vars)?,
                        compile_expr(&grid[2], vars)?,
                    ],
                    shared_mem: l.shared_mem.as_ref().map(|e| compile_expr(e, vars)).transpose()?,
                }
            }
        };
        launches.push(Launch { kind, slots, ctx: format!("{cctx} launch #{li} (`{}`)", l.entry) });
    }
    Ok(())
}

/// Lower a buffer/state arg to its finished pointer value.
fn pointer_arg(
    arg: &Arg,
    buffers: &BTreeMap<String, DeviceBuf>,
    states: &BTreeMap<String, DeviceBuf>,
) -> Result<RVal> {
    let (map, name, offset, what) = match arg {
        Arg::Buf { buf, offset } => (buffers, buf, *offset, "buffer"),
        Arg::State { state, offset } => (states, state, *offset, "state"),
        _ => bail!(Manifest, "expected buffer/state arg, got {arg}"),
    };
    let Some(b) = map.get(name) else {
        bail!(Manifest, "unknown {what} `{name}`");
    };
    offset_into(b, offset, || format!("{what} `{name}`"))
}

fn offset_into(b: &DeviceBuf, offset: u64, what: impl Fn() -> String) -> Result<RVal> {
    let Some(bytes) = b.bytes.checked_sub(offset) else {
        bail!(Manifest, "offset {offset} outside {} ({} bytes)", what(), b.bytes);
    };
    Ok(RVal { val: b.ptr + offset, bytes })
}

/// Lower a scalar arg: literals finish now, vars and expressions become
/// dense-indexed expressions.
fn scalar_arg(arg: &Arg, vars: &BTreeMap<&str, usize>) -> Result<Slot> {
    Ok(match arg {
        Arg::Var { var } => Slot::Expr(CExpr::Var(var_index(vars, var)?)),
        Arg::Expr { expr } => Slot::Expr(compile_expr(expr, vars)?),
        Arg::I32 { i32: v } => lit(*v as u32 as u64),
        Arg::I64 { i64: v } => lit(*v as u64),
        Arg::U8 { u8: v } => lit(*v as u64),
        Arg::F32 { f32: v } => lit(v.to_bits() as u64),
        Arg::Buf { .. } | Arg::State { .. } => bail!(Manifest, "expected scalar arg, got {arg}"),
    })
}

fn lit(val: u64) -> Slot {
    Slot::Const(RVal { val, bytes: 0 })
}

fn compile_expr(e: &Expr, vars: &BTreeMap<&str, usize>) -> Result<CExpr> {
    Ok(match e {
        Expr::Const(c) => CExpr::Const(*c),
        Expr::Var(var) => CExpr::Var(var_index(vars, var)?),
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            if *c == 0 {
                bail!(Manifest, "expression: division by zero");
            }
            CExpr::CeilDiv(Box::new(compile_expr(inner, vars)?), *c)
        }
        Expr::Mul { mul: (inner, c) } => CExpr::Mul(Box::new(compile_expr(inner, vars)?), *c),
    })
}

fn var_index(vars: &BTreeMap<&str, usize>, var: &str) -> Result<usize> {
    match vars.get(var) {
        Some(&i) => Ok(i),
        None => bail!(Manifest, "unknown var `{var}`"),
    }
}
