//! Load-time verification: refuse any manifest that is not provably
//! self-consistent, before anything touches the GPU. All errors are
//! collected and reported together, rustc-style.
//!
//! Checks:
//!   1. meta: format version
//!   2. symbols: bounds sane
//!   3. states: non-zero size, power-of-two alignment
//!   4. buffers: shapes resolve, byte sizes don't overflow at symbol upper
//!      bounds
//!   5. kernels (interface + implementation):
//!      - scratch shapes resolve, every scratch is used
//!      - per step: CUDA block/grid limits at symbol upper bounds (and
//!        grid non-zero at lower bounds), shared-mem opt-in cap, arg/param
//!        arity, per-position wiring types (forwarded interface arg must
//!        match kind/dtype, a step may not write through an interface `in`
//!        param, scratch offsets aligned and in bounds, literal types)
//!      - impl dataflow: scratch never read before a step wrote it; every
//!        interface `out` param written by some step
//!   6. dispatches: kernel refs resolve, arg/param arity and per-position
//!      type match against the interface, symbol ranges fit scalar params
//!   7. dataflow per program: no read-before-write, no writes to input or
//!      weight buffers, every output buffer written
//!   8. no unused declarations
//!
//! What this deliberately cannot check: kernel *behavior*, and the
//! *semantics* of interface params (that a replacement implementation
//! interprets position #3 as the same row stride). A cubin that lies about
//! what it touches is inside the trust boundary; the manifest only makes
//! the lie explicit and diffable. Cross-checking step param layouts against
//! `cuFuncGetParamInfo` is a load-time (phase 2) concern in the runtime
//! crate, since it needs the CUDA driver.

use crate::types::*;
use std::collections::{BTreeMap, BTreeSet};

const MAX_GRID_X: u64 = (1 << 31) - 1;
const MAX_GRID_YZ: u64 = 65_535;
const MAX_BLOCK_THREADS: u64 = 1024;
const MAX_BLOCK_Z: u32 = 64;
/// Per-block dynamic shared memory after the `cuFuncSetAttribute` opt-in the
/// runtime performs for any step declaring `shared_mem`: 227 KiB on
/// sm90/sm100/sm103 datacenter parts.
const MAX_DYN_SHARED_MEM: u64 = 232_448;

/// Sized like buffers: dtype bytes x dims at symbol upper bounds.
fn shaped_size(
    what: &str,
    dtype: DType,
    shape: &[Dim],
    env_max: &BTreeMap<String, u64>,
    used_syms: &mut BTreeSet<String>,
    errs: &mut Vec<String>,
) -> Option<u64> {
    if shape.is_empty() {
        errs.push(format!("{what}: shape must not be empty"));
        return None;
    }
    let mut dim_err = false;
    let mut size: Option<u64> = Some(dtype.bytes());
    for dim in shape {
        let extent = match dim {
            Dim::Const(0) => {
                errs.push(format!("{what}: zero-sized dimension"));
                dim_err = true;
                None
            }
            Dim::Const(c) => Some(*c),
            Dim::Sym(s) => match env_max.get(s) {
                Some(mx) => {
                    used_syms.insert(s.clone());
                    Some(*mx)
                }
                None => {
                    errs.push(format!("{what}: unknown symbol `{s}` in shape"));
                    dim_err = true;
                    None
                }
            },
        };
        size = match (size, extent) {
            (Some(a), Some(e)) => a.checked_mul(e),
            _ => None,
        };
    }
    if size.is_none() && !dim_err {
        errs.push(format!("{what}: byte size overflows u64 at symbol upper bounds"));
    }
    size
}

/// Every diagnostic [`verify`] collected, reported together rustc-style.
/// Derefs to the individual messages.
#[derive(Debug)]
pub struct VerifyErrors(pub Vec<String>);

impl std::fmt::Display for VerifyErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("manifest failed verification:")?;
        for e in &self.0 {
            write!(f, "\n  - {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for VerifyErrors {}

impl std::ops::Deref for VerifyErrors {
    type Target = [String];
    fn deref(&self) -> &[String] {
        &self.0
    }
}

pub fn verify(m: &Manifest) -> Result<(), VerifyErrors> {
    let mut errs: Vec<String> = Vec::new();
    let mut used_syms: BTreeSet<String> = BTreeSet::new();
    let mut used_buffers: BTreeSet<String> = BTreeSet::new();
    let mut used_states: BTreeSet<String> = BTreeSet::new();
    let mut used_kernels: BTreeSet<String> = BTreeSet::new();

    // 1. meta
    if m.meta.version != 2 {
        errs.push(format!("meta: unsupported manifest version {}", m.meta.version));
    }

    // 2. symbols
    for (name, s) in &m.symbols {
        if s.max == 0 {
            errs.push(format!("symbol `{name}`: max must be > 0"));
        }
        if s.min > s.max {
            errs.push(format!("symbol `{name}`: min {} > max {}", s.min, s.max));
        }
    }
    let env_max: BTreeMap<String, u64> =
        m.symbols.iter().map(|(k, v)| (k.clone(), v.max)).collect();
    let env_min: BTreeMap<String, u64> =
        m.symbols.iter().map(|(k, v)| (k.clone(), v.min)).collect();

    // 3. states
    for (name, st) in &m.states {
        if st.bytes_per_token == 0 {
            errs.push(format!("state `{name}`: bytes_per_token must be > 0"));
        }
        if !st.align.is_power_of_two() {
            errs.push(format!("state `{name}`: align {} is not a power of two", st.align));
        }
    }

    // 4. buffers
    let mut buf_sizes: BTreeMap<&str, u64> = BTreeMap::new();
    for (name, b) in &m.buffers {
        if let Some(sz) = shaped_size(
            &format!("buffer `{name}`"),
            b.dtype,
            &b.shape,
            &env_max,
            &mut used_syms,
            &mut errs,
        ) {
            buf_sizes.insert(name, sz);
        }
    }

    // 5. kernels: interface + implementation
    for (kname, k) in &m.kernels {
        let imp = &k.imp;

        let mut scratch_sizes: BTreeMap<&str, u64> = BTreeMap::new();
        for (sname, s) in &imp.scratch {
            if let Some(sz) = shaped_size(
                &format!("kernel `{kname}` scratch `{sname}`"),
                s.dtype,
                &s.shape,
                &env_max,
                &mut used_syms,
                &mut errs,
            ) {
                scratch_sizes.insert(sname, sz);
            }
        }

        if imp.steps.is_empty() {
            errs.push(format!("kernel `{kname}`: implementation has no steps"));
        }

        // Impl-level dataflow over slots: interface `out` params and scratch
        // start unwritten; `in`/`inout` interface params are caller-provided.
        let mut iface_written: Vec<bool> = k
            .params
            .iter()
            .map(|p| !matches!(p, ParamType::Buf { dir: Dir::Out, .. } | ParamType::Ptr { dir: Dir::Out }))
            .collect();
        let mut scratch_written: BTreeSet<&str> = BTreeSet::new();
        let mut scratch_used: BTreeSet<&str> = BTreeSet::new();

        for (si, step) in imp.steps.iter().enumerate() {
            let ctx = format!("kernel `{kname}` step #{si} ({})", step.symbol);
            if step.symbol.is_empty() {
                errs.push(format!("{ctx}: empty entry symbol"));
            }
            if let Some(sha) = &step.sha256 {
                if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                    errs.push(format!("{ctx}: sha256 `{sha}` is not 64 hex chars"));
                }
                if step.cubin.is_none() {
                    errs.push(format!("{ctx}: sha256 without a cubin file"));
                }
            }
            if step.cubin.as_deref() == Some("") {
                errs.push(format!("{ctx}: empty cubin file name"));
            }

            let threads: u64 = step.block.iter().map(|&x| x as u64).product();
            if step.block.contains(&0) || threads > MAX_BLOCK_THREADS {
                errs.push(format!(
                    "{ctx}: block {:?} exceeds {MAX_BLOCK_THREADS} threads or has a zero dim",
                    step.block
                ));
            }
            if step.block[2] > MAX_BLOCK_Z {
                errs.push(format!("{ctx}: block.z {} > {MAX_BLOCK_Z}", step.block[2]));
            }
            for (axis, e) in ["x", "y", "z"].iter().zip(&step.grid) {
                let ectx = format!("{ctx}: grid.{axis}");
                check_expr(e, m, &mut used_syms, &mut errs, &ectx);
                match e.eval(&env_max) {
                    Ok(v) => {
                        let limit = if *axis == "x" { MAX_GRID_X } else { MAX_GRID_YZ };
                        if v > limit {
                            errs.push(format!(
                                "{ectx}: {v} exceeds CUDA limit {limit} at symbol upper bounds"
                            ));
                        }
                    }
                    Err(err) => errs.push(format!("{ectx}: {err}")),
                }
                if let Ok(v) = e.eval(&env_min) {
                    if v == 0 {
                        errs.push(format!("{ectx}: evaluates to 0 at symbol lower bounds"));
                    }
                }
            }
            if let Some(e) = &step.shared_mem {
                let ectx = format!("{ctx}: shared_mem");
                check_expr(e, m, &mut used_syms, &mut errs, &ectx);
                if let Ok(v) = e.eval(&env_max) {
                    if v > MAX_DYN_SHARED_MEM {
                        errs.push(format!(
                            "{ectx}: {v} bytes exceeds opt-in limit {MAX_DYN_SHARED_MEM} at symbol upper bounds"
                        ));
                    }
                }
            }

            if step.args.len() != step.params.len() {
                errs.push(format!(
                    "{ctx}: takes {} params, got {} args",
                    step.params.len(),
                    step.args.len()
                ));
                continue;
            }
            for (j, (arg, param)) in step.args.iter().zip(&step.params).enumerate() {
                let actx = format!("{ctx}: arg #{j}");
                match arg {
                    StepArg::Arg { arg: i } => {
                        let Some(iface) = k.params.get(*i) else {
                            errs.push(format!(
                                "{actx}: interface arg #{i} out of range ({} interface params)",
                                k.params.len()
                            ));
                            continue;
                        };
                        // Kind and dtype must match; a step may not write
                        // through an interface param declared `in`.
                        let (iface_dir, compatible) = match (iface, param) {
                            (
                                ParamType::Buf { dtype: a, dir },
                                ParamType::Buf { dtype: b, .. },
                            ) => (Some(*dir), a == b),
                            (ParamType::Ptr { dir }, ParamType::Ptr { .. }) => (Some(*dir), true),
                            (ParamType::Scalar(a), ParamType::Scalar(b)) => (None, a == b),
                            _ => (None, false),
                        };
                        if !compatible {
                            errs.push(format!(
                                "{actx}: interface param `{iface}` does not match step param `{param}`"
                            ));
                            continue;
                        }
                        let step_dir = match param {
                            ParamType::Buf { dir, .. } | ParamType::Ptr { dir } => Some(*dir),
                            ParamType::Scalar(_) => None,
                        };
                        if let (Some(idir), Some(sdir)) = (iface_dir, step_dir) {
                            if matches!(sdir, Dir::Out | Dir::InOut) && idir == Dir::In {
                                errs.push(format!(
                                    "{actx}: step writes through interface `in` param #{i}"
                                ));
                            }
                            if matches!(sdir, Dir::In | Dir::InOut) && !iface_written[*i] {
                                errs.push(format!(
                                    "{actx}: interface `out` param #{i} read before any step wrote it"
                                ));
                            }
                            if matches!(sdir, Dir::Out | Dir::InOut) {
                                iface_written[*i] = true;
                            }
                        }
                    }
                    StepArg::Scratch { scratch, offset } => {
                        let Some(sdecl) = imp.scratch.get(scratch) else {
                            errs.push(format!("{actx}: unknown scratch `{scratch}`"));
                            continue;
                        };
                        scratch_used.insert(scratch);
                        let ParamType::Buf { dtype, dir } = param else {
                            errs.push(format!(
                                "{actx}: scratch `{scratch}` bound to non-buffer param `{param}`"
                            ));
                            continue;
                        };
                        if sdecl.dtype != *dtype {
                            errs.push(format!(
                                "{actx}: scratch `{scratch}` has dtype {} but param expects {}",
                                sdecl.dtype, dtype
                            ));
                        }
                        if *offset > 0 {
                            if offset % sdecl.dtype.bytes() != 0 {
                                errs.push(format!(
                                    "{actx}: offset {offset} into scratch `{scratch}` is not {}-aligned for {}",
                                    sdecl.dtype.bytes(),
                                    sdecl.dtype
                                ));
                            }
                            if let Some(&sz) = scratch_sizes.get(scratch.as_str()) {
                                if *offset >= sz {
                                    errs.push(format!(
                                        "{actx}: offset {offset} is outside scratch `{scratch}` ({sz} bytes at symbol upper bounds)"
                                    ));
                                }
                            }
                        }
                        if matches!(dir, Dir::In | Dir::InOut)
                            && !scratch_written.contains(scratch.as_str())
                        {
                            errs.push(format!(
                                "{actx}: scratch `{scratch}` is read before any step wrote it"
                            ));
                        }
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            scratch_written.insert(scratch);
                        }
                    }
                    StepArg::I32 { .. } if matches!(param, ParamType::Scalar(ScalarType::I32)) => {}
                    StepArg::U32 { .. } if matches!(param, ParamType::Scalar(ScalarType::U32)) => {}
                    StepArg::I64 { .. } if matches!(param, ParamType::Scalar(ScalarType::I64)) => {}
                    StepArg::F32 { .. } if matches!(param, ParamType::Scalar(ScalarType::F32)) => {}
                    StepArg::U8 { .. } if matches!(param, ParamType::Scalar(ScalarType::U8)) => {}
                    arg => {
                        errs.push(format!("{actx}: {arg} does not match step param `{param}`"));
                    }
                }
            }
        }

        for (i, (p, written)) in k.params.iter().zip(&iface_written).enumerate() {
            if !written {
                errs.push(format!(
                    "kernel `{kname}`: interface `{p}` param #{i} is never written by any step"
                ));
            }
        }
        for sname in imp.scratch.keys() {
            if !scratch_used.contains(sname.as_str()) {
                errs.push(format!("kernel `{kname}`: scratch `{sname}` is never used"));
            }
        }
    }

    // 6 + 7. programs
    if m.programs.is_empty() {
        errs.push("no programs declared".to_string());
    }
    let initially_written: BTreeSet<String> = m
        .buffers
        .iter()
        .filter(|(_, b)| {
            // Carry buffers hold another program's output; whether that
            // program ran first is the caller's sequencing contract, so
            // per-program dataflow treats them as initially written.
            matches!(b.class, BufferClass::Input | BufferClass::Weight | BufferClass::Carry)
        })
        .map(|(n, _)| n.clone())
        .collect();
    let mut actually_written: BTreeSet<String> = BTreeSet::new();

    for (pname, prog) in &m.programs {
        let mut written = initially_written.clone();
        for (i, d) in prog.dispatches.iter().enumerate() {
            let ctx = match &d.label {
                Some(l) => format!("program `{pname}` dispatch #{i} ({l})"),
                None => format!("program `{pname}` dispatch #{i}"),
            };
            let Some(kernel) = m.kernels.get(&d.kernel) else {
                errs.push(format!("{ctx}: unknown kernel `{}`", d.kernel));
                continue;
            };
            used_kernels.insert(d.kernel.clone());

            if d.args.len() != kernel.params.len() {
                errs.push(format!(
                    "{ctx}: kernel `{}` takes {} params, got {} args",
                    d.kernel,
                    kernel.params.len(),
                    d.args.len()
                ));
                continue;
            }
            for (j, (arg, param)) in d.args.iter().zip(&kernel.params).enumerate() {
                let actx = format!("{ctx}: arg #{j}");
                match (arg, param) {
                    (Arg::Buf { buf, offset }, ParamType::Buf { dtype, dir }) => {
                        used_buffers.insert(buf.clone());
                        let Some(b) = m.buffers.get(buf) else {
                            errs.push(format!("{actx}: unknown buffer `{buf}`"));
                            continue;
                        };
                        if b.dtype != *dtype {
                            errs.push(format!(
                                "{actx}: buffer `{buf}` has dtype {} but param expects {}",
                                b.dtype, dtype
                            ));
                        }
                        if *offset > 0 {
                            if offset % b.dtype.bytes() != 0 {
                                errs.push(format!(
                                    "{actx}: offset {offset} into buffer `{buf}` is not {}-aligned for {}",
                                    b.dtype.bytes(),
                                    b.dtype
                                ));
                            }
                            if let Some(&sz) = buf_sizes.get(buf.as_str()) {
                                if *offset >= sz {
                                    errs.push(format!(
                                        "{actx}: offset {offset} is outside buffer `{buf}` ({sz} bytes at symbol upper bounds)"
                                    ));
                                }
                            }
                        }
                        if matches!(dir, Dir::In | Dir::InOut) && !written.contains(buf) {
                            errs.push(format!("{actx}: buffer `{buf}` is read before ever being written"));
                        }
                        if matches!(dir, Dir::Out | Dir::InOut) {
                            if matches!(b.class, BufferClass::Input | BufferClass::Weight) {
                                errs.push(format!(
                                    "{actx}: kernel writes to read-only {} buffer `{buf}`",
                                    b.class
                                ));
                            }
                            written.insert(buf.clone());
                            actually_written.insert(buf.clone());
                        }
                    }
                    (Arg::State { state, .. }, ParamType::Ptr { .. }) => {
                        // state offsets are provider layout arithmetic over a
                        // runtime-scaled pool; there is no static bound to
                        // check them against.
                        used_states.insert(state.clone());
                        if !m.states.contains_key(state) {
                            errs.push(format!("{actx}: unknown state `{state}`"));
                        }
                    }
                    (Arg::Sym { sym }, ParamType::Scalar(st)) => {
                        used_syms.insert(sym.clone());
                        match m.symbols.get(sym) {
                            None => errs.push(format!("{actx}: unknown symbol `{sym}`")),
                            Some(s) => match st {
                                ScalarType::F32 => errs.push(format!(
                                    "{actx}: symbol `{sym}` cannot bind to an f32 param"
                                )),
                                ScalarType::I32 if s.max > i32::MAX as u64 => errs.push(format!(
                                    "{actx}: symbol `{sym}` max {} exceeds i32 range",
                                    s.max
                                )),
                                ScalarType::U32 if s.max > u32::MAX as u64 => errs.push(format!(
                                    "{actx}: symbol `{sym}` max {} exceeds u32 range",
                                    s.max
                                )),
                                ScalarType::I64 if s.max > i64::MAX as u64 => errs.push(format!(
                                    "{actx}: symbol `{sym}` max {} exceeds i64 range",
                                    s.max
                                )),
                                ScalarType::U8 if s.max > u8::MAX as u64 => errs.push(format!(
                                    "{actx}: symbol `{sym}` max {} exceeds u8 range",
                                    s.max
                                )),
                                _ => {}
                            },
                        }
                    }
                    (Arg::Expr { expr }, ParamType::Scalar(st)) => {
                        check_expr(expr, m, &mut used_syms, &mut errs, &actx);
                        if *st == ScalarType::F32 {
                            errs.push(format!(
                                "{actx}: an expression cannot bind to an f32 param"
                            ));
                        } else if let Ok(v) = expr.eval(&env_max) {
                            let fits = match st {
                                ScalarType::I32 => v <= i32::MAX as u64,
                                ScalarType::U32 => v <= u32::MAX as u64,
                                ScalarType::I64 => v <= i64::MAX as u64,
                                ScalarType::U8 => v <= u8::MAX as u64,
                                ScalarType::F32 => unreachable!(),
                            };
                            if !fits {
                                errs.push(format!(
                                    "{actx}: expression reaches {v} at symbol upper \
                                     bounds, exceeding {st} range"
                                ));
                            }
                        }
                    }
                    (Arg::I32 { .. }, ParamType::Scalar(ScalarType::I32))
                    | (Arg::U32 { .. }, ParamType::Scalar(ScalarType::U32))
                    | (Arg::I64 { .. }, ParamType::Scalar(ScalarType::I64))
                    | (Arg::F32 { .. }, ParamType::Scalar(ScalarType::F32))
                    | (Arg::U8 { .. }, ParamType::Scalar(ScalarType::U8)) => {}
                    (arg, param) => {
                        errs.push(format!("{actx}: {arg} does not match param `{param}`"));
                    }
                }
            }
        }
    }
    // Outputs and carries must be produced by *some* program — a
    // prefill-style program whose only effect is state mutation legitimately
    // writes none itself.
    for (bname, b) in &m.buffers {
        if matches!(b.class, BufferClass::Output | BufferClass::Carry)
            && !actually_written.contains(bname)
        {
            errs.push(format!("{} buffer `{bname}` is never written by any program", b.class));
        }
    }

    // 8. unused declarations
    for name in m.buffers.keys() {
        if !used_buffers.contains(name) {
            errs.push(format!("buffer `{name}` is never used by any program"));
        }
    }
    for name in m.kernels.keys() {
        if !used_kernels.contains(name) {
            errs.push(format!("kernel `{name}` is never dispatched"));
        }
    }
    for name in m.states.keys() {
        if !used_states.contains(name) {
            errs.push(format!("state `{name}` is never used by any program"));
        }
    }
    for name in m.symbols.keys() {
        if !used_syms.contains(name) {
            errs.push(format!("symbol `{name}` is never used"));
        }
    }

    if errs.is_empty() {
        Ok(())
    } else {
        Err(VerifyErrors(errs))
    }
}

fn check_expr(
    e: &Expr,
    m: &Manifest,
    used_syms: &mut BTreeSet<String>,
    errs: &mut Vec<String>,
    ctx: &str,
) {
    match e {
        Expr::Const(_) => {}
        Expr::Sym { sym } => {
            if m.symbols.contains_key(sym) {
                used_syms.insert(sym.clone());
            } else {
                errs.push(format!("{ctx}: unknown symbol `{sym}`"));
            }
        }
        Expr::CeilDiv { ceil_div: (inner, c) } => {
            if *c == 0 {
                errs.push(format!("{ctx}: division by zero"));
            }
            check_expr(inner, m, used_syms, errs, ctx);
        }
        Expr::Mul { mul: (inner, c) } => {
            if *c == 0 {
                errs.push(format!("{ctx}: multiplication by constant zero"));
            }
            check_expr(inner, m, used_syms, errs, ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Manifest;

    /// The base fixture deliberately exercises the impl machinery: `attn` is
    /// a two-step implementation with a private scratch buffer.
    const BASE: &str = r#"{
      "meta": { "version": 2, "model": "toy" },
      "symbols": { "tokens": { "max": 128 } },
      "states": { "kv": { "bytes_per_token": 4096 } },
      "buffers": {
        "x": { "dtype": "i32", "shape": ["tokens"], "class": "input" },
        "w": { "dtype": "bf16", "shape": [64, 64], "class": "weight" },
        "h": { "dtype": "bf16", "shape": ["tokens", 64], "class": "workspace" },
        "y": { "dtype": "bf16", "shape": ["tokens", 64], "class": "output" }
      },
      "kernels": {
        "embed": {
          "params": ["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
          "impl": {
            "steps": [
              {
                "symbol": "embed_k",
                "params": ["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
                "block": [128, 1, 1],
                "grid": [{ "ceil_div": [{ "sym": "tokens" }, 128] }, 1, 1],
                "args": [{ "arg": 0 }, { "arg": 1 }, { "arg": 2 }, { "arg": 3 }]
              }
            ]
          }
        },
        "attn": {
          "params": ["in buffer<bf16>", "inout ptr", "out buffer<bf16>", "i32", "i64"],
          "impl": {
            "scratch": {
              "part": { "dtype": "f32", "shape": ["tokens", 8] }
            },
            "steps": [
              {
                "symbol": "attn_part_k",
                "params": ["in buffer<bf16>", "inout ptr", "out buffer<f32>", "i32", "i64"],
                "block": [128, 1, 1],
                "grid": [{ "sym": "tokens" }, 8, 1],
                "args": [{ "arg": 0 }, { "arg": 1 }, { "scratch": "part" }, { "arg": 3 }, { "arg": 4 }]
              },
              {
                "symbol": "attn_reduce_k",
                "params": ["in buffer<f32>", "out buffer<bf16>", "i32"],
                "block": [128, 1, 1],
                "grid": [{ "sym": "tokens" }, 1, 1],
                "args": [{ "scratch": "part" }, { "arg": 2 }, { "i32": 8 }]
              }
            ]
          }
        }
      },
      "programs": {
        "decode": {
          "dispatches": [
            {
              "label": "embed",
              "kernel": "embed",
              "args": [{ "buf": "x" }, { "buf": "w" }, { "buf": "h" }, { "sym": "tokens" }]
            },
            {
              "label": "attn",
              "kernel": "attn",
              "args": [{ "buf": "h" }, { "state": "kv" }, { "buf": "y" }, { "sym": "tokens" }, { "i64": 0 }]
            }
          ]
        }
      }
    }"#;

    fn base() -> serde_json::Value {
        serde_json::from_str(BASE).unwrap()
    }

    fn check(v: serde_json::Value) -> Result<(), VerifyErrors> {
        let m: Manifest = serde_json::from_value(v).map_err(|e| VerifyErrors(vec![e.to_string()]))?;
        verify(&m)
    }

    fn assert_err(v: serde_json::Value, needle: &str) {
        let errs = check(v).expect_err("expected verification failure");
        assert!(
            errs.iter().any(|e| e.contains(needle)),
            "no error containing `{needle}` in {errs:#?}"
        );
    }

    #[test]
    fn base_manifest_verifies() {
        check(base()).unwrap();
    }

    #[test]
    fn wrong_version() {
        let mut v = base();
        v["meta"]["version"] = 1.into();
        assert_err(v, "unsupported manifest version 1");
    }

    #[test]
    fn dtype_mismatch() {
        let mut v = base();
        v["buffers"]["h"]["dtype"] = "f32".into();
        assert_err(v, "has dtype f32 but param expects bf16");
    }

    #[test]
    fn unknown_kernel() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"][0]["kernel"] = "nope".into();
        assert_err(v, "unknown kernel `nope`");
    }

    #[test]
    fn arg_count_mismatch() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"][1]["args"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert_err(v, "takes 5 params, got 4 args");
    }

    #[test]
    fn read_before_write() {
        let mut v = base();
        let ds = v["programs"]["decode"]["dispatches"].as_array_mut().unwrap();
        ds.swap(0, 1);
        assert_err(v, "read before ever being written");
    }

    #[test]
    fn write_to_weight() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"][0]["args"][2] =
            serde_json::json!({ "buf": "w" });
        assert_err(v, "writes to read-only weight buffer `w`");
    }

    #[test]
    fn symbol_exceeds_i32() {
        let mut v = base();
        v["symbols"]["tokens"]["max"] = 3_000_000_000u64.into();
        assert_err(v, "exceeds i32 range");
    }

    #[test]
    fn output_never_written() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"].as_array_mut().unwrap().pop();
        assert_err(v, "output buffer `y` is never written");
    }

    #[test]
    fn duplicate_name_rejected() {
        let dup = BASE.replace(
            r#""x": { "dtype": "i32", "shape": ["tokens"], "class": "input" },"#,
            r#""x": { "dtype": "i32", "shape": ["tokens"], "class": "input" },
               "x": { "dtype": "i32", "shape": ["tokens"], "class": "input" },"#,
        );
        let err = Manifest::from_json(&dup).expect_err("duplicate must fail");
        assert!(err.to_string().contains("duplicate name `x`"), "{err}");
    }

    #[test]
    fn unknown_field_rejected() {
        let mut v = base();
        v["surprise"] = 1.into();
        let errs = check(v).expect_err("unknown field must fail");
        assert!(errs[0].contains("unknown field"), "{errs:?}");
    }

    #[test]
    fn grid_division_by_zero() {
        let mut v = base();
        v["kernels"]["embed"]["impl"]["steps"][0]["grid"][0] =
            serde_json::json!({ "ceil_div": [{ "sym": "tokens" }, 0] });
        assert_err(v, "division by zero");
    }

    #[test]
    fn unused_buffer() {
        let mut v = base();
        v["buffers"]["dead"] =
            serde_json::json!({ "dtype": "bf16", "shape": [8], "class": "workspace" });
        assert_err(v, "buffer `dead` is never used");
    }

    #[test]
    fn block_too_large() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["steps"][0]["block"] = serde_json::json!([1024, 2, 1]);
        assert_err(v, "exceeds 1024 threads");
    }

    #[test]
    fn buf_offset_ok_and_roundtrip() {
        let mut v = base();
        // h is bf16 [tokens=128, 64] -> 16384 bytes max
        v["programs"]["decode"]["dispatches"][1]["args"][0] =
            serde_json::json!({ "buf": "h", "offset": 128 });
        check(v).unwrap();
    }

    #[test]
    fn buf_offset_misaligned() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"][1]["args"][0] =
            serde_json::json!({ "buf": "h", "offset": 3 });
        assert_err(v, "not 2-aligned");
    }

    #[test]
    fn buf_offset_out_of_range() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"][1]["args"][0] =
            serde_json::json!({ "buf": "h", "offset": 16384 });
        assert_err(v, "outside buffer `h`");
    }

    #[test]
    fn u8_param_and_sym_range() {
        let mut v = base();
        v["kernels"]["attn"]["params"][3] = "u8".into();
        v["kernels"]["attn"]["impl"]["steps"][0]["params"][3] = "u8".into();
        v["programs"]["decode"]["dispatches"][1]["args"][3] =
            serde_json::json!({ "u8": 1 });
        check(v).unwrap();
        // binding a symbol with max 300 to a u8 param is rejected
        let mut v = base();
        v["kernels"]["attn"]["params"][3] = "u8".into();
        v["kernels"]["attn"]["impl"]["steps"][0]["params"][3] = "u8".into();
        v["symbols"]["tokens"]["max"] = 300.into();
        assert_err(v, "exceeds u8 range");
    }

    #[test]
    fn shared_mem_within_limit_ok() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["steps"][0]["shared_mem"] = 167_184u64.into();
        check(v).unwrap();
    }

    #[test]
    fn shared_mem_exceeds_limit() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["steps"][0]["shared_mem"] = 300_000u64.into();
        assert_err(v, "exceeds opt-in limit 232448");
    }

    #[test]
    fn buffer_arg_to_ptr_param() {
        let mut v = base();
        v["programs"]["decode"]["dispatches"][1]["args"][1] =
            serde_json::json!({ "buf": "h" });
        assert_err(v, "does not match param `inout ptr`");
    }

    #[test]
    fn grid_exceeds_cuda_limit_y() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["steps"][0]["grid"][1] = 100_000u64.into();
        assert_err(v, "exceeds CUDA limit 65535");
    }

    #[test]
    fn bad_param_string_rejected() {
        let mut v = base();
        v["kernels"]["attn"]["params"][0] = "buffer<bf16>".into();
        let errs = check(v).expect_err("param without direction must fail");
        assert!(errs[0].contains("invalid param type"), "{errs:?}");
    }

    // --- impl-layer checks ---

    #[test]
    fn step_arg_index_out_of_range() {
        let mut v = base();
        v["kernels"]["embed"]["impl"]["steps"][0]["args"][3] =
            serde_json::json!({ "arg": 9 });
        assert_err(v, "interface arg #9 out of range");
    }

    #[test]
    fn step_writes_interface_in_param() {
        let mut v = base();
        v["kernels"]["embed"]["impl"]["steps"][0]["params"][0] = "out buffer<i32>".into();
        assert_err(v, "writes through interface `in` param #0");
    }

    #[test]
    fn step_iface_kind_mismatch() {
        let mut v = base();
        // step param says bf16 buffer where the interface forwards i32
        v["kernels"]["embed"]["impl"]["steps"][0]["params"][0] = "in buffer<bf16>".into();
        assert_err(v, "does not match step param");
    }

    #[test]
    fn scratch_read_before_write() {
        let mut v = base();
        let steps = v["kernels"]["attn"]["impl"]["steps"].as_array_mut().unwrap();
        steps.swap(0, 1);
        assert_err(v, "scratch `part` is read before any step wrote it");
    }

    #[test]
    fn scratch_unknown() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["steps"][0]["args"][2] =
            serde_json::json!({ "scratch": "nope" });
        assert_err(v, "unknown scratch `nope`");
    }

    #[test]
    fn scratch_unused() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["scratch"]["dead"] =
            serde_json::json!({ "dtype": "f32", "shape": [4] });
        assert_err(v, "scratch `dead` is never used");
    }

    #[test]
    fn scratch_dtype_mismatch() {
        let mut v = base();
        v["kernels"]["attn"]["impl"]["steps"][0]["params"][2] = "out buffer<bf16>".into();
        assert_err(v, "scratch `part` has dtype f32 but param expects bf16");
    }

    #[test]
    fn scratch_offset_out_of_range() {
        let mut v = base();
        // part is f32 [tokens=128, 8] -> 4096 bytes max
        v["kernels"]["attn"]["impl"]["steps"][1]["args"][0] =
            serde_json::json!({ "scratch": "part", "offset": 4096 });
        assert_err(v, "outside scratch `part`");
    }

    #[test]
    fn interface_out_never_written() {
        let mut v = base();
        // reduce step now writes scratch instead of the interface out param
        v["kernels"]["attn"]["impl"]["steps"][1]["params"][1] = "out buffer<f32>".into();
        v["kernels"]["attn"]["impl"]["steps"][1]["args"][1] =
            serde_json::json!({ "scratch": "part" });
        assert_err(v, "param #2 is never written by any step");
    }

    #[test]
    fn empty_impl_rejected() {
        let mut v = base();
        v["kernels"]["embed"]["impl"]["steps"] = serde_json::json!([]);
        assert_err(v, "implementation has no steps");
    }

    #[test]
    fn sha256_without_cubin() {
        let mut v = base();
        v["kernels"]["embed"]["impl"]["steps"][0]["sha256"] =
            serde_json::json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        assert_err(v, "sha256 without a cubin file");
    }
}
