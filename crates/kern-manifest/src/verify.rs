//! Load-time verification: refuse any manifest that is not provably
//! self-consistent, before anything touches the GPU. All errors are
//! collected and reported together, rustc-style.
//!
//! Checks:
//!   1. meta: format version, artifact hash shape
//!   2. symbols: bounds sane
//!   3. states: non-zero size, power-of-two alignment
//!   4. buffers: shapes resolve, byte sizes don't overflow at symbol upper
//!      bounds
//!   5. kernels: CUDA block limits, shared-mem expressions resolve
//!   6. dispatches: kernel refs resolve, arg/param arity and per-position
//!      type match, symbol ranges fit scalar params, grid within CUDA
//!      limits at symbol upper bounds and non-zero at lower bounds
//!   7. dataflow per program: no read-before-write, no writes to input or
//!      weight buffers, every output buffer written
//!   8. no unused declarations
//!
//! What this deliberately cannot check: kernel *behavior*. A cubin that
//! lies about what it touches is inside the trust boundary; the manifest
//! only makes the lie explicit and diffable. Cross-checking param layouts
//! against `cuKernelGetParamInfo` is a load-time (phase 2) concern in the
//! runtime crate, since it needs the CUDA driver.

use crate::types::*;
use std::collections::{BTreeMap, BTreeSet};

const MAX_GRID_X: u64 = (1 << 31) - 1;
const MAX_GRID_YZ: u64 = 65_535;
const MAX_BLOCK_THREADS: u64 = 1024;
const MAX_BLOCK_Z: u32 = 64;

pub fn verify(m: &Manifest) -> Result<(), Vec<String>> {
    let mut errs: Vec<String> = Vec::new();
    let mut used_syms: BTreeSet<String> = BTreeSet::new();
    let mut used_buffers: BTreeSet<String> = BTreeSet::new();
    let mut used_states: BTreeSet<String> = BTreeSet::new();
    let mut used_kernels: BTreeSet<String> = BTreeSet::new();

    // 1. meta
    if m.meta.version != 1 {
        errs.push(format!("meta: unsupported manifest version {}", m.meta.version));
    }
    let sha = &m.meta.cubin.sha256;
    if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        errs.push(format!("meta: cubin sha256 `{sha}` is not 64 hex chars"));
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
    for (name, b) in &m.buffers {
        if b.shape.is_empty() {
            errs.push(format!("buffer `{name}`: shape must not be empty"));
            continue;
        }
        let mut dim_err = false;
        let mut size: Option<u64> = Some(b.dtype.bytes());
        for dim in &b.shape {
            let extent = match dim {
                Dim::Const(0) => {
                    errs.push(format!("buffer `{name}`: zero-sized dimension"));
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
                        errs.push(format!("buffer `{name}`: unknown symbol `{s}` in shape"));
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
            errs.push(format!("buffer `{name}`: byte size overflows u64 at symbol upper bounds"));
        }
    }

    // 5. kernels
    for (name, k) in &m.kernels {
        if k.symbol.is_empty() {
            errs.push(format!("kernel `{name}`: empty entry symbol"));
        }
        let threads: u64 = k.block.iter().map(|&x| x as u64).product();
        if k.block.contains(&0) || threads > MAX_BLOCK_THREADS {
            errs.push(format!("kernel `{name}`: block {:?} exceeds {MAX_BLOCK_THREADS} threads or has a zero dim", k.block));
        }
        if k.block[2] > MAX_BLOCK_Z {
            errs.push(format!("kernel `{name}`: block.z {} > {MAX_BLOCK_Z}", k.block[2]));
        }
        if let Some(e) = &k.shared_mem {
            check_expr(e, m, &mut used_syms, &mut errs, &format!("kernel `{name}` shared_mem"));
        }
    }

    // 6 + 7. programs
    if m.programs.is_empty() {
        errs.push("no programs declared".to_string());
    }
    let initially_written: BTreeSet<String> = m
        .buffers
        .iter()
        .filter(|(_, b)| matches!(b.class, BufferClass::Input | BufferClass::Weight))
        .map(|(n, _)| n.clone())
        .collect();

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

            for (axis, e) in ["x", "y", "z"].iter().zip(&d.grid) {
                let ectx = format!("{ctx}: grid.{axis}");
                check_expr(e, m, &mut used_syms, &mut errs, &ectx);
                match e.eval(&env_max) {
                    Ok(v) => {
                        let limit = if *axis == "x" { MAX_GRID_X } else { MAX_GRID_YZ };
                        if v > limit {
                            errs.push(format!("{ectx}: {v} exceeds CUDA limit {limit} at symbol upper bounds"));
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
                    (Arg::Buf { buf }, ParamType::Buf { dtype, dir }) => {
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
                        }
                    }
                    (Arg::State { state }, ParamType::Ptr { .. }) => {
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
                                _ => {}
                            },
                        }
                    }
                    (Arg::I32 { .. }, ParamType::Scalar(ScalarType::I32))
                    | (Arg::U32 { .. }, ParamType::Scalar(ScalarType::U32))
                    | (Arg::I64 { .. }, ParamType::Scalar(ScalarType::I64))
                    | (Arg::F32 { .. }, ParamType::Scalar(ScalarType::F32)) => {}
                    (arg, param) => {
                        errs.push(format!("{actx}: {arg} does not match param `{param}`"));
                    }
                }
            }
        }
        for (bname, b) in &m.buffers {
            if b.class == BufferClass::Output && !written.contains(bname) {
                errs.push(format!("program `{pname}`: output buffer `{bname}` is never written"));
            }
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
        Err(errs)
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

    const BASE: &str = r#"{
      "meta": {
        "version": 1,
        "model": "toy",
        "cubin": {
          "file": "k.cubin",
          "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
      },
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
          "symbol": "embed_k",
          "params": ["in buffer<i32>", "in buffer<bf16>", "out buffer<bf16>", "i32"],
          "block": [128, 1, 1]
        },
        "attn": {
          "symbol": "attn_k",
          "params": ["in buffer<bf16>", "inout ptr", "out buffer<bf16>", "i32", "i64"],
          "block": [128, 1, 1]
        }
      },
      "programs": {
        "decode": {
          "dispatches": [
            {
              "label": "embed",
              "kernel": "embed",
              "grid": [{ "ceil_div": [{ "sym": "tokens" }, 128] }, 1, 1],
              "args": [{ "buf": "x" }, { "buf": "w" }, { "buf": "h" }, { "sym": "tokens" }]
            },
            {
              "label": "attn",
              "kernel": "attn",
              "grid": [{ "sym": "tokens" }, 1, 1],
              "args": [{ "buf": "h" }, { "state": "kv" }, { "buf": "y" }, { "sym": "tokens" }, { "i64": 0 }]
            }
          ]
        }
      }
    }"#;

    fn base() -> serde_json::Value {
        serde_json::from_str(BASE).unwrap()
    }

    fn check(v: serde_json::Value) -> Result<(), Vec<String>> {
        let m: Manifest = serde_json::from_value(v).map_err(|e| vec![e.to_string()])?;
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
        v["programs"]["decode"]["dispatches"][0]["grid"][0] =
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
        v["kernels"]["attn"]["block"] = serde_json::json!([1024, 2, 1]);
        assert_err(v, "exceeds 1024 threads");
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
        v["programs"]["decode"]["dispatches"][1]["grid"][1] = 100_000u64.into();
        assert_err(v, "exceeds CUDA limit 65535");
    }

    #[test]
    fn bad_param_string_rejected() {
        let mut v = base();
        v["kernels"]["attn"]["params"][0] = "buffer<bf16>".into();
        let errs = check(v).expect_err("param without direction must fail");
        assert!(errs[0].contains("invalid param type"), "{errs:?}");
    }
}
