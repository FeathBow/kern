//! Thin verifier-driven executor for kern manifests.
//!
//! The runtime knows nothing about models. It loads a verified manifest,
//! resolves each declared kernel against the cubins in a directory, allocates
//! every buffer/state, binds weight buffers by name from a safetensors blob,
//! and replays the program's dispatch list. The only kernels it understands
//! natively are `extern:` ops (currently `extern:cublaslt_bf16_tn`).
//!
//! Same-name Triton kernels ship multiple constexpr instances with different
//! ABIs across modules; resolution picks the instance whose
//! `cuFuncGetParamInfo` layout matches the manifest's declared params — the
//! phase-2 ABI check doubles as instance selection.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::fmt::Write as _;
use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    result as cu, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice,
    PinnedHostSlice, SyncOnDrop,
};
use half::bf16;
use kern_manifest::types::{Arg, BufferClass, Dispatch, Expr, Manifest, ParamType, StepArg};
use sha2::Digest;

pub type Error = Box<dyn std::error::Error + Send + Sync>;
pub type Result<T> = std::result::Result<T, Error>;

macro_rules! bail {
    ($($t:tt)*) => { return Err(format!($($t)*).into()) };
}

pub struct DeviceBuf {
    slice: CudaSlice<u8>,
    pub ptr: u64,
    pub bytes: u64,
}

enum StepImpl {
    Cubin { func: sys::CUfunction, module: String },
    GemmBf16Tn { beta: f32 },
}

/// A kernel implementation, resolved: one entry per step, plus the private
/// scratch buffers the impl declared (allocated once at symbol max, reused
/// every dispatch — contents are dead outside a single dispatch).
struct ResolvedKernel {
    steps: Vec<StepImpl>,
    scratch: BTreeMap<String, DeviceBuf>,
}

/// A dispatch arg or step arg resolved to its launch value: the low bytes
/// of `val` are what the param slot receives; `bytes` is the remaining
/// buffer size for pointer args (0 for scalars), used by extern ops.
#[derive(Clone, Copy)]
struct RVal {
    val: u64,
    bytes: u64,
}

/// Raw device pointer presented as a `DevicePtr<bf16>`/`DevicePtrMut<bf16>`
/// for the cublasLt extern op. Synchronization is trivially correct: the
/// whole runtime is single-stream and cublasLt is bound to that stream.
struct RawBf16 {
    ptr: sys::CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
}

impl DeviceSlice<bf16> for RawBf16 {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<bf16> for RawBf16 {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<bf16> for RawBf16 {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

pub struct Runtime {
    pub manifest: Manifest,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blt: CudaBlasLT,
    kernels: BTreeMap<String, ResolvedKernel>,
    buffers: BTreeMap<String, DeviceBuf>,
    states: BTreeMap<String, DeviceBuf>,
    /// Persistent pinned staging, one per input buffer: H2D from pageable
    /// memory degrades to a synchronous driver-staged copy (tens of µs per
    /// call); through page-locked staging it is a true async DMA. The pinned
    /// slice's event guards reuse across steps.
    staging: BTreeMap<String, PinnedHostSlice<u8>>,
    n_modules: usize,
    /// Program name -> instantiated CUDA graph + the symbol values it was
    /// captured with (grid dims and scalar args are baked in at capture).
    graphs: BTreeMap<String, (sys::CUgraphExec, BTreeMap<String, u64>)>,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        for (exec, _) in self.graphs.values() {
            unsafe { sys::cuGraphExecDestroy(*exec) };
        }
    }
}

fn ev(e: &Expr, env: &BTreeMap<String, u64>) -> Result<u64> {
    e.eval(env).map_err(|e| format!("expression eval: {e}").into())
}

fn cuda_check(r: sys::CUresult, what: &str) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        bail!("{what}: CUDA error {r:?}")
    }
}

/// Parameter byte sizes of a loaded function, per `cuFuncGetParamInfo`.
fn param_sizes(func: sys::CUfunction) -> Result<Vec<usize>> {
    let mut sizes = Vec::new();
    loop {
        let (mut off, mut size) = (0usize, 0usize);
        let r = unsafe { sys::cuFuncGetParamInfo(func, sizes.len(), &mut off, &mut size) };
        if r == sys::CUresult::CUDA_ERROR_INVALID_VALUE {
            return Ok(sizes);
        }
        cuda_check(r, "cuFuncGetParamInfo")?;
        sizes.push(size);
    }
}

impl Runtime {
    /// Verify the manifest, load every `*.cubin` under `kernels_dir`, resolve
    /// kernels, and allocate all buffers and states. `state_capacity_tokens`
    /// scales each declared state by its `bytes_per_token`.
    pub fn load(
        manifest_json: &str,
        kernels_dir: &std::path::Path,
        gpu: usize,
        state_capacity_tokens: u64,
    ) -> Result<Runtime> {
        let manifest = Manifest::from_json(manifest_json)?;
        if let Err(errs) = kern_manifest::verify(&manifest) {
            let mut msg = String::from("manifest failed verification:\n");
            for e in errs {
                let _ = writeln!(msg, "  - {e}");
            }
            return Err(msg.into());
        }

        let ctx = CudaContext::new(gpu)?;
        // A created (non-legacy) stream: the NULL stream cannot be captured
        // into a CUDA graph.
        let stream = ctx.new_stream()?;
        let blt = CudaBlasLT::new(stream.clone())?;
        ctx.bind_to_thread()?;

        // Load modules in name order so instance resolution is deterministic.
        let mut cubins: Vec<_> = std::fs::read_dir(kernels_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "cubin"))
            .collect();
        cubins.sort();
        if cubins.is_empty() {
            bail!("no .cubin files in {}", kernels_dir.display());
        }
        let mut modules = Vec::new();
        for path in &cubins {
            let cmod = cu::module::load(CString::new(path.to_str().unwrap())?)
                .map_err(|e| format!("loading {}: {e:?}", path.display()))?;
            let file = path.file_name().unwrap().to_string_lossy().into_owned();
            modules.push((file, cmod));
        }

        let max_env: BTreeMap<_, _> =
            manifest.symbols.iter().map(|(s, v)| (s.clone(), v.max)).collect();

        let mut kernels = BTreeMap::new();
        for (name, k) in &manifest.kernels {
            let mut steps = Vec::new();
            for (si, st) in k.imp.steps.iter().enumerate() {
                if let Some(ext) = st.symbol.strip_prefix("extern:") {
                    match ext {
                        "cublaslt_bf16_tn" => steps.push(StepImpl::GemmBf16Tn { beta: 0.0 }),
                        "cublaslt_bf16_tn_acc" => steps.push(StepImpl::GemmBf16Tn { beta: 1.0 }),
                        _ => bail!("kernel `{name}` step #{si}: unsupported extern op `{ext}`"),
                    }
                    continue;
                }
                // Pinned-artifact integrity: when the step names its cubin
                // (the pluggable path), verify the file hash if declared.
                if let (Some(cb), Some(sha)) = (&st.cubin, &st.sha256) {
                    let path = kernels_dir.join(cb);
                    let data = std::fs::read(&path)
                        .map_err(|e| format!("kernel `{name}` step #{si}: reading {}: {e}", path.display()))?;
                    let got = format!("{:x}", sha2::Sha256::digest(&data));
                    if got != sha.to_lowercase() {
                        bail!(
                            "kernel `{name}` step #{si}: cubin `{cb}` sha256 mismatch: \
                             manifest declares {sha}, file is {got}"
                        );
                    }
                }
                let want: Vec<usize> =
                    st.params.iter().map(|p| p.size_bytes() as usize).collect();
                let sym = CString::new(st.symbol.as_str())?;
                let mut resolved = None;
                let mut seen = Vec::new();
                for (file, cmod) in &modules {
                    if let Some(cb) = &st.cubin {
                        if file != cb {
                            continue;
                        }
                    }
                    let Ok(func) = (unsafe { cu::module::get_function(*cmod, sym.clone()) })
                    else {
                        continue;
                    };
                    let got = param_sizes(func)?;
                    if got == want {
                        resolved = Some(StepImpl::Cubin { func, module: file.clone() });
                        break;
                    }
                    seen.push(format!("{file}: {got:?}"));
                }
                let Some(r) = resolved else {
                    bail!(
                        "kernel `{name}` step #{si} ({}): no loaded instance matches declared \
                         param layout {want:?} (cubin filter {:?}); saw {seen:?}",
                        st.symbol,
                        st.cubin
                    );
                };
                // Opt in to >48KB dynamic shared memory where the step needs it.
                if let (StepImpl::Cubin { func, .. }, Some(sm)) = (&r, &st.shared_mem) {
                    let bytes =
                        sm.eval(&max_env).map_err(|e| format!("kernel `{name}`: {e}"))?;
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
                steps.push(r);
            }
            let mut scratch = BTreeMap::new();
            for (sname, sd) in &k.imp.scratch {
                let mut elems = 1u64;
                for dim in &sd.shape {
                    let n = match dim {
                        kern_manifest::types::Dim::Const(c) => *c,
                        kern_manifest::types::Dim::Sym(s) => max_env[s],
                    };
                    elems = elems.checked_mul(n).ok_or("scratch size overflow")?;
                }
                scratch.insert(sname.clone(), alloc(&stream, elems * sd.dtype.bytes())?);
            }
            kernels.insert(name.clone(), ResolvedKernel { steps, scratch });
        }

        // Buffer sizes are static: shapes only reference symbols, sized at max.
        let mut buffers = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            let mut elems = 1u64;
            for d in &b.shape {
                let n = match d {
                    kern_manifest::types::Dim::Const(c) => *c,
                    kern_manifest::types::Dim::Sym(s) => max_env[s],
                };
                elems = elems.checked_mul(n).ok_or("buffer size overflow")?;
            }
            let bytes = elems * b.dtype.bytes();
            buffers.insert(name.clone(), alloc(&stream, bytes)?);
        }
        let mut states = BTreeMap::new();
        for (name, s) in &manifest.states {
            states.insert(name.clone(), alloc(&stream, s.bytes_per_token * state_capacity_tokens)?);
        }
        let mut staging = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            if b.class == BufferClass::Input {
                let mut pinned =
                    unsafe { ctx.alloc_pinned::<u8>(buffers[name].bytes.max(1) as usize)? };
                pinned.as_mut_slice()?.fill(0);
                staging.insert(name.clone(), pinned);
            }
        }

        let n_modules = modules.len();
        Ok(Runtime {
            manifest,
            ctx,
            stream,
            blt,
            kernels,
            buffers,
            states,
            staging,
            n_modules,
            graphs: BTreeMap::new(),
        })
    }

    pub fn module_count(&self) -> usize {
        self.n_modules
    }

    /// (name, class, allocated bytes) for every buffer.
    pub fn buffer_sizes(&self) -> Vec<(&str, BufferClass, u64)> {
        self.manifest
            .buffers
            .iter()
            .map(|(n, b)| (n.as_str(), b.class, self.buffers[n].bytes))
            .collect()
    }

    /// (name, bytes_per_token, allocated bytes) for every state.
    pub fn state_sizes(&self) -> Vec<(&str, u64, u64)> {
        self.manifest
            .states
            .iter()
            .map(|(n, s)| (n.as_str(), s.bytes_per_token, self.states[n].bytes))
            .collect()
    }

    /// Per kernel: the module each impl step resolved to, in step order.
    pub fn kernel_resolution(&self) -> Vec<(String, Vec<String>)> {
        self.kernels
            .iter()
            .map(|(n, r)| {
                let mods = r
                    .steps
                    .iter()
                    .map(|s| match s {
                        StepImpl::Cubin { module, .. } => module.clone(),
                        StepImpl::GemmBf16Tn { .. } => "runtime built-in (cublasLt)".into(),
                    })
                    .collect();
                (n.clone(), mods)
            })
            .collect()
    }

    /// Bind every `weight` buffer by name from a safetensors blob.
    pub fn load_weights(&mut self, blob: &[u8]) -> Result<()> {
        let st = safetensors::SafeTensors::deserialize(blob)?;
        for (name, b) in &self.manifest.buffers {
            if b.class != BufferClass::Weight {
                continue;
            }
            let t = st
                .tensor(name)
                .map_err(|e| format!("weight `{name}` missing from artifact: {e}"))?;
            let dst = self.buffers.get_mut(name).unwrap();
            if t.data().len() as u64 != dst.bytes {
                bail!(
                    "weight `{name}`: artifact has {} bytes, manifest declares {}",
                    t.data().len(),
                    dst.bytes
                );
            }
            self.stream.memcpy_htod(t.data(), &mut dst.slice)?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn write_input(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!("no buffer `{name}`");
        };
        if b.class != BufferClass::Input {
            bail!("buffer `{name}` is {}, not input", b.class);
        }
        let dst = self.buffers.get_mut(name).unwrap();
        if data.len() as u64 > dst.bytes {
            bail!("input `{name}`: got {} bytes, buffer is {}", data.len(), dst.bytes);
        }
        let pinned = self.staging.get_mut(name).unwrap();
        // Waits on the pinned slice's event: the previous step's DMA from
        // this staging must finish before we overwrite it. A prefix write
        // (variable-length inputs) still DMAs the whole buffer — the stale
        // tail is never read, grids are bounded by the symbols.
        pinned.as_mut_slice()?[..data.len()].copy_from_slice(data);
        self.stream.memcpy_htod(pinned, &mut dst.slice)?;
        Ok(())
    }

    pub fn read_output(&self, name: &str) -> Result<Vec<u8>> {
        let Some(b) = self.manifest.buffers.get(name) else {
            bail!("no buffer `{name}`");
        };
        if b.class != BufferClass::Output {
            bail!("buffer `{name}` is {}, not output", b.class);
        }
        Ok(self.stream.clone_dtoh(&self.buffers[name].slice)?)
    }

    fn check_env(&self, env: &BTreeMap<String, u64>) -> Result<()> {
        for (sym, decl) in &self.manifest.symbols {
            let Some(v) = env.get(sym) else {
                bail!("symbol `{sym}` not provided");
            };
            if *v < decl.min || *v > decl.max {
                bail!("symbol `{sym}` = {v} outside declared [{}, {}]", decl.min, decl.max);
            }
        }
        Ok(())
    }

    /// Execute one program with the given symbol values, then synchronize.
    pub fn run(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.manifest.programs.get(program) else {
            bail!("no program `{program}`");
        };
        self.check_env(env)?;
        self.ctx.bind_to_thread()?;
        for (i, d) in prog.dispatches.iter().enumerate() {
            self.dispatch(d, env).map_err(|e| {
                let label = d.label.as_deref().unwrap_or("");
                format!("dispatch #{i} {label} (kernel `{}`): {e}", d.kernel)
            })?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// Capture one program into an instantiated CUDA graph. Grid dims and
    /// scalar args (symbol values included) are baked in at capture; input
    /// buffer *contents* are read at replay, so per-step H2D writes stay
    /// outside the graph and `run_captured` replays the whole dispatch list
    /// with one launch.
    pub fn capture(&mut self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.manifest.programs.get(program) else {
            bail!("no program `{program}`");
        };
        self.check_env(env)?;
        self.ctx.bind_to_thread()?;
        cuda_check(
            unsafe {
                sys::cuStreamBeginCapture_v2(
                    self.stream.cu_stream(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
            },
            "cuStreamBeginCapture",
        )?;
        let mut failed = None;
        for (i, d) in prog.dispatches.iter().enumerate() {
            if let Err(e) = self.dispatch(d, env) {
                let label = d.label.as_deref().unwrap_or("");
                failed = Some(format!("dispatch #{i} {label} (kernel `{}`): {e}", d.kernel));
                break;
            }
        }
        // Always end the capture, even on error — a stream stuck in capture
        // mode poisons every later operation on it.
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let end = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
        if let Some(e) = failed {
            if !graph.is_null() {
                unsafe { sys::cuGraphDestroy(graph) };
            }
            return Err(e.into());
        }
        cuda_check(end, "cuStreamEndCapture")?;
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        unsafe { sys::cuGraphDestroy(graph) };
        cuda_check(r, "cuGraphInstantiateWithFlags")?;
        if let Some((old, _)) = self.graphs.insert(program.to_string(), (exec, env.clone())) {
            unsafe { sys::cuGraphExecDestroy(old) };
        }
        Ok(())
    }

    /// Replay a previously captured program, then synchronize.
    pub fn run_captured(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some((exec, captured)) = self.graphs.get(program) else {
            bail!("program `{program}` has not been captured");
        };
        if captured != env {
            bail!("graph for `{program}` was captured with {captured:?}, called with {env:?}");
        }
        self.ctx.bind_to_thread()?;
        cuda_check(
            unsafe { sys::cuGraphLaunch(*exec, self.stream.cu_stream()) },
            "cuGraphLaunch",
        )?;
        self.stream.synchronize()?;
        Ok(())
    }

    fn arg_ptr(&self, arg: &Arg) -> Result<(u64, u64)> {
        match arg {
            Arg::Buf { buf, offset } => {
                let b = &self.buffers[buf];
                Ok((b.ptr + offset, b.bytes - offset))
            }
            Arg::State { state, offset } => {
                let s = &self.states[state];
                Ok((s.ptr + offset, s.bytes - offset))
            }
            _ => bail!("expected buffer/state arg, got {arg}"),
        }
    }

    fn scalar(&self, arg: &Arg, env: &BTreeMap<String, u64>) -> Result<u64> {
        Ok(match arg {
            Arg::Sym { sym } => env[sym],
            Arg::Expr { expr } => ev(expr, env)?,
            Arg::I32 { i32: v } => *v as u32 as u64,
            Arg::U32 { u32: v } => *v as u64,
            Arg::I64 { i64: v } => *v as u64,
            Arg::U8 { u8: v } => *v as u64,
            Arg::F32 { f32: v } => v.to_bits() as u64,
            _ => bail!("expected scalar arg, got {arg}"),
        })
    }

    fn dispatch(&self, d: &Dispatch, env: &BTreeMap<String, u64>) -> Result<()> {
        let rk = &self.kernels[&d.kernel];
        let k = &self.manifest.kernels[&d.kernel];
        // Resolve the interface args once; each step then wires its own launch
        // params from these, its scratch, and its private literals.
        let mut vals = Vec::with_capacity(d.args.len());
        for (arg, pty) in d.args.iter().zip(&k.params) {
            let v = match pty {
                ParamType::Buf { .. } | ParamType::Ptr { .. } => {
                    let (ptr, bytes) = self.arg_ptr(arg)?;
                    RVal { val: ptr, bytes }
                }
                ParamType::Scalar(_) => RVal { val: self.scalar(arg, env)?, bytes: 0 },
            };
            vals.push(v);
        }
        for (si, (st, imp)) in k.imp.steps.iter().zip(&rk.steps).enumerate() {
            let mut slots = Vec::with_capacity(st.args.len());
            for sa in &st.args {
                let rv = match sa {
                    StepArg::Arg { arg } => vals[*arg],
                    StepArg::Scratch { scratch, offset } => {
                        let b = &rk.scratch[scratch];
                        RVal { val: b.ptr + offset, bytes: b.bytes - offset }
                    }
                    StepArg::I32 { i32: v } => RVal { val: *v as u32 as u64, bytes: 0 },
                    StepArg::U32 { u32: v } => RVal { val: *v as u64, bytes: 0 },
                    StepArg::I64 { i64: v } => RVal { val: *v as u64, bytes: 0 },
                    StepArg::U8 { u8: v } => RVal { val: *v as u64, bytes: 0 },
                    StepArg::F32 { f32: v } => RVal { val: v.to_bits() as u64, bytes: 0 },
                };
                slots.push(rv);
            }
            let StepImpl::Cubin { func, .. } = imp else {
                let StepImpl::GemmBf16Tn { beta } = imp else { unreachable!() };
                self.gemm_bf16_tn(&slots, *beta).map_err(|e| format!("step #{si}: {e}"))?;
                continue;
            };
            let grid = [ev(&st.grid[0], env)?, ev(&st.grid[1], env)?, ev(&st.grid[2], env)?];
            let smem = match &st.shared_mem {
                Some(e) => ev(e, env)?,
                None => 0,
            };
            // Every param slot staged as a little-endian u64; the launch ABI
            // reads the low `size_bytes()` of each slot.
            let raw: Vec<u64> = slots.iter().map(|r| r.val).collect();
            let mut params: Vec<*mut c_void> =
                raw.iter().map(|s| s as *const u64 as *mut c_void).collect();
            unsafe {
                cu::launch_kernel(
                    *func,
                    (grid[0] as u32, grid[1] as u32, grid[2] as u32),
                    (st.block[0], st.block[1], st.block[2]),
                    smem as u32,
                    self.stream.cu_stream(),
                    &mut params,
                )
                .map_err(|e| format!("step #{si} ({}): {e:?}", st.symbol))?;
            }
        }
        Ok(())
    }

    /// `extern:cublaslt_bf16_tn`: row-major `C[m,n] = A[m,k] @ W[n,k]^T`,
    /// resolved args `[a, w, c, m, n, k]`. Column-major mapping: compute
    /// `C_cm[n,m] = W_cm^T[n,k] x A_cm[k,m]` -> transa=T on W (lda=k),
    /// transb=N on A (ldb=k), m'=n, n'=m, ldc=n.
    /// `extern:cublaslt_bf16_tn_acc` is the same with beta=1: `C += A @ W^T`.
    fn gemm_bf16_tn(&self, args: &[RVal], beta: f32) -> Result<()> {
        let [a, w, c, m, n, k] = args else {
            bail!("gemm expects 6 args, got {}", args.len());
        };
        let (a_ptr, a_bytes) = (a.val, a.bytes);
        let (w_ptr, w_bytes) = (w.val, w.bytes);
        let (c_ptr, c_bytes) = (c.val, c.bytes);
        let (m, n, k) = (m.val, n.val, k.val);
        let view = |ptr, bytes| RawBf16 { ptr, len: (bytes / 2) as usize, stream: self.stream.clone() };
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: n,
            n: m,
            k,
            alpha: 1.0,
            beta,
            lda: k as i64,
            ldb: k as i64,
            ldc: n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let mut out = view(c_ptr, c_bytes);
        unsafe {
            self.blt
                .matmul(cfg, &view(w_ptr, w_bytes), &view(a_ptr, a_bytes), &mut out, None, None)
                .map_err(|e| format!("cublasLt matmul (m={m} n={n} k={k}): {e:?}"))?;
        }
        Ok(())
    }
}

fn alloc(stream: &Arc<CudaStream>, bytes: u64) -> Result<DeviceBuf> {
    let slice = stream.alloc_zeros::<u8>(bytes.max(1) as usize)?;
    let ptr = {
        let (p, _sync) = slice.device_ptr(stream);
        p
    };
    Ok(DeviceBuf { slice, ptr, bytes })
}
