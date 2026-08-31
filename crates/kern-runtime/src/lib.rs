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
use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{
    result as cu, sys, CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice,
    PinnedHostSlice, SyncOnDrop,
};
use half::bf16;
use kern_manifest::types::{
    Arg, BufferClass, Dispatch, Expr, Manifest, ParamType, RegistryRef, StepArg,
};
use sha2::Digest;

/// Runtime errors, grouped by who has to act on them.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The manifest JSON does not parse. Provider-side: fix the generator
    /// output.
    #[error("manifest parse: {0}")]
    ManifestParse(#[from] serde_json::Error),

    /// The manifest parsed but failed static verification (all diagnostics
    /// collected). Provider-side: fix the generator.
    #[error(transparent)]
    ManifestVerify(#[from] kern_manifest::VerifyErrors),

    /// A manifest inconsistency only detectable past static verification
    /// (size overflow at symbol bounds, unsupported extern op, expression
    /// evaluation, wiring arity). Provider-side bug the verifier missed.
    #[error("manifest: {0}")]
    Manifest(String),

    /// The kernel artifacts don't satisfy the manifest: missing or
    /// unreadable cubin, sha256 mismatch, or no loaded instance matching a
    /// declared param layout. Re-extract the kernels or re-pin them.
    #[error("kernel artifact: {0}")]
    KernelArtifact(String),

    /// The weight artifact doesn't satisfy the manifest: unparseable
    /// safetensors, missing tensor, or byte-size mismatch. Re-export the
    /// weights for this manifest.
    #[error("weight artifact: {0}")]
    WeightArtifact(String),

    /// The caller broke the runtime API contract: unknown buffer/program
    /// name, wrong buffer class, oversized input write, symbol value
    /// outside declared bounds, or replaying a program that wasn't captured
    /// (or captured with different symbol values). Caller-side bug.
    #[error("caller contract: {0}")]
    Api(String),

    /// One dispatch of a program failed; `context` locates it in the
    /// dispatch list, `source` is the underlying failure.
    #[error("{context}: {source}")]
    Dispatch {
        context: String,
        #[source]
        source: Box<Error>,
    },

    /// A raw CUDA driver or cublasLt call failed at load or execution time.
    #[error("cuda: {0}")]
    Cuda(String),

    /// A CUDA driver call through cudarc failed (allocation, memcpy,
    /// synchronize, context/stream setup).
    #[error(transparent)]
    Driver(#[from] cudarc::driver::DriverError),

    /// cublasLt handle creation failed.
    #[error(transparent)]
    Blas(#[from] cudarc::cublaslt::result::CublasError),

    /// Filesystem access failed (kernels dir listing, cubin reads).
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `bail!(Variant, "...")` for the message-carrying variants above.
macro_rules! bail {
    ($variant:ident, $($t:tt)*) => { return Err(crate::Error::$variant(format!($($t)*))) };
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
    e.eval(env).map_err(|e| Error::Manifest(format!("expression eval: {e}")))
}

fn cuda_check(r: sys::CUresult, what: &str) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        bail!(Cuda, "{what}: {r:?}")
    }
}

/// Error context locating one entry of a program's dispatch list.
fn dispatch_ctx(i: usize, d: &Dispatch) -> String {
    match &d.label {
        Some(l) => format!("dispatch #{i} `{l}` (kernel `{}`)", d.kernel),
        None => format!("dispatch #{i} (kernel `{}`)", d.kernel),
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

/// Materialize a registry cubin into the content-addressed cache
/// (`$KERN_CACHE_DIR` or `~/.cache/kern`, `blobs/<sha256>`) and return its
/// local path. A cache hit is re-hashed (corruption re-fetches); a download
/// is hash-checked before it lands in the cache, so the transport — the
/// Hugging Face `resolve/` endpoint or whatever fronts it — is untrusted.
fn fetch_registry_cubin(reg: &RegistryRef, sha256: &str) -> Result<std::path::PathBuf> {
    let sha = sha256.to_lowercase();
    let cache_root = std::env::var_os("KERN_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache/kern")))
        .ok_or_else(|| {
            Error::KernelArtifact("registry cubin cache: neither $KERN_CACHE_DIR nor $HOME is set".into())
        })?;
    let blobs = cache_root.join("blobs");
    let cached = blobs.join(&sha);
    if let Ok(data) = std::fs::read(&cached) {
        if format!("{:x}", sha2::Sha256::digest(&data)) == sha {
            return Ok(cached);
        }
    }

    let url = format!(
        "https://huggingface.co/{}/{}/resolve/{}/{}",
        reg.org, reg.repo, reg.revision, reg.path
    );
    tracing::info!("fetching {url}");
    let mut req = ureq::get(&url);
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        req = req.set("Authorization", &format!("Bearer {tok}"));
    }
    let resp = req
        .call()
        .map_err(|e| Error::KernelArtifact(format!("fetching {url}: {e}")))?;
    let mut data = Vec::new();
    std::io::Read::read_to_end(&mut resp.into_reader(), &mut data)
        .map_err(|e| Error::KernelArtifact(format!("reading {url}: {e}")))?;
    let got = format!("{:x}", sha2::Sha256::digest(&data));
    if got != sha {
        bail!(
            KernelArtifact,
            "registry cubin hf:{}/{}/{}@{}: sha256 mismatch: manifest declares {sha}, \
             fetched bytes are {got}",
            reg.org,
            reg.repo,
            reg.path,
            reg.revision
        );
    }
    std::fs::create_dir_all(&blobs)?;
    let tmp = blobs.join(format!("{sha}.tmp.{}", std::process::id()));
    std::fs::write(&tmp, &data)?;
    std::fs::rename(&tmp, &cached)?;
    Ok(cached)
}

fn le16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}
fn le32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}
fn le64(b: &[u8], o: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(o..o + 8)?.try_into().ok()?))
}

/// Find a named section in a 64-bit little-endian ELF.
fn elf_section<'a>(bytes: &'a [u8], name: &str) -> Option<&'a [u8]> {
    if bytes.get(..4)? != b"\x7fELF" || *bytes.get(4)? != 2 {
        return None;
    }
    let shoff = le64(bytes, 0x28)? as usize;
    let shentsize = le16(bytes, 0x3a)? as usize;
    let shnum = le16(bytes, 0x3c)? as usize;
    let shstrndx = le16(bytes, 0x3e)? as usize;
    let sh = |i: usize| {
        let base = shoff.checked_add(i.checked_mul(shentsize)?)?;
        Some((
            le32(bytes, base)? as usize,
            le64(bytes, base + 0x18)? as usize,
            le64(bytes, base + 0x20)? as usize,
        ))
    };
    let (_, stroff, strsz) = sh(shstrndx)?;
    let strtab = bytes.get(stroff..stroff.checked_add(strsz)?)?;
    for i in 0..shnum {
        let (n, off, size) = sh(i)?;
        let nm = strtab.get(n..)?.split(|&c| c == 0).next()?;
        if nm == name.as_bytes() {
            return bytes.get(off..off.checked_add(size)?);
        }
    }
    None
}

const FATBIN_MAGIC: u32 = 0xba55_ed50;

/// Split an `.nv_fatbin` section into its fatbin containers (one per
/// translation unit): each is a 16-byte header (magic, version, header size,
/// payload size) followed by the payload, 8-aligned with zero padding.
fn split_fatbins(sec: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off + 16 <= sec.len() && le32(sec, off) == Some(FATBIN_MAGIC) {
        let hsize = le16(sec, off + 6).unwrap() as usize;
        let fatsize = le64(sec, off + 8).unwrap() as usize;
        let Some(end) = hsize.checked_add(fatsize).and_then(|t| off.checked_add(t)) else {
            break;
        };
        let Some(chunk) = sec.get(off..end) else { break };
        out.push(chunk);
        off = end;
        while off < sec.len() && sec[off] == 0 {
            off += 1;
        }
    }
    out
}

/// ELF `e_machine` value for device-code objects: a raw `.cubin` is itself
/// an ELF, but for the CUDA architecture.
const EM_CUDA: u16 = 190;

/// Load every device-code module an artifact carries. A raw cubin loads
/// directly. A host shared library — e.g. a torch-extension kernel package
/// from the HF kernel hub — has its device code embedded in `.nv_fatbin`;
/// the host half of the .so (torch/python bindings) is dead weight to kern,
/// so each embedded fatbin container is loaded on its own and the usual
/// symbol + param-layout resolution picks the right function out.
fn load_device_modules(path: &std::path::Path) -> Result<Vec<sys::CUmodule>> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::KernelArtifact(format!("reading {}: {e}", path.display())))?;
    let host_elf = bytes.get(..4) == Some(b"\x7fELF".as_slice()) && le16(&bytes, 0x12) != Some(EM_CUDA);
    if !host_elf {
        let cpath = CString::new(path.to_str().unwrap())
            .map_err(|e| Error::KernelArtifact(format!("cubin path {}: {e}", path.display())))?;
        let cmod = cu::module::load(cpath)
            .map_err(|e| Error::KernelArtifact(format!("loading {}: {e:?}", path.display())))?;
        return Ok(vec![cmod]);
    }
    let sec = elf_section(&bytes, ".nv_fatbin").ok_or_else(|| {
        Error::KernelArtifact(format!("{}: host ELF without a .nv_fatbin section", path.display()))
    })?;
    let chunks = split_fatbins(sec);
    if chunks.is_empty() {
        bail!(KernelArtifact, "{}: .nv_fatbin holds no fatbin containers", path.display());
    }
    let mut mods = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        // Copy to an 8-aligned buffer; the slice into the ELF has no
        // alignment guarantee.
        let mut buf = vec![0u64; chunk.len().div_ceil(8)];
        unsafe {
            std::ptr::copy_nonoverlapping(chunk.as_ptr(), buf.as_mut_ptr() as *mut u8, chunk.len());
        }
        let mut m: sys::CUmodule = std::ptr::null_mut();
        let r = unsafe { sys::cuModuleLoadData(&mut m, buf.as_ptr() as *const c_void) };
        match r {
            sys::CUresult::CUDA_SUCCESS => mods.push(m),
            // A container may hold only device code this GPU can't use
            // (e.g. relocatable-only); resolution below just won't see it.
            _ => tracing::warn!("{}: fatbin container #{i} not loadable: {r:?}", path.display()),
        }
    }
    if mods.is_empty() {
        bail!(KernelArtifact, "{}: no loadable fatbin container", path.display());
    }
    Ok(mods)
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
        kern_manifest::verify(&manifest)?;

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

        // Materialize registry refs (`hf:...`) into the content-addressed
        // cache. Their module key is the full ref string, so per-step cubin
        // pinning below matches them like any local file name.
        let mut remote: BTreeMap<String, std::path::PathBuf> = BTreeMap::new();
        for (name, k) in &manifest.kernels {
            for (si, st) in k.imp.steps.iter().enumerate() {
                let Some(reg) = st.cubin.as_deref().and_then(RegistryRef::parse) else {
                    continue;
                };
                let reg = reg.map_err(|e| Error::Manifest(format!("kernel `{name}` step #{si}: {e}")))?;
                let cb = st.cubin.as_deref().unwrap();
                if remote.contains_key(cb) {
                    continue;
                }
                // The verifier enforces sha256 on registry refs.
                let sha = st.sha256.as_deref().ok_or_else(|| {
                    Error::Manifest(format!("kernel `{name}` step #{si}: registry cubin without sha256"))
                })?;
                let path = fetch_registry_cubin(&reg, sha)?;
                remote.insert(cb.to_string(), path);
            }
        }

        if cubins.is_empty() && remote.is_empty() {
            bail!(KernelArtifact, "no .cubin files in {}", kernels_dir.display());
        }
        let mut modules = Vec::new();
        let local = cubins
            .iter()
            .map(|p| (p.file_name().unwrap().to_string_lossy().into_owned(), p.clone()));
        for (key, path) in local.chain(remote.iter().map(|(r, p)| (r.clone(), p.clone()))) {
            for cmod in load_device_modules(&path)? {
                modules.push((key.clone(), cmod));
            }
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
                        _ => bail!(Manifest, "kernel `{name}` step #{si}: unsupported extern op `{ext}`"),
                    }
                    continue;
                }
                // Pinned-artifact integrity: when the step names its cubin
                // (the pluggable path), verify the file hash if declared.
                if let (Some(cb), Some(sha)) = (&st.cubin, &st.sha256) {
                    let path = match remote.get(cb.as_str()) {
                        Some(p) => p.clone(),
                        None => kernels_dir.join(cb),
                    };
                    let data = std::fs::read(&path).map_err(|e| {
                        Error::KernelArtifact(format!(
                            "kernel `{name}` step #{si}: reading {}: {e}",
                            path.display()
                        ))
                    })?;
                    let got = format!("{:x}", sha2::Sha256::digest(&data));
                    if got != sha.to_lowercase() {
                        bail!(
                            KernelArtifact,
                            "kernel `{name}` step #{si}: cubin `{cb}` sha256 mismatch: \
                             manifest declares {sha}, file is {got}"
                        );
                    }
                }
                let want: Vec<usize> =
                    st.params.iter().map(|p| p.size_bytes() as usize).collect();
                let sym = CString::new(st.symbol.as_str())
                    .map_err(|e| Error::Manifest(format!("kernel `{name}` symbol: {e}")))?;
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
                        KernelArtifact,
                        "kernel `{name}` step #{si} ({}): no loaded instance matches declared \
                         param layout {want:?} (cubin filter {:?}); saw {seen:?}",
                        st.symbol,
                        st.cubin
                    );
                };
                // Opt in to >48KB dynamic shared memory where the step needs it.
                if let (StepImpl::Cubin { func, .. }, Some(sm)) = (&r, &st.shared_mem) {
                    let bytes = sm
                        .eval(&max_env)
                        .map_err(|e| Error::Manifest(format!("kernel `{name}`: {e}")))?;
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
                    elems = elems
                        .checked_mul(n)
                        .ok_or_else(|| Error::Manifest("scratch size overflow".into()))?;
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
                elems = elems
                    .checked_mul(n)
                    .ok_or_else(|| Error::Manifest("buffer size overflow".into()))?;
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
        let st = safetensors::SafeTensors::deserialize(blob)
            .map_err(|e| Error::WeightArtifact(format!("unparseable safetensors: {e}")))?;
        for (name, b) in &self.manifest.buffers {
            if b.class != BufferClass::Weight {
                continue;
            }
            let t = st.tensor(name).map_err(|e| {
                Error::WeightArtifact(format!("weight `{name}` missing from artifact: {e}"))
            })?;
            let dst = self.buffers.get_mut(name).unwrap();
            if t.data().len() as u64 != dst.bytes {
                bail!(
                    WeightArtifact,
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
            bail!(Api, "no buffer `{name}`");
        };
        if b.class != BufferClass::Input {
            bail!(Api, "buffer `{name}` is {}, not input", b.class);
        }
        let dst = self.buffers.get_mut(name).unwrap();
        if data.len() as u64 > dst.bytes {
            bail!(Api, "input `{name}`: got {} bytes, buffer is {}", data.len(), dst.bytes);
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
            bail!(Api, "no buffer `{name}`");
        };
        if b.class != BufferClass::Output {
            bail!(Api, "buffer `{name}` is {}, not output", b.class);
        }
        Ok(self.stream.clone_dtoh(&self.buffers[name].slice)?)
    }

    fn check_env(&self, env: &BTreeMap<String, u64>) -> Result<()> {
        for (sym, decl) in &self.manifest.symbols {
            let Some(v) = env.get(sym) else {
                bail!(Api, "symbol `{sym}` not provided");
            };
            if *v < decl.min || *v > decl.max {
                bail!(Api, "symbol `{sym}` = {v} outside declared [{}, {}]", decl.min, decl.max);
            }
        }
        Ok(())
    }

    /// Execute one program with the given symbol values, then synchronize.
    pub fn run(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.manifest.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        self.check_env(env)?;
        self.ctx.bind_to_thread()?;
        for (i, d) in prog.dispatches.iter().enumerate() {
            self.dispatch(d, env).map_err(|e| Error::Dispatch {
                context: dispatch_ctx(i, d),
                source: Box::new(e),
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
            bail!(Api, "no program `{program}`");
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
                failed = Some(Error::Dispatch {
                    context: dispatch_ctx(i, d),
                    source: Box::new(e),
                });
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
            return Err(e);
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
            bail!(Api, "program `{program}` has not been captured");
        };
        if captured != env {
            bail!(Api, "graph for `{program}` was captured with {captured:?}, called with {env:?}");
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
            _ => bail!(Manifest, "expected buffer/state arg, got {arg}"),
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
            _ => bail!(Manifest, "expected scalar arg, got {arg}"),
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
                self.gemm_bf16_tn(&slots, *beta).map_err(|e| Error::Dispatch {
                    context: format!("step #{si}"),
                    source: Box::new(e),
                })?;
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
                .map_err(|e| Error::Cuda(format!("step #{si} ({}): {e:?}", st.symbol)))?;
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
            bail!(Manifest, "gemm expects 6 args, got {}", args.len());
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
                .map_err(|e| Error::Cuda(format!("cublasLt matmul (m={m} n={n} k={k}): {e:?}")))?;
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
