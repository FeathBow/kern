//! Device allocations — pool allocations and virtual-memory mappings that
//! other ranks can import — and the runtime's built-in extern ops (cublasLt).

use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublas;
use cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
use cudarc::driver::{sys, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use half::bf16;

use crate::compile::RVal;
use crate::error::{bail, cuda_check, Error, Result};

/// A fabric handle another process — on this tray or across the NVL72
/// fabric — can map with [`import`]. `bytes` is the mapped size (the
/// requested size rounded up to the allocation granularity), which the
/// importer must map in full.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PeerHandle {
    pub fabric: [u8; 64],
    pub bytes: u64,
}

impl PeerHandle {
    pub const BYTES: usize = 72;

    /// Wire form: 64 bytes of fabric handle, then the mapped size, little-endian.
    pub fn to_bytes(&self) -> [u8; Self::BYTES] {
        let mut out = [0u8; Self::BYTES];
        out[..64].copy_from_slice(&self.fabric);
        out[64..].copy_from_slice(&self.bytes.to_le_bytes());
        out
    }

    pub fn from_bytes(b: &[u8]) -> Option<PeerHandle> {
        if b.len() != Self::BYTES {
            return None;
        }
        Some(PeerHandle { fabric: b[..64].try_into().ok()?, bytes: u64::from_le_bytes(b[64..].try_into().ok()?) })
    }
}

impl std::fmt::Debug for PeerHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PeerHandle({} bytes, {:02x}{:02x}{:02x}{:02x}…)",
            self.bytes, self.fabric[0], self.fabric[1], self.fabric[2], self.fabric[3]
        )
    }
}

/// One device allocation. `ptr`/`bytes` are what the compiled programs
/// bake in; the backing decides whether a peer can map it.
pub(crate) struct DeviceBuf {
    pub(crate) ptr: u64,
    pub(crate) bytes: u64,
    stream: Arc<CudaStream>,
    backing: Backing,
}

enum Backing {
    /// `cuMemAlloc` through cudarc: local only.
    Pool(#[allow(dead_code)] CudaSlice<u8>),
    /// `cuMemCreate` + reserve + map: exportable when created with a fabric
    /// handle, or a peer's allocation mapped into this address space.
    Vmm(Vmm),
}

/// A physical allocation mapped at a reserved address; unmapped, freed and
/// released in that order on drop.
struct Vmm {
    handle: sys::CUmemGenericAllocationHandle,
    va: sys::CUdeviceptr,
    size: usize,
    shareable: bool,
}

impl Drop for Vmm {
    fn drop(&mut self) {
        unsafe {
            sys::cuMemUnmap(self.va, self.size);
            sys::cuMemAddressFree(self.va, self.size);
            sys::cuMemRelease(self.handle);
        }
    }
}

/// How a state or exported buffer asks for a shareable handle.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Share {
    /// A fabric handle or nothing (a state on a device without one).
    IfSupported,
    /// A fabric handle or an error (an `export: true` buffer).
    Required,
}

/// Pool allocation, zeroed on the stream.
pub(crate) fn alloc(stream: &Arc<CudaStream>, bytes: u64) -> Result<DeviceBuf> {
    let slice = stream.alloc_zeros::<u8>(bytes.max(1) as usize)?;
    let ptr = {
        let (p, _sync) = slice.device_ptr(stream);
        p
    };
    Ok(DeviceBuf { ptr, bytes, stream: stream.clone(), backing: Backing::Pool(slice) })
}

fn fabric_handle_type() -> sys::CUmemAllocationHandleType {
    sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC
}

fn none_handle_type() -> sys::CUmemAllocationHandleType {
    sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE
}

/// Whether the device can hand out fabric handles (`cuMemCreate` with
/// `CU_MEM_HANDLE_TYPE_FABRIC`).
pub(crate) fn fabric_supported(dev: i32) -> Result<bool> {
    let mut v: i32 = 0;
    cuda_check(
        unsafe {
            sys::cuDeviceGetAttribute(
                &mut v,
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                dev,
            )
        },
        "cuDeviceGetAttribute(HANDLE_TYPE_FABRIC_SUPPORTED)",
    )?;
    Ok(v != 0)
}

fn alloc_prop(dev: i32, handle_type: sys::CUmemAllocationHandleType) -> sys::CUmemAllocationProp {
    // Zeroed, then the fields that matter: the struct grows across CUDA
    // versions and cudarc's bindings follow the one it was built against.
    let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.requestedHandleTypes = handle_type;
    prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop
}

fn granularity(prop: &sys::CUmemAllocationProp) -> Result<usize> {
    let mut g: usize = 0;
    cuda_check(
        unsafe {
            sys::cuMemGetAllocationGranularity(
                &mut g,
                prop,
                sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        },
        "cuMemGetAllocationGranularity",
    )?;
    Ok(g.max(1))
}

/// Reserve `size` bytes of address space, map `handle` there and grant
/// this device read/write access. On failure nothing leaks: the handle is
/// the caller's to release.
fn map_handle(
    dev: i32,
    handle: sys::CUmemGenericAllocationHandle,
    size: usize,
    align: usize,
) -> Result<sys::CUdeviceptr> {
    let mut va: sys::CUdeviceptr = 0;
    cuda_check(unsafe { sys::cuMemAddressReserve(&mut va, size, align, 0, 0) }, "cuMemAddressReserve")?;
    if let Err(e) = cuda_check(unsafe { sys::cuMemMap(va, size, 0, handle, 0) }, "cuMemMap") {
        unsafe { sys::cuMemAddressFree(va, size) };
        return Err(e);
    }
    let mut access: sys::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = dev;
    access.flags = sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if let Err(e) = cuda_check(unsafe { sys::cuMemSetAccess(va, size, &access, 1) }, "cuMemSetAccess") {
        unsafe {
            sys::cuMemUnmap(va, size);
            sys::cuMemAddressFree(va, size);
        }
        return Err(e);
    }
    Ok(va)
}

/// Virtual-memory allocation on `dev`, zeroed on the stream. With
/// `Share::Required` the allocation carries a fabric handle or the call
/// fails; with `Share::IfSupported` it carries one when the device offers
/// them and is a plain local mapping otherwise.
pub(crate) fn alloc_vmm(stream: &Arc<CudaStream>, dev: i32, bytes: u64, share: Share, what: &str) -> Result<DeviceBuf> {
    let fabric = fabric_supported(dev)?;
    if !fabric && share == Share::Required {
        bail!(Cuda, "{what}: device {dev} does not support fabric handles, cannot export");
    }
    let prop = alloc_prop(dev, if fabric { fabric_handle_type() } else { none_handle_type() });
    let g = granularity(&prop)?;
    let size = (bytes.max(1) as usize).div_ceil(g) * g;
    let mut handle: sys::CUmemGenericAllocationHandle = 0;
    let created = unsafe { sys::cuMemCreate(&mut handle, size, &prop, 0) };
    let (handle, shareable) = match (created, fabric, share) {
        (sys::CUresult::CUDA_SUCCESS, _, _) => (handle, fabric),
        // The attribute says fabric, the driver says no (no IMEX channel,
        // say): a state falls back to a local mapping, an export cannot.
        (r, true, Share::IfSupported) => {
            tracing::warn!("{what}: cuMemCreate with a fabric handle failed ({r:?}); allocating without one");
            let prop = alloc_prop(dev, none_handle_type());
            cuda_check(unsafe { sys::cuMemCreate(&mut handle, size, &prop, 0) }, "cuMemCreate")?;
            (handle, false)
        }
        (r, _, _) => return Err(Error::Cuda(format!("{what}: cuMemCreate({size} bytes, fabric={fabric}): {r:?}"))),
    };
    let va = match map_handle(dev, handle, size, g) {
        Ok(va) => va,
        Err(e) => {
            unsafe { sys::cuMemRelease(handle) };
            return Err(e);
        }
    };
    let vmm = Vmm { handle, va, size, shareable };
    cuda_check(unsafe { sys::cuMemsetD8Async(va, 0, size, stream.cu_stream()) }, "cuMemsetD8Async")?;
    Ok(DeviceBuf { ptr: va, bytes, stream: stream.clone(), backing: Backing::Vmm(vmm) })
}

/// Map a peer's exported allocation into this device's address space.
/// The mapping is a [`DeviceBuf`] so it lives exactly as long as the
/// pointers derived from it.
pub(crate) fn import(stream: &Arc<CudaStream>, dev: i32, h: &PeerHandle, what: &str) -> Result<DeviceBuf> {
    let mut fh = sys::CUmemFabricHandle { data: h.fabric };
    let mut handle: sys::CUmemGenericAllocationHandle = 0;
    cuda_check(
        unsafe {
            sys::cuMemImportFromShareableHandle(&mut handle, &mut fh as *mut _ as *mut c_void, fabric_handle_type())
        },
        &format!("{what}: cuMemImportFromShareableHandle"),
    )?;
    let g = granularity(&alloc_prop(dev, fabric_handle_type()))?;
    let size = h.bytes as usize;
    if size == 0 || !size.is_multiple_of(g) {
        unsafe { sys::cuMemRelease(handle) };
        bail!(Cuda, "{what}: peer handle maps {size} bytes, not a multiple of the {g}-byte granularity");
    }
    let va = match map_handle(dev, handle, size, g) {
        Ok(va) => va,
        Err(e) => {
            unsafe { sys::cuMemRelease(handle) };
            return Err(Error::Cuda(format!("{what}: {e}")));
        }
    };
    let vmm = Vmm { handle, va, size, shareable: false };
    Ok(DeviceBuf { ptr: va, bytes: h.bytes, stream: stream.clone(), backing: Backing::Vmm(vmm) })
}

impl DeviceBuf {
    /// The fabric handle a peer imports, for an allocation that has one.
    pub(crate) fn export(&self) -> Result<Option<PeerHandle>> {
        let Backing::Vmm(v) = &self.backing else { return Ok(None) };
        if !v.shareable {
            return Ok(None);
        }
        let mut fh = sys::CUmemFabricHandle { data: [0; 64] };
        cuda_check(
            unsafe {
                sys::cuMemExportToShareableHandle(&mut fh as *mut _ as *mut c_void, v.handle, fabric_handle_type(), 0)
            },
            "cuMemExportToShareableHandle",
        )?;
        Ok(Some(PeerHandle { fabric: fh.data, bytes: v.size as u64 }))
    }

    /// Whether this allocation carries a fabric handle.
    pub(crate) fn is_shareable(&self) -> bool {
        matches!(&self.backing, Backing::Vmm(v) if v.shareable)
    }

    /// A byte range of the allocation, for the cudarc copy/memset entry
    /// points.
    pub(crate) fn view(&self, range: std::ops::Range<usize>) -> Result<BufView> {
        if range.start > range.end || range.end as u64 > self.bytes {
            bail!(Api, "byte range [{}, {}) outside the {}-byte allocation", range.start, range.end, self.bytes);
        }
        Ok(BufView { ptr: self.ptr + range.start as u64, len: range.end - range.start, stream: self.stream.clone() })
    }
}

/// Synchronization is trivially correct for these raw views: the whole
/// runtime is single-stream.
impl DeviceSlice<u8> for DeviceBuf {
    fn len(&self) -> usize {
        self.bytes as usize
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<u8> for DeviceBuf {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<u8> for DeviceBuf {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// A byte range of a [`DeviceBuf`].
pub(crate) struct BufView {
    ptr: u64,
    len: usize,
    stream: Arc<CudaStream>,
}

impl DeviceSlice<u8> for BufView {
    fn len(&self) -> usize {
        self.len
    }
    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl DevicePtr<u8> for BufView {
    fn device_ptr<'a>(&'a self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl DevicePtrMut<u8> for BufView {
    fn device_ptr_mut<'a>(&'a mut self, _: &'a CudaStream) -> (sys::CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

/// Raw device pointer presented as a `DevicePtr<bf16>`/`DevicePtrMut<bf16>`
/// for the cublasLt extern op.
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

/// `extern:cublaslt_bf16_tn`: row-major `C[m,n] = A[m,k] @ W[n,k]^T`,
/// resolved args `[a, w, c, m, n, k]`. Column-major mapping: compute
/// `C_cm[n,m] = W_cm^T[n,k] x A_cm[k,m]` -> transa=T on W (lda=k),
/// transb=N on A (ldb=k), m'=n, n'=m, ldc=n.
/// `extern:cublaslt_bf16_tn_acc` is the same with beta=1: `C += A @ W^T`.
pub(crate) fn gemm_bf16_tn(blt: &CudaBlasLT, stream: &Arc<CudaStream>, args: &[RVal], beta: f32) -> Result<()> {
    // `c[m, n] (+)= a[m, k] @ w[n, k]^T`; an optional 7th arg is C's row
    // stride in elements (default n) so a call can write every `ldc`-th row.
    let (a, w, c, m, n, k, ldc) = match args {
        [a, w, c, m, n, k] => (a, w, c, m.val, n.val, k.val, n.val),
        [a, w, c, m, n, k, ldc] => (a, w, c, m.val, n.val, k.val, ldc.val),
        _ => bail!(Manifest, "gemm expects 6 or 7 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "gemm: ldc {ldc} < n {n}");
    }
    let view = |rv: &RVal| RawBf16 { ptr: rv.val, len: (rv.bytes / 2) as usize, stream: stream.clone() };
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
        ldc: ldc as i64,
        stride_a: None,
        stride_b: None,
        stride_c: None,
        stride_bias: None,
        batch_size: None,
    };
    let mut out = view(c);
    unsafe {
        blt.matmul(cfg, &view(w), &view(a), &mut out, None, None)
            .map_err(|e| Error::Cuda(format!("cublasLt matmul (m={m} n={n} k={k}): {e:?}")))?;
    }
    Ok(())
}

/// A cuBLAS handle bound to the runtime's stream with its own workspace, for
/// the f32-result GEMM built-in (`cublasGemmEx`; cublasLt's typed `Matmul`
/// only lands in the operand type). Kept separate from the Lt handle so the
/// two never share a workspace.
pub(crate) struct Blas {
    handle: cublas::sys::cublasHandle_t,
    _workspace: CudaSlice<u8>,
}

// The handle is only ever used from the runtime's own thread, on its stream.
unsafe impl Send for Blas {}
unsafe impl Sync for Blas {}

impl Blas {
    const WORKSPACE: usize = 32 << 20;

    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Blas> {
        let handle = cublas::result::create_handle().map_err(|e| Error::Cuda(format!("cublasCreate: {e:?}")))?;
        let workspace: CudaSlice<u8> = stream.alloc_zeros(Self::WORKSPACE)?;
        unsafe {
            cublas::result::set_stream(handle, stream.cu_stream() as *mut _)
                .map_err(|e| Error::Cuda(format!("cublasSetStream: {e:?}")))?;
            let (ws, _g) = workspace.device_ptr(stream);
            cublas::sys::cublasSetWorkspace_v2(handle, ws as *mut c_void, Self::WORKSPACE)
                .result()
                .map_err(|e| Error::Cuda(format!("cublasSetWorkspace: {e:?}")))?;
            cublas::sys::cublasSetMathMode(handle, cublas::sys::cublasMath_t::CUBLAS_TENSOR_OP_MATH)
                .result()
                .map_err(|e| Error::Cuda(format!("cublasSetMathMode: {e:?}")))?;
        }
        Ok(Blas { handle, _workspace: workspace })
    }
}

impl Drop for Blas {
    fn drop(&mut self) {
        unsafe {
            let _ = cublas::sys::cublasDestroy_v2(self.handle);
        }
    }
}

/// `extern:cublas_bf16_tn_f32`: row-major `C[m,n] = A[m,k] @ W[n,k]^T` with
/// bf16 operands, f32 accumulation and an **f32** result — cublasGemmEx with
/// `CUBLAS_COMPUTE_32F` / `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, the call an
/// engine that lands its own bf16 partials makes. Args as
/// [`gemm_bf16_tn`]: `[a, w, c, m, n, k]` or 7 with C's row stride.
pub(crate) fn gemm_bf16_tn_f32(blas: &Blas, args: &[RVal]) -> Result<()> {
    let (a, w, c, m, n, k, ldc) = match args {
        [a, w, c, m, n, k] => (a, w, c, m.val, n.val, k.val, n.val),
        [a, w, c, m, n, k, ldc] => (a, w, c, m.val, n.val, k.val, ldc.val),
        _ => bail!(Manifest, "gemm expects 6 or 7 args, got {}", args.len()),
    };
    if ldc < n {
        bail!(Manifest, "gemm: ldc {ldc} < n {n}");
    }
    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if a.bytes < m * k * 2 || w.bytes < n * k * 2 || c.bytes < ((m - 1) * ldc + n) * 4 {
        bail!(
            Manifest,
            "gemm f32: operands too small for m={m} n={n} k={k} ldc={ldc}: a {} B, w {} B, c {} B",
            a.bytes,
            w.bytes,
            c.bytes
        );
    }
    let dim = |v: u64| i32::try_from(v).map_err(|_| Error::Manifest(format!("gemm f32: dimension {v} exceeds i32")));
    let (alpha, beta) = (1.0f32, 0.0f32);
    use cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType};
    unsafe {
        cublas::result::gemm_ex(
            blas.handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            dim(n)?,
            dim(m)?,
            dim(k)?,
            &alpha as *const f32 as *const c_void,
            w.val as *const c_void,
            cudaDataType::CUDA_R_16BF,
            dim(k)?,
            a.val as *const c_void,
            cudaDataType::CUDA_R_16BF,
            dim(k)?,
            &beta as *const f32 as *const c_void,
            c.val as *mut c_void,
            cudaDataType::CUDA_R_32F,
            dim(ldc)?,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
        .map_err(|e| Error::Cuda(format!("cublasGemmEx f32 (m={m} n={n} k={k}): {e:?}")))?;
    }
    Ok(())
}
