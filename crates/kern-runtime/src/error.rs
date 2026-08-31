//! Runtime errors, grouped by who has to act on them.

use cudarc::driver::sys;

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

    /// One launch of a program failed; `context` locates it in the
    /// program's dispatch list, `source` is the underlying failure.
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
    ($variant:ident, $($t:tt)*) => { return Err($crate::Error::$variant(format!($($t)*))) };
}
pub(crate) use bail;

pub(crate) fn cuda_check(r: sys::CUresult, what: &str) -> Result<()> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        bail!(Cuda, "{what}: {r:?}")
    }
}
