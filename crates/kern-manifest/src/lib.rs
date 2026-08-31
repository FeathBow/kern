//! kern-manifest: the typed execution manifest a model provider ships
//! alongside compiled kernels (`manifest.json + kernels.cubin + weights`),
//! and the load-time verifier that refuses anything not provably consistent.
//!
//! The runtime assigns no meaning to any name in the manifest. It schedules
//! opaque kernel dispatches, provisions opaque per-token state bytes, and
//! evaluates a closed set of scalar expressions for launch geometry. All
//! model semantics live on the provider's side of this boundary.

pub mod types;
pub mod verify;

pub use types::Manifest;
pub use verify::{verify, VerifyErrors};
