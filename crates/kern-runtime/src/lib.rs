//! Thin verifier-driven executor for kern manifests.
//!
//! The runtime knows nothing about models. It loads a verified manifest,
//! resolves each declared kernel against the cubins in a directory, allocates
//! every buffer/state, binds weight buffers by name from a safetensors blob,
//! and replays the program's dispatch list. The only kernels it understands
//! natively are `extern:` ops (currently `extern:cublaslt_bf16_tn`).
//!
//! Names stop at load time: device pointers are static once buffers, states
//! and scratch are allocated, so [`Runtime::load`] lowers every program into
//! a flat launch list (see `compile`) whose slots are finished values or
//! symbol-indexed expressions. The name-keyed maps that remain exist only on
//! the caller API surface (`write_input("token_ids")`, `run("decode")`);
//! the execution path performs no name lookups.
//!
//! Same-name Triton kernels ship multiple constexpr instances with different
//! ABIs across modules; resolution picks the instance whose
//! `cuFuncGetParamInfo` layout matches the manifest's declared params — the
//! phase-2 ABI check doubles as instance selection.

mod compile;
mod cubin;
mod device;
mod error;

use std::collections::BTreeMap;
use std::os::raw::c_void;
use std::sync::Arc;

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{result as cu, sys, CudaContext, CudaStream, PinnedHostSlice};
use kern_manifest::types::{BufferClass, Manifest};

use compile::{CompiledProgram, Launch, LaunchKind, RVal, Slot};
use device::{alloc, gemm_bf16_tn, DeviceBuf};
use error::{bail, cuda_check};
pub use error::{Error, Result};

pub struct Runtime {
    pub manifest: Manifest,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blt: CudaBlasLT,
    /// Name-keyed because names are the caller API (`write_input`,
    /// `read_output`, weight binding); execution never looks these up —
    /// their device pointers are baked into `programs`.
    buffers: BTreeMap<String, DeviceBuf>,
    states: BTreeMap<String, DeviceBuf>,
    /// Persistent pinned staging, one per input buffer: H2D from pageable
    /// memory degrades to a synchronous driver-staged copy (tens of µs per
    /// call); through page-locked staging it is a true async DMA. The pinned
    /// slice's event guards reuse across steps.
    staging: BTreeMap<String, PinnedHostSlice<u8>>,
    /// Programs lowered to flat launch lists at load.
    programs: BTreeMap<String, CompiledProgram>,
    /// Owners of the impl-private scratch allocations whose pointers are
    /// baked into `programs`.
    #[allow(dead_code)]
    scratch: Vec<DeviceBuf>,
    /// Per kernel: the module each impl step resolved to (introspection).
    resolution: Vec<(String, Vec<String>)>,
    n_modules: usize,
    /// Program name -> instantiated CUDA graph + the dense symbol values it
    /// was captured with (grid dims and scalar args are baked in at capture).
    graphs: BTreeMap<String, (sys::CUgraphExec, Vec<u64>)>,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        for (exec, _) in self.graphs.values() {
            unsafe { sys::cuGraphExecDestroy(*exec) };
        }
    }
}

impl Runtime {
    /// Verify the manifest, load every `*.cubin` under `kernels_dir`, resolve
    /// kernels, allocate all buffers and states, and lower every program.
    /// `state_capacity_tokens` scales each declared state by its
    /// `bytes_per_token`.
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

        let remote = cubin::fetch_registry_cubins(&manifest)?;
        let modules = cubin::load_all_modules(kernels_dir, &remote)?;

        let max_env: BTreeMap<_, _> =
            manifest.symbols.iter().map(|(s, v)| (s.clone(), v.max)).collect();

        // Buffer sizes are static: shapes only reference symbols, sized at max.
        let mut buffers = BTreeMap::new();
        for (name, b) in &manifest.buffers {
            let bytes = compile::shaped_bytes(
                &format!("buffer `{name}`"),
                &b.shape,
                b.dtype.bytes(),
                &max_env,
            )?;
            buffers.insert(name.clone(), alloc(&stream, bytes)?);
        }
        let mut states = BTreeMap::new();
        for (name, s) in &manifest.states {
            let bytes = s
                .bytes_per_token
                .checked_mul(state_capacity_tokens)
                .ok_or_else(|| Error::Manifest(format!("state `{name}`: size overflow")))?;
            states.insert(name.clone(), alloc(&stream, bytes)?);
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

        let resolved =
            compile::resolve_kernels(&manifest, &modules, &remote, kernels_dir, &stream, &max_env)?;
        let resolution = resolved.iter().map(|(n, rk)| (n.clone(), rk.step_modules())).collect();
        let programs = compile::compile_programs(&manifest, &resolved, &buffers, &states)?;
        let scratch = resolved.into_values().flat_map(|rk| rk.scratch.into_values()).collect();

        Ok(Runtime {
            manifest,
            ctx,
            stream,
            blt,
            buffers,
            states,
            staging,
            programs,
            scratch,
            resolution,
            n_modules: modules.len(),
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
        self.resolution.clone()
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

    /// Validate the caller's symbol values and densify them into manifest
    /// symbol order — the index space every compiled expression uses.
    fn dense_env(&self, env: &BTreeMap<String, u64>) -> Result<Vec<u64>> {
        self.manifest
            .symbols
            .iter()
            .map(|(sym, decl)| {
                let Some(&v) = env.get(sym) else {
                    bail!(Api, "symbol `{sym}` not provided");
                };
                if v < decl.min || v > decl.max {
                    bail!(Api, "symbol `{sym}` = {v} outside declared [{}, {}]", decl.min, decl.max);
                }
                Ok(v)
            })
            .collect()
    }

    /// `sym=value` in manifest symbol order, for error messages.
    fn fmt_env(&self, env: &[u64]) -> String {
        self.manifest
            .symbols
            .keys()
            .zip(env)
            .map(|(s, v)| format!("{s}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Execute one program with the given symbol values, then synchronize.
    pub fn run(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let env = self.dense_env(env)?;
        self.ctx.bind_to_thread()?;
        self.replay(prog, &env)?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Capture one program into an instantiated CUDA graph. Grid dims and
    /// scalar args (symbol values included) are baked in at capture; input
    /// buffer *contents* are read at replay, so per-step H2D writes stay
    /// outside the graph and `run_captured` replays the whole dispatch list
    /// with one launch.
    pub fn capture(&mut self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some(prog) = self.programs.get(program) else {
            bail!(Api, "no program `{program}`");
        };
        let env = self.dense_env(env)?;
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
        let replayed = self.replay(prog, &env);
        // Always end the capture, even on error — a stream stuck in capture
        // mode poisons every later operation on it.
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let end = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
        if let Err(e) = replayed {
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
        if let Some((old, _)) = self.graphs.insert(program.to_string(), (exec, env)) {
            unsafe { sys::cuGraphExecDestroy(old) };
        }
        Ok(())
    }

    /// Replay a previously captured program, then synchronize.
    pub fn run_captured(&self, program: &str, env: &BTreeMap<String, u64>) -> Result<()> {
        let Some((exec, captured)) = self.graphs.get(program) else {
            bail!(Api, "program `{program}` has not been captured");
        };
        let env = self.dense_env(env)?;
        if *captured != env {
            bail!(
                Api,
                "graph for `{program}` was captured with {{{}}}, called with {{{}}}",
                self.fmt_env(captured),
                self.fmt_env(&env)
            );
        }
        self.ctx.bind_to_thread()?;
        cuda_check(
            unsafe { sys::cuGraphLaunch(*exec, self.stream.cu_stream()) },
            "cuGraphLaunch",
        )?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Issue every launch of a compiled program onto the stream (no sync).
    fn replay(&self, prog: &CompiledProgram, env: &[u64]) -> Result<()> {
        for l in &prog.launches {
            self.launch(l, env).map_err(|e| Error::Dispatch {
                context: l.ctx.clone(),
                source: Box::new(e),
            })?;
        }
        Ok(())
    }

    fn launch(&self, l: &Launch, env: &[u64]) -> Result<()> {
        // Materialize the slots; only symbol-dependent scalars are left to
        // compute, everything else was finished at load.
        let mut vals = Vec::with_capacity(l.slots.len());
        for s in &l.slots {
            vals.push(match s {
                Slot::Const(rv) => *rv,
                Slot::Expr(e) => RVal { val: e.eval(env)?, bytes: 0 },
            });
        }
        match &l.kind {
            LaunchKind::Gemm { beta } => gemm_bf16_tn(&self.blt, &self.stream, &vals, *beta),
            LaunchKind::Cubin { func, block, grid, shared_mem } => {
                let grid =
                    (grid[0].eval(env)? as u32, grid[1].eval(env)? as u32, grid[2].eval(env)? as u32);
                let smem = match shared_mem {
                    Some(e) => e.eval(env)? as u32,
                    None => 0,
                };
                // Every param slot staged as a little-endian u64; the launch
                // ABI reads the low `size_bytes()` of each slot.
                let raw: Vec<u64> = vals.iter().map(|r| r.val).collect();
                let mut params: Vec<*mut c_void> =
                    raw.iter().map(|s| s as *const u64 as *mut c_void).collect();
                unsafe {
                    cu::launch_kernel(
                        *func,
                        grid,
                        (block[0], block[1], block[2]),
                        smem,
                        self.stream.cu_stream(),
                        &mut params,
                    )
                    .map_err(|e| Error::Cuda(format!("cuLaunchKernel: {e:?}")))
                }
            }
        }
    }
}
