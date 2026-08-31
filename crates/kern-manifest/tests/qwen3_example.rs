//! The mined Qwen3-4B decode manifest must parse and verify clean: real
//! vLLM cubin ABIs for the flat kernels, wiring cross-checked against a
//! captured decode forward. Regenerate with
//! `python3 tools/gen_qwen3_decode.py` (needs the capture dump).

use kern_manifest::{verify, Manifest};

const QWEN3: &str = include_str!("../../../examples/qwen3-4b.json");

#[test]
fn qwen3_decode_mined_verifies() {
    let m = Manifest::from_json(QWEN3).expect("parse");
    if let Err(errs) = verify(&m) {
        panic!("mined manifest failed verification:\n{}", errs.join("\n"));
    }
    // decode: embed + l0 norm + 36 layers x 12 + lm_head + sample; attention's
    // reduce step and argmax's two stages live inside their kernel impls.
    assert_eq!(m.programs["decode"].dispatches.len(), 2 + 36 * 12 + 2);
    // prefill: same forward minus the final_norm/lm_head/sample tail — the
    // last prompt token goes through `decode` instead.
    assert_eq!(m.programs["prefill"].dispatches.len(), 2 + 36 * 12 - 1);
    // The runtime's entire knowledge of the KV cache: a byte count.
    assert_eq!(m.states["kv"].bytes_per_token, 36 * 2 * 1024 * 2);

    let again = Manifest::from_json(&m.to_json()).expect("reparse");
    assert_eq!(m.to_json(), again.to_json());

    // Domains: the structural inputs carry priors the runtime enforces and
    // attestation synthesizes from; activations deliberately carry none.
    for name in ["token_ids", "slot_mapping", "block_table", "cu_seqlens_q", "next_token"] {
        assert!(m.buffers[name].domain.is_some(), "{name} has no domain");
    }
    assert!(m.buffers["residual"].domain.is_none());
}

/// The A/B fixture for `kern-attest`: identical to qwen3-4b.json except
/// `silu_mul`'s *impl* is the mined vLLM cubin (6-param launch ABI) instead
/// of the HF hub package (3-param ABI forwarding interface args 0..3). The
/// interface and every dispatch are untouched — a pure impl swap.
const QWEN3_SILU_MINED: &str = include_str!("../../../examples/qwen3-4b-silu-mined.json");

#[test]
fn qwen3_silu_mined_fixture_verifies() {
    let m = Manifest::from_json(QWEN3_SILU_MINED).expect("parse");
    if let Err(errs) = verify(&m) {
        panic!("silu-mined fixture failed verification:\n{}", errs.join("\n"));
    }
    let a = Manifest::from_json(QWEN3).unwrap();
    let (ka, kb) = (&a.kernels["silu_mul"], &m.kernels["silu_mul"]);
    assert_eq!(ka.params, kb.params, "interface must be unchanged");
    assert_eq!(ka.imp.steps[0].params.len(), 3);
    assert!(ka.imp.steps[0].cubin.as_deref().unwrap_or("").starts_with("hf:"));
    assert_eq!(kb.imp.steps[0].params.len(), 6);
    assert!(kb.imp.steps[0].cubin.is_none());
    for p in ["decode", "prefill"] {
        assert_eq!(
            serde_json::to_string(&a.programs[p]).unwrap(),
            serde_json::to_string(&m.programs[p]).unwrap(),
            "{p}: dispatches must be identical"
        );
    }
}

const QWEN3_DSPARK: &str = include_str!("../../../examples/qwen3-4b-dspark.json");

#[test]
fn qwen3_dspark_mined_verifies() {
    let m = Manifest::from_json(QWEN3_DSPARK).expect("parse");
    if let Err(errs) = verify(&m) {
        panic!("dspark manifest failed verification:\n{}", errs.join("\n"));
    }
    // draft: embed + l0 norm + 5 layers x 12 (incl. final norm) + lm_head +
    // 7 unrolled markov steps x (embed, bias-accumulate, argmax).
    assert_eq!(m.programs["draft"].dispatches.len(), 2 + 5 * 12 + 1 + 7 * 3);
    // verify: prefill body + final norm + 5 fc taps + 8-row lm_head + argmax.
    assert_eq!(m.programs["verify"].dispatches.len(), 2 + 36 * 12 + 5 + 2);
    // precompute: hidden_norm + fused KV GEMM + 5 x (k_norm, rope, cache).
    assert_eq!(m.programs["draft_precompute"].dispatches.len(), 2 + 5 * 3);
    // spec-ready prefill carries the 5 fc taps; decode stays clean.
    assert_eq!(m.programs["prefill"].dispatches.len(), 2 + 36 * 12 - 1 + 5);
    assert_eq!(m.programs["decode"].dispatches.len(), 2 + 36 * 12 + 2);
    assert_eq!(m.states["draft_kv"].bytes_per_token, 5 * 2 * 1024 * 2);
    // Both 28-param unified instances are ABI-identical; the manifest must
    // pin each to its own cubin or resolution would be ambiguous.
    for k in ["attn_prefill", "attn_draft"] {
        let st = &m.kernels[k].imp.steps[0];
        assert!(st.cubin.is_some() && st.sha256.is_some(), "{k} not pinned");
    }
    assert_ne!(
        m.kernels["attn_prefill"].imp.steps[0].sha256,
        m.kernels["attn_draft"].imp.steps[0].sha256
    );
}
