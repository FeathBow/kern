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
    // reduce launch and argmax's two stages live inside their op impls.
    assert_eq!(m.programs["decode"].len(), 2 + 36 * 12 + 2);
    // prefill: same forward minus the final_norm/lm_head/sample tail — the
    // last prompt token goes through `decode` instead.
    assert_eq!(m.programs["prefill"].len(), 2 + 36 * 12 - 1);
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

/// The A/B fixture for `kern test`: identical to qwen3-4b.json except
/// `silu_mul`'s *impl* is the mined vLLM cubin (6-param launch ABI) instead
/// of the HF hub package (3-param ABI). The interface — `(out, in)`, every
/// ABI constant folded into the impl — and every call are untouched: a pure
/// impl swap.
const QWEN3_SILU_MINED: &str = include_str!("../../../examples/qwen3-4b-silu-mined.json");

#[test]
fn qwen3_silu_mined_fixture_verifies() {
    let m = Manifest::from_json(QWEN3_SILU_MINED).expect("parse");
    if let Err(errs) = verify(&m) {
        panic!("silu-mined fixture failed verification:\n{}", errs.join("\n"));
    }
    let a = Manifest::from_json(QWEN3).unwrap();
    let (oa, ob) = (&a.ops["silu_mul"], &m.ops["silu_mul"]);
    assert_eq!(oa.params, ob.params, "interface must be unchanged");
    assert_eq!(oa.params.len(), 2);
    let (la, lb) = (&oa.imp.launches[0], &ob.imp.launches[0]);
    assert_eq!(la.params_of(oa).len(), 3);
    assert!(a.modules[la.module.as_deref().unwrap()].source.starts_with("hf:"));
    assert_eq!(lb.params_of(ob).len(), 6);
    assert!(lb.module.is_none());
    for p in ["decode", "prefill"] {
        assert_eq!(
            serde_json::to_string(&a.programs[p]).unwrap(),
            serde_json::to_string(&m.programs[p]).unwrap(),
            "{p}: calls must be identical"
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
    assert_eq!(m.programs["draft"].len(), 2 + 5 * 12 + 1 + 7 * 3);
    // verify: prefill body + final norm + 5 fc taps + 8-row lm_head + argmax.
    assert_eq!(m.programs["verify"].len(), 2 + 36 * 12 + 5 + 2);
    // precompute: hidden_norm + fused KV GEMM + 5 x (k_norm, rope, cache).
    assert_eq!(m.programs["draft_precompute"].len(), 2 + 5 * 3);
    // spec-ready prefill carries the 5 fc taps; decode stays clean.
    assert_eq!(m.programs["prefill"].len(), 2 + 36 * 12 - 1 + 5);
    assert_eq!(m.programs["decode"].len(), 2 + 36 * 12 + 2);
    assert_eq!(m.states["draft_kv"].bytes_per_token, 5 * 2 * 1024 * 2);
    // Both unified instances are ABI-identical; the manifest must pin each
    // to its own module or resolution would be ambiguous.
    let pinned = |op: &str| {
        let l = &m.ops[op].imp.launches[0];
        m.modules[l.module.as_deref().unwrap_or_else(|| panic!("{op} not pinned"))].sha256.clone()
    };
    assert_ne!(pinned("attn_prefill"), pinned("attn_draft"));
}
