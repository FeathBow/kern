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
}
