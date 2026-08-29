//! The generated Qwen3-4B example manifest must parse and verify clean.
//! Regenerate with `python3 tools/gen_qwen3.py` after schema changes.

use kern_manifest::{verify, Manifest};

const QWEN3: &str = include_str!("../../../examples/qwen3-4b.json");

#[test]
fn qwen3_example_verifies() {
    let m = Manifest::from_json(QWEN3).expect("parse");
    if let Err(errs) = verify(&m) {
        panic!("qwen3 example failed verification:\n{}", errs.join("\n"));
    }

    assert_eq!(m.programs.len(), 2);
    // embed + 36 layers x 14 dispatches + final norm + lm_head
    let expected = 1 + 36 * 14 + 2;
    for (name, p) in &m.programs {
        assert_eq!(p.dispatches.len(), expected, "program {name}");
    }
    // The runtime's entire knowledge of the KV cache: a byte count.
    assert_eq!(m.states["kv"].bytes_per_token, 36 * 2 * 1024 * 2);
}

#[test]
fn qwen3_example_roundtrips() {
    let m = Manifest::from_json(QWEN3).expect("parse");
    let again = Manifest::from_json(&m.to_json()).expect("reparse");
    assert_eq!(m.to_json(), again.to_json());
}
