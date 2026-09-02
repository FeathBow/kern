//! Parse and verify manifests: `cargo run -p kern-manifest --example verify -- a.json b.json`.
fn main() {
    let mut bad = 0;
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).expect("read manifest");
        match kern_manifest::Manifest::from_json(&text).map_err(|e| e.to_string()) {
            Ok(m) => match kern_manifest::verify(&m) {
                Ok(()) => println!("{path}: ok"),
                Err(e) => {
                    bad += 1;
                    println!("{path}: {e}");
                }
            },
            Err(e) => {
                bad += 1;
                println!("{path}: {e}");
            }
        }
    }
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
