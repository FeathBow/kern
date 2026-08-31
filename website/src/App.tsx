import { useState } from "react";

type AttentionMode = "prefill" | "decode";
type VerifyMode = "type" | "dataflow" | "abi";

const verifyExamples: Record<
  VerifyMode,
  { phase: string; source: string; target: string; error: string }
> = {
  type: {
    phase: "MANIFEST / TYPE",
    source: "q · buffer<bf16>",
    target: "arg 2 · buffer<fp8e4m3>",
    error: "buffer `q` has dtype bf16\nbut param expects fp8e4m3",
  },
  dataflow: {
    phase: "MANIFEST / DATAFLOW",
    source: "segm_out · unreadied",
    target: "reduce · in buffer<f32>",
    error: "scratch `segm_out` is read\nbefore any step wrote it",
  },
  abi: {
    phase: "LOAD / CUBIN ABI",
    source: "declared · [8, 8, 4]",
    target: "loaded · [8, 16, 4]",
    error: "no loaded instance matches\ndeclared param layout",
  },
};

function Arrow({ className = "" }: { className?: string }) {
  return (
    <svg className={`arrow ${className}`} viewBox="0 0 120 34" aria-hidden="true">
      <path d="M3 19 C34 12, 72 24, 109 15" />
      <path d="M99 7 L110 15 L100 26" />
    </svg>
  );
}

function Header() {
  return (
    <header className="site-header">
      <a className="wordmark" href="#top" aria-label="Kern home">
        KERN<span className="wordmark-dot">■</span>
      </a>
      <nav aria-label="Primary navigation">
        <a href="#artifact">ARTIFACT</a>
        <a href="#proof">PROOF</a>
        <a
          className="github-link"
          href="https://github.com/pegainfer-project/kern"
          target="_blank"
          rel="noreferrer"
        >
          GITHUB ↗
        </a>
      </nav>
    </header>
  );
}

function HeroDiagram() {
  return (
    <div className="hero-diagram" aria-label="Kern artifact flows through verification into a thin runtime">
      <div className="artifact-stack">
        <div className="file-sheet file-sheet-back">WEIGHTS</div>
        <div className="file-sheet file-sheet-mid">KERNELS</div>
        <div className="file-sheet file-sheet-front">
          <span>MANIFEST</span>
          <span className="file-code">program decode</span>
          <span className="file-code">kernel attn</span>
          <span className="file-code">state kv</span>
        </div>
      </div>
      <Arrow className="hero-arrow-one" />
      <div className="verify-stamp">
        <span className="verify-check">✓</span>
        <span>VERIFIED</span>
      </div>
      <Arrow className="hero-arrow-two" />
      <div className="runtime-chip">
        <span>THIN</span>
        <strong>RUNTIME</strong>
        <div className="chip-pins" aria-hidden="true" />
      </div>
    </div>
  );
}

function Hero() {
  return (
    <section className="hero" id="top">
      <Header />
      <div className="hero-copy">
        <p className="eyebrow">MODEL-AGNOSTIC GPU EXECUTION</p>
        <h1>
          MODELS SHIP AS
          <span>VERIFIED GPU</span>
          PROGRAMS.
        </h1>
        <div className="hero-bottomline">
          <p>Manifest. Kernels. Weights.</p>
          <a href="#artifact">SEE THE PROGRAM ↓</a>
        </div>
      </div>
      <HeroDiagram />
      <div className="hero-scale">
        <strong>&lt;3K</strong>
        <span>RUST SOURCE LINES<br />END TO END</span>
        <small>447 run · 655 runtime · 1,537 schema + verifier</small>
      </div>
      <div className="hero-proof">
        <strong>92%</strong>
        <span>of vLLM decode throughput</span>
        <small>Qwen3-4B · batch 1 · single GB300 · 377 vs 409 tok/s</small>
      </div>
    </section>
  );
}

function Artifact() {
  return (
    <section className="section artifact-section" id="artifact">
      <div className="section-number">01 / ARTIFACT</div>
      <div className="artifact-title">
        <h2>A MODEL IS<br />A PROGRAM.</h2>
        <p>Everything the runtime needs.<br />Nothing about the model architecture.</p>
      </div>
      <div className="program-blueprint">
        <div className="blueprint-node source-node">
          <span className="node-kicker">PROVIDER</span>
          <strong>manifest.json</strong>
          <strong>kernels/*.cubin</strong>
          <strong>weights</strong>
        </div>
        <Arrow />
        <div className="manifest-window">
          <div className="manifest-rail">
            <span>V2</span>
            <span>QWEN3-4B</span>
          </div>
          <div className="manifest-body">
            <div><b>1</b><span>symbol</span></div>
            <div><b>1</b><span>opaque state</span></div>
            <div><b>310</b><span>buffers</span></div>
            <div><b>12</b><span>kernel interfaces</span></div>
            <div className="manifest-programs"><b>2</b><span>programs</span></div>
          </div>
        </div>
        <Arrow />
        <div className="blueprint-node executor-node">
          <span className="node-kicker">RUNTIME KNOWS</span>
          <strong>buffers</strong>
          <strong>state bytes</strong>
          <strong>dispatches</strong>
          <span className="executor-no">NO MODEL BRANCHES</span>
        </div>
      </div>
    </section>
  );
}

function ReplaceableImplementation() {
  const [mode, setMode] = useState<AttentionMode>("decode");
  const decode = mode === "decode";

  return (
    <section className="section implementation-section">
      <div className="section-number">02 / REPLACEABLE IMPLEMENTATION</div>
      <div className="implementation-layout">
        <div className="implementation-copy">
          <h2>ONE<br />INTERFACE.</h2>
          <p className="large-note">The call site stays still.</p>
          <div className="mode-switch" role="group" aria-label="Attention implementation">
            <button
              className={!decode ? "active" : ""}
              onClick={() => setMode("prefill")}
              aria-pressed={!decode}
            >
              PREFILL
            </button>
            <button
              className={decode ? "active" : ""}
              onClick={() => setMode("decode")}
              aria-pressed={decode}
            >
              DECODE
            </button>
          </div>
        </div>

        <div className="implementation-visual">
          <div className="interface-block">
            <span>KERNEL INTERFACE</span>
            <strong>attn</strong>
            <code>28 typed parameters</code>
            <div className="type-row">
              <i>IN</i> buffer&lt;bf16&gt;
              <i>INOUT</i> ptr
              <i>OUT</i> buffer&lt;bf16&gt;
            </div>
          </div>
          <Arrow className="vertical-arrow" />
          <div className={`micro-program ${decode ? "decode" : "prefill"}`}>
            <div className="micro-header">
              <span>{decode ? "DECODE IMPL" : "PREFILL IMPL"}</span>
              <b>{decode ? "2 LAUNCHES" : "1 LAUNCH"}</b>
            </div>
            <div className="launch-flow">
              <div className="launch-box">
                <span>01</span>
                <strong>unified_attention</strong>
              </div>
              {decode && (
                <>
                  <div className="scratch-bus">
                    <span>PRIVATE SCRATCH × 3</span>
                    <svg viewBox="0 0 150 48" aria-hidden="true">
                      <path d="M2 27 C35 2, 97 46, 147 17" />
                    </svg>
                  </div>
                  <div className="launch-box accent-launch">
                    <span>02</span>
                    <strong>reduce_segments</strong>
                  </div>
                </>
              )}
            </div>
          </div>
          <p className="visual-caption">
            {decode
              ? "Multi-launch details remain private to the implementation."
              : "The same interface resolves to a single 2D attention launch."}
          </p>
        </div>
      </div>
    </section>
  );
}

function Verifier() {
  const [mode, setMode] = useState<VerifyMode>("type");
  const example = verifyExamples[mode];

  return (
    <section className="section verifier-section">
      <div className="section-number">03 / CRASH EARLY</div>
      <div className="verifier-header">
        <h2>BAD DECLARATIONS<br /><span>STOP HERE.</span></h2>
        <p>Before the first GPU launch.</p>
      </div>
      <div className="verifier-console">
        <div className="verify-tabs" role="group" aria-label="Verifier failure example">
          {(["type", "dataflow", "abi"] as VerifyMode[]).map((item) => (
            <button
              key={item}
              className={mode === item ? "active" : ""}
              onClick={() => setMode(item)}
              aria-pressed={mode === item}
            >
              {item.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="broken-wire">
          <div className="wire-end">
            <span>SOURCE</span>
            <strong>{example.source}</strong>
          </div>
          <svg viewBox="0 0 330 80" aria-hidden="true">
            <path className="wire-left" d="M3 39 C75 20, 112 62, 148 39" />
            <path className="wire-right" d="M181 39 C224 13, 274 58, 326 33" />
            <path className="wire-break" d="M151 24 L178 57 M178 23 L151 58" />
          </svg>
          <div className="wire-end wire-target">
            <span>TARGET</span>
            <strong>{example.target}</strong>
          </div>
        </div>
        <div className="diagnostic">
          <span>{example.phase}</span>
          <pre>{example.error}</pre>
          <b>EXECUTION REFUSED</b>
        </div>
      </div>
      <div className="trust-boundary">
        <span>PROVES</span> declaration consistency
        <i />
        <span>TRUSTS</span> kernel behavior
      </div>
    </section>
  );
}

function ProgramGraph({ speculative }: { speculative: boolean }) {
  return (
    <div className={`program-graph ${speculative ? "speculative" : "standard"}`}>
      <svg className="graph-lines" viewBox="0 0 760 360" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <marker
            id="graph-arrow"
            viewBox="0 0 12 12"
            refX="10"
            refY="6"
            markerWidth="12"
            markerHeight="12"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
          >
            <path className="graph-arrow-head" d="M1 1 L10 6 L1 11" />
          </marker>
        </defs>
        {speculative ? (
          <>
            <path markerEnd="url(#graph-arrow)" d="M118 75 C188 73, 229 76, 275 75" />
            <path markerEnd="url(#graph-arrow)" d="M382 75 C485 73, 553 64, 650 65" />
            <path markerEnd="url(#graph-arrow)" d="M675 96 C675 159, 568 153, 496 214" />
            <path markerEnd="url(#graph-arrow)" d="M395 237 C352 260, 327 294, 267 306" />
            <path markerEnd="url(#graph-arrow)" d="M220 304 C218 255, 124 252, 116 214" />
            <path markerEnd="url(#graph-arrow)" d="M153 178 C211 177, 245 129, 278 100" />
          </>
        ) : (
          <path markerEnd="url(#graph-arrow)" d="M205 180 C304 135, 453 225, 550 180" />
        )}
      </svg>
      {speculative ? (
        <>
          <div className="graph-node n-prefill">prefill</div>
          <div className="graph-node n-decode-spec">decode_spec</div>
          <div className="graph-node n-draft">draft</div>
          <div className="graph-node n-verify">verify</div>
          <div className="graph-node n-precompute">draft_precompute</div>
          <div className="graph-node n-decode">decode</div>
        </>
      ) : (
        <>
          <div className="graph-node n-standard-prefill">prefill</div>
          <div className="graph-node n-standard-decode">decode</div>
        </>
      )}
    </div>
  );
}

function SchemaDiff() {
  return (
    <div className="schema-diff" aria-label="Manifest v2 schema diff from standard decoding to DSpark">
      <div className="schema-contract">
        <span>UNCHANGED CONTRACT</span>
        <strong>manifest.v2</strong>
        <code>meta · symbols · states · buffers · kernels · programs</code>
      </div>
      <div className="diff-window">
        <div className="diff-header">
          <span>STANDARD</span>
          <svg viewBox="0 0 80 20" aria-hidden="true">
            <path d="M2 10 C25 7, 49 13, 73 10 M67 4 L75 10 L67 16" />
          </svg>
          <span>+ DSPARK</span>
        </div>
        <div className="diff-line context"><b> </b><code>"meta": {'{'} "version": 2 {'}'}</code></div>
        <div className="diff-line context"><b> </b><code>"states": {'{'} "kv": …</code></div>
        <div className="diff-line added"><b>+</b><code>"draft_kv": …</code></div>
        <div className="diff-line context"><b> </b><code>"buffers": {'{'} …</code></div>
        <div className="diff-line added"><b>+</b><code>"fc_out": {'{'} "class": "carry" {'}'}</code></div>
        <div className="diff-line reused"><b>=</b><code>"kernels": same cubin primitives</code></div>
        <div className="diff-line context"><b> </b><code>"programs": {'{'} "prefill": …, "decode": …</code></div>
        <div className="diff-line added"><b>+</b><code>"decode_spec" · "draft" · "verify" · "draft_precompute"</code></div>
      </div>
      <div className="schema-payoff">
        <span>SAME PARSER</span><i>→</i><span>SAME VERIFIER</span><i>→</i><span>SAME RUNTIME</span>
        <strong>0 NEW KERNEL CODE</strong>
      </div>
    </div>
  );
}

function Composition() {
  const [speculative, setSpeculative] = useState(true);

  return (
    <section className="section composition-section" id="composition">
      <div className="section-number">04 / PROGRAM COMPOSITION</div>
      <div className="composition-title">
        <div>
          <span className="giant-count">{speculative ? "06" : "02"}</span>
          <p>PROGRAMS</p>
        </div>
        <h2>NEW BEHAVIOR.<br />SAME SCHEMA.</h2>
      </div>
      <div className="composition-switch" role="group" aria-label="Program composition example">
        <button
          className={!speculative ? "active" : ""}
          onClick={() => setSpeculative(false)}
          aria-pressed={!speculative}
        >
          STANDARD
        </button>
        <button
          className={speculative ? "active" : ""}
          onClick={() => setSpeculative(true)}
          aria-pressed={speculative}
        >
          + DSPARK
        </button>
      </div>
      <ProgramGraph speculative={speculative} />
      <SchemaDiff />
      <div className="composition-facts">
        <div><strong>V2</strong><span>schema</span></div>
        <div><strong>{speculative ? "02" : "01"}</strong><span>states</span></div>
        <div><strong>{speculative ? "01" : "00"}</strong><span>carry buffers</span></div>
        <div className="green-fact"><strong>0</strong><span>new handwritten kernels</span></div>
      </div>
    </section>
  );
}

function Proof() {
  return (
    <section className="section proof-section" id="proof">
      <div className="section-number">05 / MEASURED</div>
      <h2>REAL MODEL.<br />REAL KERNELS.</h2>
      <div className="benchmark benchmark-decode">
        <div className="benchmark-label">
          <span>DECODE</span>
          <small>tok/s · higher is better</small>
        </div>
        <div className="bar-row">
          <span>KERN</span>
          <div className="bar-track"><i style={{ width: "92%" }} /></div>
          <strong>377</strong>
        </div>
        <div className="bar-row baseline">
          <span>vLLM</span>
          <div className="bar-track"><i style={{ width: "100%" }} /></div>
          <strong>409</strong>
        </div>
      </div>
      <div className="proof-pair">
        <div className="proof-stat prefill-stat">
          <span>CHUNKED PREFILL</span>
          <strong>37×</strong>
          <p><b>~60 ms</b> vs 2.18 s</p>
          <small>709-token prompt · chunked vs repeated decode path</small>
        </div>
        <div className="proof-stat spec-stat">
          <span>SPECULATIVE DECODE</span>
          <strong>2.4×</strong>
          <p><b>948</b> vs 388 tok/s</p>
          <small>32-token prompt · byte-equal greedy output in this run</small>
        </div>
      </div>
      <p className="benchmark-footnote">
        Repository measurements · Qwen3-4B · batch 1 · single GB300. Each comparison uses its stated control.
      </p>
    </section>
  );
}

function Lifecycle() {
  return (
    <section className="section lifecycle-section">
      <div className="section-number">06 / LIFECYCLE</div>
      <div className="lifecycle-heading">
        <h2>AUTHOR.<br />PACKAGE.<br /><span>RUN.</span></h2>
        <p>Keep model intelligence out of the serving runtime.</p>
      </div>
      <div className="lifecycle-flow">
        <div className="lifecycle-stage author-stage">
          <span>01</span>
          <strong>AUTHOR + OPTIMIZE</strong>
          <p>Agent-held compiler<br />reference + fast implementation</p>
          <em>TileFoundry-inspired authoring layer</em>
        </div>
        <Arrow />
        <div className="lifecycle-stage package-stage">
          <span>02</span>
          <strong>PACKAGE</strong>
          <p>manifest<br />kernels<br />weights</p>
          <em>kern artifact</em>
        </div>
        <Arrow />
        <div className="lifecycle-stage run-stage">
          <span>03</span>
          <strong>VERIFY + RUN</strong>
          <p>load-time checks<br />CUDA graph replay</p>
          <em>kern runtime</em>
        </div>
      </div>
      <p className="integration-note">Conceptual alignment · no direct TileFoundry → kern exporter today.</p>
    </section>
  );
}

function Footer() {
  return (
    <footer>
      <div>
        <span className="footer-mark">KERN■</span>
        <h2>THE RUNTIME<br />DOESN'T NEED<br />THE MODEL.</h2>
      </div>
      <div className="footer-links">
        <a href="https://github.com/pegainfer-project/kern" target="_blank" rel="noreferrer">SOURCE ↗</a>
        <a href="#top">BACK TO TOP ↑</a>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <main>
      <Hero />
      <Artifact />
      <ReplaceableImplementation />
      <Verifier />
      <Composition />
      <Proof />
      <Lifecycle />
      <Footer />
    </main>
  );
}
