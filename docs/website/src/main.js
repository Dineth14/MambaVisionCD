import "./styles.css";

const app = document.querySelector("#app");
const fig = (name) => `results/figures/${name}`;
const freqLabels = ["0.01", "0.05", "0.10", "0.20", "0.30", "0.40", "0.50"];

const fmt = {
  params: (n) => `${(n / 1_000_000).toFixed(2)}M`,
  n: (n, d = 2) => Number(n).toFixed(d),
  px: (n) => `${Number(n).toFixed(1)} px`,
  ips: (n) => `${Number(n).toFixed(1)} img/s`,
  pct: (n) => `${(Number(n) * 100).toFixed(1)}%`
};

function okResults(data) {
  return data.results.filter((row) => row.status === "ok");
}

function getStage4R90(row) {
  return row.erf[3].energy_radii.r90;
}

function getTopEsi(row) {
  return row.edge.top_layers[0].edge_selectivity_index;
}

function fastest(rows) {
  return rows.reduce((best, row) =>
    row.throughput.images_per_second > best.throughput.images_per_second ? row : best
  );
}

function largestErf(rows) {
  return rows.reduce((best, row) => (getStage4R90(row) > getStage4R90(best) ? row : best));
}

function modelTone(label) {
  return {
    tiny: "compact reference",
    tiny2: "deeper tiny-width variant",
    small: "wider mid-size model",
    base: "high-capacity baseline",
    large: "largest tested family member"
  }[label] || "measured checkpoint";
}

function caveatBox() {
  return `
    <div class="caveat">
      <b>Measurement discipline:</b> every numeric claim on this page comes from
      <code>analysis/results/json/model_family_analysis.json</code>. Explanations describe what the probes
      support architecturally; they do not assert accuracy, downstream change-detection performance, or training-time causality.
    </div>
  `;
}

function renderHero(rows) {
  const speed = fastest(rows);
  const reach = largestErf(rows);
  const totalParams = rows.reduce((sum, row) => sum + row.architecture.total_parameters, 0);
  return `
    <section class="hero" id="top">
      <div class="hero-grid"></div>
      <div class="hero-content">
        <div class="eyebrow">Measured model-family analysis</div>
        <h1>MambaVision Architecture Lab</h1>
        <p class="lede">
          A deployable research portal for Tiny, Tiny2, Small, Base, and Large MambaVision checkpoints:
          receptive fields, frequency probes, edge sensitivity, SSM stability, branch behavior, and throughput.
        </p>
        <div class="hero-actions">
          <a class="button primary" href="#family">Compare Models</a>
          <a class="button" href="#reasoning">Read The Reasoning</a>
        </div>
        <div class="hero-metrics">
          <div><span>${rows.length}</span><label>models measured</label></div>
          <div><span>${fmt.params(totalParams)}</span><label>total params inspected</label></div>
          <div><span>${speed.label}</span><label>fastest measured throughput</label></div>
          <div><span>${reach.label}</span><label>largest stage-4 ERF r90</label></div>
        </div>
      </div>
    </section>
  `;
}

function renderNav() {
  return `
    <nav class="topbar">
      <a class="brand" href="#top">MambaVision<span>Lab</span></a>
      <div class="navlinks">
        <a href="#family">Family</a>
        <a href="#experiments">Experiments</a>
        <a href="#reasoning">Reasoning</a>
        <a href="#figures">Figures</a>
        <a href="#deploy">Deploy</a>
      </div>
    </nav>
  `;
}

function renderFamilyCards(rows) {
  return `
    <section id="family" class="section">
      <div class="section-head">
        <div class="eyebrow">01 / Model family</div>
        <h2>Capacity, Geometry, And Measured Throughput</h2>
        <p>
          The family keeps the 56→28→14→7 spatial pyramid, while channels and depth vary by checkpoint.
          Parameter count and measured throughput are direct outputs from the run.
        </p>
      </div>
      <div class="model-grid">
        ${rows.map(renderModelCard).join("")}
      </div>
      <div class="note">
        Small measured faster than Tiny in this run. That is a measured throughput fact, not a general law:
        kernel dispatch, GPU occupancy, cached weights, and implementation details can make runtime non-monotonic.
      </div>
    </section>
  `;
}

function renderModelCard(row) {
  const arch = row.architecture;
  const ssm = row.ssm;
  const r90 = getStage4R90(row);
  const topEsi = row.edge.top_layers[0];
  return `
    <article class="model-card" data-model="${row.label}">
      <div class="model-card-top">
        <div>
          <h3>${row.label}</h3>
          <span>${modelTone(row.label)}</span>
        </div>
        <strong>${fmt.params(arch.total_parameters)}</strong>
      </div>
      <div class="chip-row">
        <span>${arch.channels.join(" → ")}</span>
        <span>${arch.ssm_mixer_count} SSM</span>
        <span>${arch.attention_block_count} Attn</span>
      </div>
      <dl class="metric-list">
        <div><dt>Throughput</dt><dd>${fmt.ips(row.throughput.images_per_second)}</dd></div>
        <div><dt>Stage-4 ERF r90</dt><dd>${fmt.px(r90)}</dd></div>
        <div><dt>Top ESI</dt><dd>${fmt.n(topEsi.edge_selectivity_index, 4)}</dd></div>
        <div><dt>SSM stability</dt><dd>${ssm.stable_count}/${ssm.ssm_count}</dd></div>
      </dl>
    </article>
  `;
}

function renderExperimentTables(rows) {
  return `
    <section id="experiments" class="section alt">
      <div class="section-head">
        <div class="eyebrow">02 / Experiments</div>
        <h2>What Was Measured</h2>
        <p>
          The probes are controlled diagnostics. They reveal architecture behavior under synthetic inputs,
          not downstream segmentation or change-detection accuracy.
        </p>
      </div>
      ${renderComparisonTable(rows)}
      <div class="split">
        ${renderBarPanel("Throughput", rows, (r) => r.throughput.images_per_second, "img/s")}
        ${renderBarPanel("Stage-4 ERF r90", rows, getStage4R90, "px")}
      </div>
      <div class="split">
        ${renderBarPanel("Top Edge Selectivity Index", rows, getTopEsi, "ESI")}
        ${renderBarPanel("Mean SSM |eigenvalue|", rows, (r) => r.ssm.mean_eigenvalue_magnitude, "|eig|")}
      </div>
    </section>
  `;
}

function renderComparisonTable(rows) {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Params</th>
            <th>Channels</th>
            <th>Throughput</th>
            <th>ERF r90</th>
            <th>Top ESI</th>
            <th>Stable SSM</th>
            <th>Branch Corr</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row) => `
            <tr>
              <td>${row.label}</td>
              <td>${fmt.params(row.architecture.total_parameters)}</td>
              <td>${row.architecture.channels.join(" / ")}</td>
              <td>${fmt.ips(row.throughput.images_per_second)}</td>
              <td>${fmt.px(getStage4R90(row))}</td>
              <td>${fmt.n(getTopEsi(row), 4)}</td>
              <td>${row.ssm.stable_count}/${row.ssm.ssm_count}</td>
              <td>${fmt.n(row.math_verification.branch_correlation, 4)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderBarPanel(title, rows, valueFn, unit) {
  const max = Math.max(...rows.map(valueFn));
  return `
    <div class="panel">
      <h3>${title}</h3>
      <div class="bars">
        ${rows.map((row) => {
          const value = valueFn(row);
          const width = Math.max(3, (value / max) * 100);
          return `
            <div class="bar-row">
              <span>${row.label}</span>
              <div class="bar-track"><i style="width:${width}%"></i></div>
              <b>${unit === "img/s" ? fmt.ips(value) : `${fmt.n(value, unit === "ESI" ? 4 : 2)} ${unit}`}</b>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function renderReasoning(rows) {
  return `
    <section id="reasoning" class="section">
      <div class="section-head">
        <div class="eyebrow">03 / Reasoned interpretation</div>
        <h2>Why These Results Look The Way They Do</h2>
        <p>
          Each explanation below separates the measurement, the architecture-based interpretation,
          and the limitation. This is deliberate: the experiments are probes, not benchmarks.
        </p>
      </div>
      ${caveatBox()}
      <div class="reason-grid">
        ${renderReasonCard("Architecture scaling", `
          The measured channel widths grow from Tiny/Tiny2 [80,160,320,640] to Large [196,392,784,1568].
          This directly explains the parameter range from ${fmt.params(rows[0].architecture.total_parameters)}
          to ${fmt.params(rows[rows.length - 1].architecture.total_parameters)}. More channels mean more matrix
          and convolution weights, but throughput is not perfectly monotonic because CUDA kernel efficiency,
          memory layout, and batch-size utilization also matter.
        `)}
        ${renderReasonCard("Effective receptive field", `
          Early stages have small r90 values because their features are close to the input grid and dominated by
          local convolutions. Stage 3 and 4 r90 values jump near or above 100 pixels because the model has already
          downsampled to 14×14 and 7×7 maps, then applies windowed token mixing through SSM and attention blocks.
          This supports global-leaning gradient reach, but it does not prove every distant pixel is semantically important.
        `)}
        ${renderReasonCard("Frequency response", `
          All tested frequencies had maximum raw mean activation in stage 4. The careful reading is not "stage 4 is a
          universal frequency detector"; raw activation magnitude is affected by depth, channel scale, normalization,
          and feature distribution. The result says late-stage features carry the largest activation energy under these
          sinusoidal probes, while per-stage normalized curves should be used for true selectivity claims.
        `)}
        ${renderReasonCard("Edge selectivity", `
          The top ESI layer is the stage-1 downsample convolution for every model. That is plausible because strided
          convolution sees intensity discontinuities while reducing resolution, so controlled edges create stronger
          responses than a flat image. ESI only compares synthetic edges against a flat stimulus; it does not mean that
          layer alone is sufficient for real boundary detection.
        `)}
        ${renderReasonCard("SSM stability", `
          Every inspected SSM block had sampled discrete eigenvalues inside the unit circle. This matches the
          Mamba-style parameterization A = -exp(A_log) combined with positive Δ, which pushes exp(ΔA) toward stable
          magnitudes below one. The half-life values are short in this diagonal approximation, so the learned scan
          behaves as stable filtering rather than unbounded accumulation.
        `)}
        ${renderReasonCard("Branch complementarity", `
          The first-mixer SSM/symmetric branch correlations are all near zero. Under the synthetic input used here,
          the branches are not linearly redundant. The safe conclusion is "different measured activation patterns";
          stronger claims about complementarity require validation on real remote-sensing/change-detection datasets.
        `)}
      </div>
      <div class="model-details">
        ${rows.map(renderModelReasoning).join("")}
      </div>
    </section>
  `;
}

function renderReasonCard(title, text) {
  return `
    <article class="reason-card">
      <h3>${title}</h3>
      <p>${text}</p>
    </article>
  `;
}

function renderModelReasoning(row) {
  const arch = row.architecture;
  const erf = row.erf.map((item) => item.energy_radii.r90);
  const top = row.edge.top_layers[0];
  const mv = row.math_verification;
  return `
    <details class="model-detail">
      <summary>${row.label}: measured explanation</summary>
      <div class="detail-grid">
        <div>
          <h4>Measured geometry</h4>
          <p>Channels ${arch.channels.join(" → ")} at ${arch.resolutions.map((r) => r.join("×")).join(" → ")}. Parameter count ${arch.total_parameters.toLocaleString()}.</p>
        </div>
        <div>
          <h4>ERF behavior</h4>
          <p>r90 by stage: ${erf.map((v) => fmt.px(v)).join(", ")}. Late-stage reach is larger because features are lower-resolution and token-mixed.</p>
        </div>
        <div>
          <h4>Selective scan</h4>
          <p>Δ diff ${fmt.n(mv.delta_diff, 4)}, B diff ${fmt.n(mv.B_diff, 4)}, C diff ${fmt.n(mv.C_diff, 4)}. These nonzero differences verify input-dependent scan parameters for the inspected mixer.</p>
        </div>
        <div>
          <h4>Edge response</h4>
          <p>Top ESI layer ${top.layer}, ESI ${fmt.n(top.edge_selectivity_index, 4)}. This is a controlled stimulus result, not a natural-image edge benchmark.</p>
        </div>
      </div>
    </details>
  `;
}

function renderFigures() {
  const cards = [
    ["Family ERF r90", "family_erf_r90.png", "Compares stage-wise effective receptive field growth across model sizes."],
    ["Throughput", "family_throughput.png", "Measured local throughput with batch-size fallback if memory constrained."],
    ["Top ESI", "family_edge_top_esi.png", "Highest edge selectivity index per model."],
    ["Frequency Dominance", "family_frequency_dominance.png", "Dominant raw-activation stage for each sinusoidal frequency."],
    ["SSM Summary", "family_ssm_summary.png", "Mean eigenvalue magnitude and impulse half-life by model."],
    ["Stage Diagram", "mambavision_stage_diagram.png", "Tiny checkpoint architecture map generated by script 01."],
    ["ERF Heatmaps", "erf_per_stage.png", "Single-model detailed ERF heatmaps for the Tiny checkpoint."],
    ["Layer Summary", "layer_function_summary.png", "Representative layer function panels from script 05."]
  ];
  return `
    <section id="figures" class="section alt">
      <div class="section-head">
        <div class="eyebrow">04 / Generated figures</div>
        <h2>Visual Evidence</h2>
        <p>The deployed app ships with generated PNG outputs copied into <code>docs/website/public/results/figures</code>.</p>
      </div>
      <div class="figure-grid">
        ${cards.map(([title, image, caption]) => `
          <figure class="figure-card">
            <img src="${fig(image)}" alt="${title}" loading="lazy" />
            <figcaption><b>${title}</b><span>${caption}</span></figcaption>
          </figure>
        `).join("")}
      </div>
    </section>
  `;
}

function renderDeploy() {
  return `
    <section id="deploy" class="section">
      <div class="section-head">
        <div class="eyebrow">05 / Deployment</div>
        <h2>Node/Vite Project Ready For GitHub Pages</h2>
        <p>
          The website is now a Node project under <code>docs/website</code>. It can be developed locally,
          built into static assets, and deployed through the included GitHub Pages workflow.
        </p>
      </div>
      <div class="deploy-grid">
        <div class="code-card">
          <h3>Local development</h3>
          <pre><code>cd docs/website
npm install
npm run dev</code></pre>
        </div>
        <div class="code-card">
          <h3>Static build</h3>
          <pre><code>cd docs/website
npm run build
npm run preview</code></pre>
        </div>
        <div class="code-card">
          <h3>GitHub Pages</h3>
          <pre><code>.github/workflows/deploy-website.yml
publishes docs/website/dist</code></pre>
        </div>
      </div>
      <div class="note">
        To refresh the deployed evidence, rerun <code>analysis/scripts/08_model_family_analysis.py</code>,
        copy updated JSON/figures into <code>docs/website/public</code>, then rebuild.
      </div>
    </section>
  `;
}

function renderMathPrimer() {
  return `
    <section class="section compact">
      <div class="section-head">
        <div class="eyebrow">Mathematical anchor</div>
        <h2>Why Stability Was Expected But Still Verified</h2>
      </div>
      <div class="math-panel">
        <p class="math">$$A=-\\exp(A_{\\log}),\\qquad \\bar A=\\exp(\\Delta A)$$</p>
        <p>
          Since the inspected implementation uses negative continuous-time diagonal dynamics and positive learned
          time steps, the sampled discrete eigenvalue magnitudes should be below one. The experiment verifies this
          numerically for each model rather than assuming it from the equation alone.
        </p>
      </div>
    </section>
  `;
}

function renderApp(data) {
  const rows = okResults(data);
  app.innerHTML = `
    ${renderNav()}
    ${renderHero(rows)}
    ${renderFamilyCards(rows)}
    ${renderExperimentTables(rows)}
    ${renderReasoning(rows)}
    ${renderMathPrimer()}
    ${renderFigures()}
    ${renderDeploy()}
    <footer>
      <p>Original model family: NVIDIA MambaVision. Local analysis JSON: <code>public/data/model_family_analysis.json</code>.</p>
      <pre>@article{hatamizadeh2024mambavision,
  title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  journal={arXiv preprint arXiv:2407.08083},
  year={2024}
}</pre>
    </footer>
  `;
  bindNav();
  renderKatex();
}

function bindNav() {
  const links = [...document.querySelectorAll(".navlinks a")];
  const sections = [...document.querySelectorAll("section[id]")];
  window.addEventListener("scroll", () => {
    const current = sections.filter((section) => section.getBoundingClientRect().top < 120).pop();
    links.forEach((link) => link.classList.toggle("active", current && link.getAttribute("href") === `#${current.id}`));
  }, { passive: true });
}

function renderKatex() {
  const attempt = () => {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\(", right: "\\)", display: false }
        ]
      });
    } else {
      window.setTimeout(attempt, 100);
    }
  };
  attempt();
}

async function boot() {
  try {
    const response = await fetch("data/model_family_analysis.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`Failed to load data: ${response.status}`);
    const data = await response.json();
    renderApp(data);
  } catch (error) {
    app.innerHTML = `
      <main class="error-state">
        <h1>Could not load model-family data</h1>
        <p>${error.message}</p>
        <p>Run <code>analysis/scripts/08_model_family_analysis.py</code> and copy the JSON into <code>docs/website/public/data</code>.</p>
      </main>
    `;
  }
}

boot();
