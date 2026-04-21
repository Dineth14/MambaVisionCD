(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const a of document.querySelectorAll('link[rel="modulepreload"]'))n(a);new MutationObserver(a=>{for(const s of a)if(s.type==="childList")for(const o of s.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&n(o)}).observe(document,{childList:!0,subtree:!0});function i(a){const s={};return a.integrity&&(s.integrity=a.integrity),a.referrerPolicy&&(s.referrerPolicy=a.referrerPolicy),a.crossOrigin==="use-credentials"?s.credentials="include":a.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function n(a){if(a.ep)return;a.ep=!0;const s=i(a);fetch(a.href,s)}})();const p=document.querySelector("#app"),m=e=>`results/figures/${e}`,r={params:e=>`${(e/1e6).toFixed(2)}M`,n:(e,t=2)=>Number(e).toFixed(t),px:e=>`${Number(e).toFixed(1)} px`,ips:e=>`${Number(e).toFixed(1)} img/s`,pct:e=>`${(Number(e)*100).toFixed(1)}%`};function f(e){return e.results.filter(t=>t.status==="ok")}function c(e){return e.erf[3].energy_radii.r90}function u(e){return e.edge.top_layers[0].edge_selectivity_index}function g(e){return e.reduce((t,i)=>i.throughput.images_per_second>t.throughput.images_per_second?i:t)}function v(e){return e.reduce((t,i)=>c(i)>c(t)?i:t)}function y(e){return{tiny:"compact reference",tiny2:"deeper tiny-width variant",small:"wider mid-size model",base:"high-capacity baseline",large:"largest tested family member"}[e]||"measured checkpoint"}function b(){return`
    <div class="caveat">
      <b>Measurement discipline:</b> every numeric claim on this page comes from
      <code>analysis/results/json/model_family_analysis.json</code>. Explanations describe what the probes
      support architecturally; they do not assert accuracy, downstream change-detection performance, or training-time causality.
    </div>
  `}function $(e){const t=g(e),i=v(e),n=e.reduce((a,s)=>a+s.architecture.total_parameters,0);return`
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
          <div><span>${e.length}</span><label>models measured</label></div>
          <div><span>${r.params(n)}</span><label>total params inspected</label></div>
          <div><span>${t.label}</span><label>fastest measured throughput</label></div>
          <div><span>${i.label}</span><label>largest stage-4 ERF r90</label></div>
        </div>
      </div>
    </section>
  `}function _(){return`
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
  `}function S(e){return`
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
        ${e.map(T).join("")}
      </div>
      <div class="note">
        Small measured faster than Tiny in this run. That is a measured throughput fact, not a general law:
        kernel dispatch, GPU occupancy, cached weights, and implementation details can make runtime non-monotonic.
      </div>
    </section>
  `}function T(e){const t=e.architecture,i=e.ssm,n=c(e),a=e.edge.top_layers[0];return`
    <article class="model-card" data-model="${e.label}">
      <div class="model-card-top">
        <div>
          <h3>${e.label}</h3>
          <span>${y(e.label)}</span>
        </div>
        <strong>${r.params(t.total_parameters)}</strong>
      </div>
      <div class="chip-row">
        <span>${t.channels.join(" → ")}</span>
        <span>${t.ssm_mixer_count} SSM</span>
        <span>${t.attention_block_count} Attn</span>
      </div>
      <dl class="metric-list">
        <div><dt>Throughput</dt><dd>${r.ips(e.throughput.images_per_second)}</dd></div>
        <div><dt>Stage-4 ERF r90</dt><dd>${r.px(n)}</dd></div>
        <div><dt>Top ESI</dt><dd>${r.n(a.edge_selectivity_index,4)}</dd></div>
        <div><dt>SSM stability</dt><dd>${i.stable_count}/${i.ssm_count}</dd></div>
      </dl>
    </article>
  `}function x(e){return`
    <section id="experiments" class="section alt">
      <div class="section-head">
        <div class="eyebrow">02 / Experiments</div>
        <h2>What Was Measured</h2>
        <p>
          The probes are controlled diagnostics. They reveal architecture behavior under synthetic inputs,
          not downstream segmentation or change-detection accuracy.
        </p>
      </div>
      ${M(e)}
      <div class="split">
        ${l("Throughput",e,t=>t.throughput.images_per_second,"img/s")}
        ${l("Stage-4 ERF r90",e,c,"px")}
      </div>
      <div class="split">
        ${l("Top Edge Selectivity Index",e,u,"ESI")}
        ${l("Mean SSM |eigenvalue|",e,t=>t.ssm.mean_eigenvalue_magnitude,"|eig|")}
      </div>
    </section>
  `}function M(e){return`
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
          ${e.map(t=>`
            <tr>
              <td>${t.label}</td>
              <td>${r.params(t.architecture.total_parameters)}</td>
              <td>${t.architecture.channels.join(" / ")}</td>
              <td>${r.ips(t.throughput.images_per_second)}</td>
              <td>${r.px(c(t))}</td>
              <td>${r.n(u(t),4)}</td>
              <td>${t.ssm.stable_count}/${t.ssm.ssm_count}</td>
              <td>${r.n(t.math_verification.branch_correlation,4)}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `}function l(e,t,i,n){const a=Math.max(...t.map(i));return`
    <div class="panel">
      <h3>${e}</h3>
      <div class="bars">
        ${t.map(s=>{const o=i(s),h=Math.max(3,o/a*100);return`
            <div class="bar-row">
              <span>${s.label}</span>
              <div class="bar-track"><i style="width:${h}%"></i></div>
              <b>${n==="img/s"?r.ips(o):`${r.n(o,n==="ESI"?4:2)} ${n}`}</b>
            </div>
          `}).join("")}
      </div>
    </div>
  `}function w(e){return`
    <section id="reasoning" class="section">
      <div class="section-head">
        <div class="eyebrow">03 / Reasoned interpretation</div>
        <h2>Why These Results Look The Way They Do</h2>
        <p>
          Each explanation below separates the measurement, the architecture-based interpretation,
          and the limitation. This is deliberate: the experiments are probes, not benchmarks.
        </p>
      </div>
      ${b()}
      <div class="reason-grid">
        ${d("Architecture scaling",`
          The measured channel widths grow from Tiny/Tiny2 [80,160,320,640] to Large [196,392,784,1568].
          This directly explains the parameter range from ${r.params(e[0].architecture.total_parameters)}
          to ${r.params(e[e.length-1].architecture.total_parameters)}. More channels mean more matrix
          and convolution weights, but throughput is not perfectly monotonic because CUDA kernel efficiency,
          memory layout, and batch-size utilization also matter.
        `)}
        ${d("Effective receptive field",`
          Early stages have small r90 values because their features are close to the input grid and dominated by
          local convolutions. Stage 3 and 4 r90 values jump near or above 100 pixels because the model has already
          downsampled to 14×14 and 7×7 maps, then applies windowed token mixing through SSM and attention blocks.
          This supports global-leaning gradient reach, but it does not prove every distant pixel is semantically important.
        `)}
        ${d("Frequency response",`
          All tested frequencies had maximum raw mean activation in stage 4. The careful reading is not "stage 4 is a
          universal frequency detector"; raw activation magnitude is affected by depth, channel scale, normalization,
          and feature distribution. The result says late-stage features carry the largest activation energy under these
          sinusoidal probes, while per-stage normalized curves should be used for true selectivity claims.
        `)}
        ${d("Edge selectivity",`
          The top ESI layer is the stage-1 downsample convolution for every model. That is plausible because strided
          convolution sees intensity discontinuities while reducing resolution, so controlled edges create stronger
          responses than a flat image. ESI only compares synthetic edges against a flat stimulus; it does not mean that
          layer alone is sufficient for real boundary detection.
        `)}
        ${d("SSM stability",`
          Every inspected SSM block had sampled discrete eigenvalues inside the unit circle. This matches the
          Mamba-style parameterization A = -exp(A_log) combined with positive Δ, which pushes exp(ΔA) toward stable
          magnitudes below one. The half-life values are short in this diagonal approximation, so the learned scan
          behaves as stable filtering rather than unbounded accumulation.
        `)}
        ${d("Branch complementarity",`
          The first-mixer SSM/symmetric branch correlations are all near zero. Under the synthetic input used here,
          the branches are not linearly redundant. The safe conclusion is "different measured activation patterns";
          stronger claims about complementarity require validation on real remote-sensing/change-detection datasets.
        `)}
      </div>
      <div class="model-details">
        ${e.map(E).join("")}
      </div>
    </section>
  `}function d(e,t){return`
    <article class="reason-card">
      <h3>${e}</h3>
      <p>${t}</p>
    </article>
  `}function E(e){const t=e.architecture,i=e.erf.map(s=>s.energy_radii.r90),n=e.edge.top_layers[0],a=e.math_verification;return`
    <details class="model-detail">
      <summary>${e.label}: measured explanation</summary>
      <div class="detail-grid">
        <div>
          <h4>Measured geometry</h4>
          <p>Channels ${t.channels.join(" → ")} at ${t.resolutions.map(s=>s.join("×")).join(" → ")}. Parameter count ${t.total_parameters.toLocaleString()}.</p>
        </div>
        <div>
          <h4>ERF behavior</h4>
          <p>r90 by stage: ${i.map(s=>r.px(s)).join(", ")}. Late-stage reach is larger because features are lower-resolution and token-mixed.</p>
        </div>
        <div>
          <h4>Selective scan</h4>
          <p>Δ diff ${r.n(a.delta_diff,4)}, B diff ${r.n(a.B_diff,4)}, C diff ${r.n(a.C_diff,4)}. These nonzero differences verify input-dependent scan parameters for the inspected mixer.</p>
        </div>
        <div>
          <h4>Edge response</h4>
          <p>Top ESI layer ${n.layer}, ESI ${r.n(n.edge_selectivity_index,4)}. This is a controlled stimulus result, not a natural-image edge benchmark.</p>
        </div>
      </div>
    </details>
  `}function k(){return`
    <section id="figures" class="section alt">
      <div class="section-head">
        <div class="eyebrow">04 / Generated figures</div>
        <h2>Visual Evidence</h2>
        <p>The deployed app ships with generated PNG outputs copied into <code>docs/website/public/results/figures</code>.</p>
      </div>
      <div class="figure-grid">
        ${[["Family ERF r90","family_erf_r90.png","Compares stage-wise effective receptive field growth across model sizes."],["Throughput","family_throughput.png","Measured local throughput with batch-size fallback if memory constrained."],["Top ESI","family_edge_top_esi.png","Highest edge selectivity index per model."],["Frequency Dominance","family_frequency_dominance.png","Dominant raw-activation stage for each sinusoidal frequency."],["SSM Summary","family_ssm_summary.png","Mean eigenvalue magnitude and impulse half-life by model."],["Stage Diagram","mambavision_stage_diagram.png","Tiny checkpoint architecture map generated by script 01."],["ERF Heatmaps","erf_per_stage.png","Single-model detailed ERF heatmaps for the Tiny checkpoint."],["Layer Summary","layer_function_summary.png","Representative layer function panels from script 05."]].map(([t,i,n])=>`
          <figure class="figure-card">
            <img src="${m(i)}" alt="${t}" loading="lazy" />
            <figcaption><b>${t}</b><span>${n}</span></figcaption>
          </figure>
        `).join("")}
      </div>
    </section>
  `}function A(){return`
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
  `}function F(){return`
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
  `}function R(e){const t=f(e);p.innerHTML=`
    ${_()}
    ${$(t)}
    ${S(t)}
    ${x(t)}
    ${w(t)}
    ${F()}
    ${k()}
    ${A()}
    <footer>
      <p>Original model family: NVIDIA MambaVision. Local analysis JSON: <code>public/data/model_family_analysis.json</code>.</p>
      <pre>@article{hatamizadeh2024mambavision,
  title={MambaVision: A Hybrid Mamba-Transformer Vision Backbone},
  author={Hatamizadeh, Ali and Kautz, Jan},
  journal={arXiv preprint arXiv:2407.08083},
  year={2024}
}</pre>
    </footer>
  `,j(),L()}function j(){const e=[...document.querySelectorAll(".navlinks a")],t=[...document.querySelectorAll("section[id]")];window.addEventListener("scroll",()=>{const i=t.filter(n=>n.getBoundingClientRect().top<120).pop();e.forEach(n=>n.classList.toggle("active",i&&n.getAttribute("href")===`#${i.id}`))},{passive:!0})}function L(){const e=()=>{window.renderMathInElement?window.renderMathInElement(document.body,{delimiters:[{left:"$$",right:"$$",display:!0},{left:"\\(",right:"\\)",display:!1}]}):window.setTimeout(e,100)};e()}async function I(){try{const e=await fetch("data/model_family_analysis.json",{cache:"no-store"});if(!e.ok)throw new Error(`Failed to load data: ${e.status}`);const t=await e.json();R(t)}catch(e){p.innerHTML=`
      <main class="error-state">
        <h1>Could not load model-family data</h1>
        <p>${e.message}</p>
        <p>Run <code>analysis/scripts/08_model_family_analysis.py</code> and copy the JSON into <code>docs/website/public/data</code>.</p>
      </main>
    `}}I();
