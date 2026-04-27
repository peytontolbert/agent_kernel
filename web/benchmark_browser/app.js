const state = {
  index: null,
  activePanel: "overview",
  query: "",
  repo: "",
  dataset: "",
};

const byId = (id) => document.getElementById(id);
const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const number = (value) => (Number.isFinite(Number(value)) ? Number(value) : null);
const esc = (value) =>
  String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

function metric(label, value) {
  return `<article class="metric"><span class="label">${esc(label)}</span><span class="value">${esc(value)}</span></article>`;
}

function pill(text, kind = "") {
  return `<span class="pill ${kind}">${esc(text)}</span>`;
}

function card(title, meta, body, badge = "") {
  return `<article class="card">
    <div class="cardHeader">
      <div><h3>${esc(title)}</h3>${meta ? `<p class="meta">${esc(meta)}</p>` : ""}</div>
      ${badge}
    </div>
    ${body}
  </article>`;
}

function progress(rate) {
  const width = Math.max(0, Math.min(100, Math.round((Number(rate) || 0) * 100)));
  return `<div class="progress"><div class="bar" style="width: ${width}%"></div></div>`;
}

function formatMetric(value, metric) {
  const parsed = number(value);
  if (parsed === null) return "No evidence";
  if (metric.includes("rate")) return pct(parsed);
  if (metric.includes("rating")) return Math.round(parsed).toLocaleString();
  return parsed.toLocaleString();
}

function statusKind(status) {
  if (status === "met") return "good";
  if (status === "partial" || status === "tracked") return "warn";
  return "bad";
}

function statusLabel(status) {
  return {
    met: "met",
    partial: "partial",
    tracked: "tracked",
    no_evidence: "no evidence",
  }[status] || status;
}

function allInstances() {
  return state.index.datasets.flatMap((dataset) =>
    dataset.instances.map((instance) => ({ ...instance, dataset: dataset.name, datasetPath: dataset.path })),
  );
}

function filteredInstances() {
  const query = state.query.toLowerCase();
  return allInstances().filter((instance) => {
    const matchesRepo = !state.repo || instance.repo === state.repo;
    const matchesDataset = !state.dataset || instance.dataset === state.dataset;
    const haystack = [
      instance.instance_id,
      instance.repo,
      instance.dataset,
      instance.version,
      instance.created_at,
      instance.problem_statement,
      instance.fail_to_pass.join(" "),
    ]
      .join(" ")
      .toLowerCase();
    return matchesRepo && matchesDataset && (!query || haystack.includes(query));
  });
}

function renderMetrics() {
  const datasetTasks = state.index.datasets.reduce((sum, dataset) => sum + dataset.task_count, 0);
  const resolved = state.index.results.reduce((sum, result) => sum + result.resolved, 0);
  const a8 = state.index.a8_progress || { met_gate_count: 0, gate_count: 0, claim_ready: false };
  const predictions = state.index.predictions.reduce((sum, file) => sum + file.prediction_count, 0);
  byId("metrics").innerHTML = [
    metric("A8 gates met", `${a8.met_gate_count}/${a8.gate_count}`),
    metric("A8 claim ready", a8.claim_ready ? "Yes" : "No"),
    metric("Resolved local evals", resolved),
    metric("Prediction records", predictions),
    metric("Dataset tasks indexed", datasetTasks),
  ].join("");
}

function renderGateCard(gate) {
  const current = formatMetric(gate.current_value, gate.metric);
  const target = gate.target;
  const countLine =
    gate.required_count && gate.dataset_total
      ? `<p>${gate.current_numerator ?? 0}/${gate.current_denominator ?? gate.dataset_total} observed. Full gate needs ${gate.required_count}/${gate.dataset_total}.</p>`
      : gate.current_denominator
        ? `<p>${gate.current_numerator ?? 0}/${gate.current_denominator} observed.</p>`
        : "";
  const caveats = (gate.caveats || []).map((item) => `<li>${esc(item)}</li>`).join("");
  return card(
    gate.label,
    gate.evidence_path || gate.run_spec_path || "No local evidence yet",
    `${progress(gate.progress_to_gate)}
     <p><strong>${esc(current)}</strong> toward ${esc(target)}</p>
     ${countLine}
     ${caveats ? `<ul class="caveats">${caveats}</ul>` : ""}`,
    pill(statusLabel(gate.status), statusKind(gate.status)),
  );
}

function renderOverview() {
  const thresholds = state.index.targets.thresholds || {};
  const a8 = state.index.a8_progress || { benchmark_gates: [], support_gates: [] };
  const targetRows = Object.entries(thresholds)
    .map(([key, value]) => `<tr><td>${esc(key)}</td><td>${esc(value)}</td></tr>`)
    .join("");
  const latestResults = [...state.index.results]
    .sort((a, b) => b.path.localeCompare(a.path))
    .slice(0, 8)
    .map((result) =>
      card(
        result.name,
        result.path,
        `${progress(result.resolve_rate)}
        <p>${result.resolved}/${result.total} resolved, ${result.errors} errors, ${result.incomplete} incomplete</p>`,
        pill(pct(result.resolve_rate), result.resolve_rate === 1 ? "good" : result.errors ? "bad" : "warn"),
      ),
    )
    .join("");

  byId("overview").innerHTML = `
    <section class="sectionHead">
      <div>
        <h2>A8 Superhuman Coding Progress</h2>
        <p class="meta">Human-readable tracking for the benchmark gates in the local A8 target contract.</p>
      </div>
      ${pill(a8.claim_ready ? "claim ready" : "not claim ready", a8.claim_ready ? "good" : "bad")}
    </section>
    <h2>Benchmark Gates</h2>
    <div class="grid">${a8.benchmark_gates.map(renderGateCard).join("")}</div>
    <h2>Support Gates</h2>
    <div class="grid">${a8.support_gates.map(renderGateCard).join("")}</div>
    <div class="grid">
      ${card(
        "A8 Target Contract",
        state.index.targets.path || "No target packet indexed",
        `<div class="tableWrap"><table><thead><tr><th>Metric</th><th>Target</th></tr></thead><tbody>${targetRows}</tbody></table></div>`,
      )}
      ${card(
        "Indexed Data",
        state.index.root,
        `<p>${state.index.datasets.length} datasets, ${state.index.results.length} result files, ${state.index.predictions.length} prediction files, ${state.index.run_specs.length} run specs.</p>`,
      )}
    </div>
    <h2>Local Result Files</h2>
    <div class="grid">${latestResults || '<div class="empty">No result files indexed.</div>'}</div>
  `;
}

function renderDatasets() {
  const rows = filteredInstances()
    .slice(0, 500)
    .map(
      (instance) => `<tr>
        <td><button class="linkButton" data-instance="${esc(instance.instance_id)}">${esc(instance.instance_id)}</button></td>
        <td>${esc(instance.repo)}</td>
        <td>${esc(instance.dataset)}</td>
        <td>${esc(instance.version)}</td>
        <td>${esc(instance.fail_to_pass.length)}</td>
        <td>${esc(instance.pass_to_pass_count)}</td>
        <td>${esc(instance.created_at)}</td>
      </tr>`,
    )
    .join("");
  const datasetCards = state.index.datasets
    .map((dataset) =>
      card(
        dataset.name,
        dataset.path,
        `<p>${dataset.task_count} tasks across ${Object.keys(dataset.repo_counts).length} repositories.</p>
         <p class="meta">${Object.entries(dataset.repo_counts)
           .map(([repo, count]) => `${repo}: ${count}`)
           .join(" · ")}</p>`,
      ),
    )
    .join("");
  byId("datasets").innerHTML = `
    <div class="grid">${datasetCards}</div>
    <div class="tableWrap">
      <table>
        <thead><tr><th>Instance</th><th>Repo</th><th>Dataset</th><th>Version</th><th>F2P</th><th>P2P</th><th>Created</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="7" class="empty">No matching tasks.</td></tr>'}</tbody>
      </table>
    </div>
  `;
}

function renderResults() {
  const rows = state.index.results
    .filter((result) => !state.query || JSON.stringify(result).toLowerCase().includes(state.query.toLowerCase()))
    .map(
      (result) => `<tr>
        <td>${esc(result.name)}</td>
        <td>${esc(result.total)}</td>
        <td>${esc(result.completed)}</td>
        <td>${esc(result.resolved)}</td>
        <td>${esc(result.unresolved)}</td>
        <td>${esc(result.errors)}</td>
        <td>${esc(result.incomplete)}</td>
        <td>${esc(pct(result.resolve_rate))}</td>
        <td>${esc(result.path)}</td>
      </tr>`,
    )
    .join("");
  byId("results").innerHTML = `<div class="tableWrap">
    <table>
      <thead><tr><th>Run</th><th>Total</th><th>Completed</th><th>Resolved</th><th>Unresolved</th><th>Errors</th><th>Incomplete</th><th>Rate</th><th>Path</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="9" class="empty">No matching results.</td></tr>'}</tbody>
    </table>
  </div>`;
}

function renderPredictions() {
  const rows = state.index.predictions
    .filter((file) => !state.query || JSON.stringify(file).toLowerCase().includes(state.query.toLowerCase()))
    .map(
      (file) => `<tr>
        <td>${esc(file.name)}</td>
        <td>${esc(file.prediction_count)}</td>
        <td>${esc(Object.entries(file.repo_counts).map(([repo, count]) => `${repo}: ${count}`).join(" · "))}</td>
        <td>${esc(file.path)}</td>
      </tr>`,
    )
    .join("");
  byId("predictions").innerHTML = `<div class="tableWrap">
    <table>
      <thead><tr><th>Prediction File</th><th>Records</th><th>Repos</th><th>Path</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="4" class="empty">No matching prediction files.</td></tr>'}</tbody>
    </table>
  </div>`;
}

function renderSources() {
  const sources = state.index.dataset_sources?.sources || [];
  const rows = sources
    .filter((source) => !state.query || JSON.stringify(source).toLowerCase().includes(state.query.toLowerCase()))
    .map((source) => {
      const status = source.status || (source.exists ? "available" : "missing");
      const kind = status === "available" ? "good" : status === "missing" || status === "error" ? "bad" : "warn";
      return `<tr>
        <td>${esc(source.label || source.benchmark)}</td>
        <td>${source.required_for_a8 ? pill("A8 gate", "warn") : pill("support")}</td>
        <td>${pill(status, kind)}</td>
        <td>${esc(source.kind)}</td>
        <td>${esc(source.rows ?? "")}</td>
        <td>${esc(source.dataset_id || source.repo_url || "")}</td>
        <td>${esc(source.local_path || "")}</td>
        <td>${esc(source.notes || source.error || "")}</td>
      </tr>`;
    })
    .join("");
  byId("sources").innerHTML = `
    ${card(
      "Dataset Source Coverage",
      state.index.dataset_sources?.status_path || "No status file yet",
      `<p>Use <code>python scripts/download_a8_benchmark_datasets.py --include-git</code> to refresh downloadable sources. Large or credentialed sources are shown explicitly.</p>`,
    )}
    <div class="tableWrap">
      <table>
        <thead><tr><th>Source</th><th>Role</th><th>Status</th><th>Kind</th><th>Rows</th><th>Remote</th><th>Local Path</th><th>Notes</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="8" class="empty">No dataset sources configured.</td></tr>'}</tbody>
      </table>
    </div>`;
}

function renderSpecs() {
  const rows = state.index.run_specs
    .filter((spec) => !state.query || JSON.stringify(spec).toLowerCase().includes(state.query.toLowerCase()))
    .map(
      (spec) => `<tr>
        <td>${esc(spec.name)}</td>
        <td>${esc(spec.benchmark)}</td>
        <td>${spec.ready_to_run ? pill("ready", "good") : pill("not ready", "warn")}</td>
        <td>${esc(spec.runner_kind)}</td>
        <td>${esc(spec.dataset_name)}</td>
        <td>${esc(spec.path)}</td>
      </tr>`,
    )
    .join("");
  byId("specs").innerHTML = `<div class="tableWrap">
    <table>
      <thead><tr><th>Spec</th><th>Benchmark</th><th>Status</th><th>Runner</th><th>Dataset</th><th>Path</th></tr></thead>
      <tbody>${rows || '<tr><td colspan="6" class="empty">No matching run specs.</td></tr>'}</tbody>
    </table>
  </div>`;
}

function render() {
  renderMetrics();
  renderOverview();
  renderDatasets();
  renderResults();
  renderPredictions();
  renderSources();
  renderSpecs();
}

function populateFilters() {
  const repos = [...new Set(allInstances().map((instance) => instance.repo).filter(Boolean))].sort();
  byId("repoFilter").innerHTML =
    '<option value="">All repositories</option>' +
    repos.map((repo) => `<option value="${esc(repo)}">${esc(repo)}</option>`).join("");
  byId("datasetFilter").innerHTML =
    '<option value="">All datasets</option>' +
    state.index.datasets.map((dataset) => `<option value="${esc(dataset.name)}">${esc(dataset.name)}</option>`).join("");
}

function openDetail(instanceId) {
  const instance = allInstances().find((item) => item.instance_id === instanceId);
  if (!instance) return;
  byId("detailTitle").textContent = instance.instance_id;
  byId("detailMeta").textContent = `${instance.repo} · ${instance.dataset} · ${instance.version || "unknown version"}`;
  byId("detailBody").innerHTML = `
    <section class="detailSection">
      <h3>Problem Statement</h3>
      <pre>${esc(instance.problem_statement || "No problem statement indexed.")}</pre>
    </section>
    <section class="detailSection">
      <h3>Fail-to-Pass Tests</h3>
      <pre>${esc(instance.fail_to_pass.join("\n") || "No fail-to-pass tests indexed.")}</pre>
    </section>
    <section class="detailSection">
      <h3>Hints</h3>
      <pre>${esc(instance.hints_text || "No hints indexed.")}</pre>
    </section>
    <section class="detailSection">
      <h3>Metadata</h3>
      <pre>${esc(JSON.stringify({
        created_at: instance.created_at,
        base_commit: instance.base_commit,
        pass_to_pass_count: instance.pass_to_pass_count,
        has_reference_patch: instance.has_reference_patch,
        has_test_patch: instance.has_test_patch,
        dataset_path: instance.datasetPath,
      }, null, 2))}</pre>
    </section>
  `;
  byId("detailDialog").showModal();
}

function wireEvents() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.activePanel = button.dataset.panel;
      document.querySelectorAll(".tab").forEach((item) => item.classList.toggle("active", item === button));
      document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === state.activePanel));
    });
  });
  byId("searchInput").addEventListener("input", (event) => {
    state.query = event.target.value;
    render();
  });
  byId("repoFilter").addEventListener("change", (event) => {
    state.repo = event.target.value;
    renderDatasets();
  });
  byId("datasetFilter").addEventListener("change", (event) => {
    state.dataset = event.target.value;
    renderDatasets();
  });
  document.body.addEventListener("click", (event) => {
    const button = event.target.closest("[data-instance]");
    if (button) openDetail(button.dataset.instance);
  });
  byId("closeDetail").addEventListener("click", () => byId("detailDialog").close());
}

async function load() {
  const response = await fetch("benchmark_index.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`Failed to load benchmark_index.json: ${response.status}`);
  state.index = await response.json();
  byId("generatedAt").textContent = `Generated ${state.index.generated_at}`;
  populateFilters();
  render();
}

wireEvents();
load().catch((error) => {
  byId("generatedAt").textContent = error.message;
  byId("overview").classList.add("active");
  byId("overview").innerHTML =
    '<div class="empty">Run scripts/build_benchmark_browser_index.py and serve this directory over HTTP.</div>';
});
