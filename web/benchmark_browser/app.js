const state = {
  index: null,
  live: null,
  liveRenderSignature: "",
  activePanel: "overview",
  query: "",
  repo: "",
  dataset: "",
};

const byId = (id) => document.getElementById(id);
const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const number = (value) => (Number.isFinite(Number(value)) ? Number(value) : null);
const fmtSeconds = (value) => {
  const seconds = number(value);
  if (seconds === null) return "unknown";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return `${minutes}m ${remainder}s`;
};
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

function liveRenderSignature(live) {
  if (!live) return "";
  const queueSnapshots = Object.values(live.queue_snapshots_by_benchmark || {});
  return JSON.stringify({
    active_runs: Object.values(live.active_runs_by_benchmark || {}).map((run) => ({
      benchmark: run.benchmark,
      path: run.path,
      phase: run.active_phase?.name,
      pid: run.active_phase?.pid,
      completed_phase_count: run.completed_phase_count,
      completed_phases: (run.completed_phases || []).map((phase) => ({
        name: phase.name,
        returncode: phase.returncode,
      })),
    })),
    queues: queueSnapshots.map((snapshot) => ({
      benchmark: snapshot.benchmark,
      total_jobs: snapshot.total_jobs,
      terminal_jobs: snapshot.terminal_jobs,
      completed_jobs: snapshot.completed_jobs,
      safe_stop_jobs: snapshot.safe_stop_jobs,
      queued_jobs: snapshot.queued_jobs,
      progress_rate: snapshot.progress_rate,
      state_counts: snapshot.state_counts,
      outcome_counts: snapshot.outcome_counts,
    })),
    recent_queue_events: queueSnapshots.flatMap((snapshot) => snapshot.recent_events || []).slice(-30).map((event) => ({
      at: event.at,
      event: event.event,
      task_id: event.task_id,
      state: event.state,
      outcome: event.outcome,
    })),
  });
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
  const activeHarnesses = Object.keys(state.live?.active_runs_by_benchmark || {}).length;
  byId("metrics").innerHTML = [
    metric("A8 gates met", `${a8.met_gate_count}/${a8.gate_count}`),
    metric("A8 claim ready", a8.claim_ready ? "Yes" : "No"),
    metric("Active harnesses", activeHarnesses),
    metric("Resolved local evals", resolved),
    metric("Prediction records", predictions),
    metric("Dataset tasks indexed", datasetTasks),
  ].join("");
}

function activeRunForGate(gate) {
  return state.live?.active_runs_by_benchmark?.[gate.benchmark] || gate.active_run || null;
}

function queueSnapshotForGate(gate) {
  return state.live?.queue_snapshots_by_benchmark?.[gate.benchmark] || null;
}

function renderActiveRun(run) {
  if (!run || !run.active_phase) return "";
  const phase = run.active_phase;
  const phaseProgress = run.phase_progress || {};
  const elapsed = number(phase.elapsed_seconds);
  const elapsedText = elapsed === null ? "unknown elapsed" : `${Math.round(elapsed)}s elapsed`;
  const processed = number(phaseProgress.processed_items);
  const total = number(phaseProgress.total_items);
  const selected = number(phaseProgress.selected_tasks);
  const prepProgress =
    phaseProgress.status && total
      ? `<p class="meta">Prep ${esc(phaseProgress.status)} · ${esc(processed ?? 0)}/${esc(total)} dataset rows scanned · ${esc(selected ?? 0)} tasks selected · current ${esc(phaseProgress.current_instance_id || "n/a")}.</p>`
      : "";
  return `<div class="liveRun">
    <div>${pill("live", "good")} <strong>${esc(phase.name || "active phase")}</strong></div>
    <p class="meta">${esc(elapsedText)} · pid ${esc(phase.pid ?? "")} · heartbeat ${esc(phase.heartbeat_at || "")}</p>
    ${prepProgress}
    <p class="meta">${esc(run.path || "")}</p>
  </div>`;
}

function renderGateCard(gate) {
  const current = formatMetric(gate.current_value, gate.metric);
  const target = gate.target;
  const activeRun = activeRunForGate(gate);
  const queueSnapshot = queueSnapshotForGate(gate);
  const countLine =
    gate.required_count && gate.dataset_total
      ? `<p>${gate.current_numerator ?? 0}/${gate.current_denominator ?? gate.dataset_total} observed. Full gate needs ${gate.required_count}/${gate.dataset_total}.</p>`
      : gate.current_denominator
        ? `<p>${gate.current_numerator ?? 0}/${gate.current_denominator} observed.</p>`
        : "";
  const queueLine = queueSnapshot
    ? `<p>${esc(queueSnapshot.completed_jobs)}/${esc(queueSnapshot.total_jobs)} autonomous patch jobs completed · ${esc(queueSnapshot.safe_stop_jobs)} safe-stop · ${esc(queueSnapshot.queued_jobs)} queued.</p>`
    : "";
  const caveats = (gate.caveats || []).map((item) => `<li>${esc(item)}</li>`).join("");
  return card(
    gate.label,
    gate.evidence_path || gate.run_spec_path || "No local evidence yet",
    `${progress(gate.progress_to_gate)}
     <p><strong>${esc(current)}</strong> toward ${esc(target)}</p>
     ${countLine}
     ${queueLine}
     ${renderActiveRun(activeRun)}
     ${caveats ? `<ul class="caveats">${caveats}</ul>` : ""}`,
    pill(statusLabel(gate.status), statusKind(gate.status)),
  );
}

function renderOverview() {
  const thresholds = state.index.targets.thresholds || {};
  const a8 = state.index.a8_progress || { benchmark_gates: [], support_gates: [] };
  const standalone = state.index.standalone_leaderboards || { gates: [] };
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
    <section class="sectionHead">
      <div>
        <h2>Standalone Leaderboards</h2>
        <p class="meta">Public online leaderboard runs tracked separately from the A8 lane. These are agent+model submissions, not A8 promotion gates.</p>
      </div>
      ${pill("not A8 lane", "warn")}
    </section>
    <div class="grid">${standalone.gates.map(renderGateCard).join("") || '<div class="empty">No standalone leaderboard benchmarks configured.</div>'}</div>
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

function renderLive() {
  const live = state.live || {};
  const activeRuns = Object.values(live.active_runs_by_benchmark || {});
  const queueSnapshots = Object.values(live.queue_snapshots_by_benchmark || {});
  const officialScores = Object.values(live.official_scores_by_benchmark || {});
  const rollingScores = Object.values(live.rolling_scores || {});
  const events = live.semantic_events || [];
  const livePanel = byId("live");
  const previousScroll = livePanel.querySelector(".eventList")?.scrollTop ?? 0;
  const primaryRun = activeRuns[0] || {};
  const primaryPhase = primaryRun.active_phase || {};
  const primaryQueue = queueSnapshots[0] || {};
  const terminal = Number(primaryQueue.terminal_jobs || 0);
  const total = Number(primaryQueue.total_jobs || 0);
  const completed = Number(primaryQueue.completed_jobs || 0);
  const safeStop = Number(primaryQueue.safe_stop_jobs || 0);
  const queued = Number(primaryQueue.queued_jobs || 0);
  const activeJobs = Number(primaryQueue.active_jobs || 0);
  const failed = Number(primaryQueue.failed_jobs || 0);
  const progressPct = Math.round((Number(primaryQueue.progress_rate) || 0) * 100);
  const visibleScore = [...rollingScores, ...officialScores].find(
    (score) => score.status === "partial" || score.status === "available",
  );
  const visibleScoreRate = number(visibleScore?.resolve_rate);
  const scoreText = visibleScore
    ? `${pct(visibleScoreRate || 0)} · ${visibleScore.resolved_count ?? 0}/${visibleScore.task_count ?? 0}`
    : "pending";
  const scoreLabel = visibleScore?.status === "partial" ? "partial score" : "official score";
  const countCloud = (counts = {}) =>
    Object.entries(counts)
      .map(([key, value]) => `<span>${esc(key)} <strong>${esc(value)}</strong></span>`)
      .join("");
  const activeCards = activeRuns
    .map((run) => {
      const completedPhases = (run.completed_phases || [])
        .slice(-4)
        .map((phase) => `<span>${esc(phase.name || "phase")} rc=${esc(phase.returncode ?? "")}</span>`)
        .join("");
      return card(
        run.benchmark || "active harness",
        run.path || "",
        `${renderActiveRun(run)}
         <p>${esc(run.completed_phase_count || 0)} completed harness phases.</p>
         ${completedPhases ? `<div class="countCloud">${completedPhases}</div>` : ""}`,
        pill("running", "good"),
      );
    })
    .join("");
  const queueCards = queueSnapshots
    .map((snapshot) =>
      `<article class="queueTile">
        <div class="ring" style="--pct: ${Math.round((Number(snapshot.progress_rate) || 0) * 100)}">
          <span>${Math.round((Number(snapshot.progress_rate) || 0) * 100)}%</span>
        </div>
        <div>
          <h3>${esc(snapshot.benchmark)} queue</h3>
          <p class="meta">${esc(snapshot.queue_path)}</p>
          <div class="queueStats">
            <span><strong>${esc(snapshot.active_jobs)}</strong> active</span>
            <span><strong>${esc(snapshot.completed_jobs)}</strong> completed</span>
            <span><strong>${esc(snapshot.safe_stop_jobs)}</strong> safe-stop</span>
            <span><strong>${esc(snapshot.failed_jobs)}</strong> failed</span>
            <span><strong>${esc(snapshot.queued_jobs)}</strong> queued</span>
            <span><strong>${esc(snapshot.total_jobs)}</strong> total</span>
          </div>
          <div class="countBlock">
            <p class="meta">State counts</p>
            <div class="countCloud">${countCloud(snapshot.state_counts)}</div>
          </div>
          <div class="countBlock">
            <p class="meta">Outcome counts</p>
            <div class="countCloud">${countCloud(snapshot.outcome_counts) || "<span>none yet</span>"}</div>
          </div>
        </div>
      </article>`,
    )
    .join("");
  const scoreCards = [...officialScores, ...rollingScores]
    .map((score) => {
      const available = score.status === "available" || score.status === "partial";
      const rate = number(score.resolve_rate);
      const resolved = score.resolved_count ?? "pending";
      const totalScoreTasks = score.task_count ?? "pending";
      const isRolling = score.final_leaderboard_score === false;
      const failedCount = score.failed_count ?? (number(score.task_count) !== null && number(score.resolved_count) !== null ? Number(score.task_count) - Number(score.resolved_count) : "pending");
      const passedList = (score.passed_instance_ids || []).slice(0, 8).join(" · ");
      const failedList = (score.failed_instance_ids || []).slice(0, 8).join(" · ");
      return card(
        `${score.benchmark || "benchmark"} ${isRolling ? "rolling score" : "official score"}`,
        score.summary_json || score.results_json || score.run_spec_path || "",
        `${available ? progress(rate || 0) : progress(0)}
         <p><strong>${available ? pct(rate || 0) : "pending"}</strong> ${score.status === "partial" ? "partial official subset" : "official"} resolve rate.</p>
         <p>${esc(resolved)}/${esc(totalScoreTasks)} resolved after official evaluator · ${esc(failedCount)} failed/evaluated-unresolved. Queue completion is not score.</p>
         ${score.prediction_count !== undefined && score.prediction_count !== null ? `<p class="meta">${esc(score.prediction_count)} predictions in subset.</p>` : ""}
         ${passedList ? `<p class="meta">Passed: ${esc(passedList)}</p>` : ""}
         ${failedList ? `<p class="meta">Failed: ${esc(failedList)}</p>` : ""}
         <p class="meta">source ${esc(score.score_source || "waiting for results.json")} · ${esc(score.score_kind || score.benchmark_role || "benchmark")}</p>`,
        pill(available ? (score.status === "partial" ? "partial score" : isRolling ? "subset scored" : "scored") : "pending", available ? "good" : "warn"),
      );
    })
    .join("");
  const eventRows = events
    .slice(0, 60)
    .map(
      (event) => `<li class="liveEvent">
        <span class="eventRail"></span>
        <div>
          <div class="eventLine">
            <span class="eventTime">${esc(event.at || "")}</span>
            ${pill(event.kind || "event")}
            <strong>${esc(event.benchmark || "")}</strong>
          </div>
          <p>${esc(event.message || "")}</p>
        </div>
      </li>`,
    )
    .join("");
  livePanel.innerHTML = `
    <section class="liveHero">
      <div class="liveSignal">
        <div class="signalOrb"><span></span></div>
        <div>
          <p class="eyebrow">Semantic Runtime</p>
          <h2>${esc(primaryPhase.name || "Waiting for harness activity")}</h2>
          <p class="meta">Benchmark ${esc(primaryRun.benchmark || "n/a")} · PID ${esc(primaryPhase.pid ?? "n/a")} · ${esc(fmtSeconds(primaryPhase.elapsed_seconds))} elapsed · heartbeat ${esc(primaryPhase.heartbeat_at || "not loaded")}</p>
        </div>
      </div>
      <div class="liveNumbers">
        <article><span>${esc(scoreText)}</span><small>${esc(scoreLabel)}</small></article>
        <article><span>${esc(terminal)}/${esc(total)}</span><small>terminal</small></article>
        <article><span>${esc(activeJobs)}</span><small>active</small></article>
        <article><span>${esc(completed)}</span><small>completed</small></article>
        <article><span>${esc(safeStop)}</span><small>safe-stop</small></article>
        <article><span>${esc(failed)}</span><small>failed</small></article>
        <article><span>${esc(queued)}</span><small>queued</small></article>
      </div>
      <div class="heroProgress">
        <div class="progress"><div class="bar" style="width: ${progressPct}%"></div></div>
        <p class="meta">${esc(progressPct)}% terminal queue progress. Score shown above is official-evaluator derived; partial scores use completed report.json files before final results.json exists.</p>
      </div>
    </section>

    <section class="sectionHead">
      <div>
        <h2>Runtime Deck</h2>
        <p class="meta">Harness phase, queue state, outcome counts, recent job history, artifact paths, and heartbeat timing from agent_kernel runtime logs.</p>
      </div>
      ${pill(live.generated_at ? "streaming" : "waiting", live.generated_at ? "good" : "warn")}
    </section>
    <div class="liveGrid">${activeCards || '<div class="empty">No active harness detected.</div>'}${scoreCards}${queueCards}</div>
    <article class="liveConsole">
      <div class="cardHeader">
        <div>
          <h3>Semantic Live Log</h3>
          <p class="meta">Phase heartbeats, queue summaries, and latest job-history events. Polls benchmark_live_status.json every 5 seconds without rebuilding the full index.</p>
        </div>
        ${pill(live.generated_at || "not loaded")}
      </div>
      <ol class="eventList">${eventRows || '<li class="empty">No live events yet.</li>'}</ol>
    </article>
  `;
  const nextLog = livePanel.querySelector(".eventList");
  if (nextLog && previousScroll > 0) nextLog.scrollTop = previousScroll;
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
        <td>${
          source.required_for_a8
            ? pill("A8 gate", "warn")
            : source.benchmark_role === "standalone_leaderboard"
              ? pill("leaderboard", "good")
              : pill("support")
        }</td>
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
  renderLive();
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
      if (state.activePanel === "live") renderLive();
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
  await loadLive();
  window.setInterval(loadLive, 5000);
}

async function loadLive() {
  if (!state.index) return;
  const response = await fetch(`benchmark_live_status.json?ts=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) return;
  state.live = await response.json();
  byId("generatedAt").textContent = `Generated ${state.index.generated_at} · live ${state.live.generated_at}`;
  const nextSignature = liveRenderSignature(state.live);
  const changed = nextSignature !== state.liveRenderSignature;
  state.liveRenderSignature = nextSignature;
  renderMetrics();
  if (!changed) return;
  if (state.activePanel === "overview") renderOverview();
  if (state.activePanel === "live") renderLive();
}

wireEvents();
load().catch((error) => {
  byId("generatedAt").textContent = error.message;
  byId("overview").classList.add("active");
  byId("overview").innerHTML =
    '<div class="empty">Run scripts/build_benchmark_browser_index.py and serve this directory over HTTP.</div>';
});
