const POLL_MS = 3000;
let selectedDevice = null;
let charts = {};

const $ = (sel) => document.querySelector(sel);

function stateBadgeClass(state) {
  return `badge badge-${state || "unknown"}`;
}

function fmt(n, digits = 1) {
  if (n === null || n === undefined) return "-";
  return Number(n).toFixed(digits);
}

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function renderGlobalStats(stats) {
  const counts = stats.device_state_counts || {};
  const parts = [
    ["Devices", stats.n_devices],
    ["Normal", counts.normal || 0],
    ["Degrading", (counts.degrading || 0) + (counts.critical || 0)],
    ["Failed/Recovering", (counts.failed || 0) + (counts.recovering || 0)],
    ["Active alerts", stats.active_alerts],
    ["Model", stats.model_available ? "loaded" : "training..."],
  ];
  $("#global-stats").innerHTML = parts
    .map(([label, val]) => `<div class="stat"><b>${val}</b><span>${label}</span></div>`)
    .join("");

  const fb = stats.feedback_stats || {};
  $("#trust-panel").innerHTML = `
    <span><b>${stats.alerts_suppressed_total || 0}</b> repeat alerts suppressed by feedback (fatigue reduction)</span>
    <span>Operator-confirmed precision: <b>${fb.precision !== null && fb.precision !== undefined ? (fb.precision * 100).toFixed(0) + "%" : "n/a"}</b> (${fb.total_rated || 0} rated)</span>
  `;
}

function renderFleet(devices) {
  $("#fleet-count").textContent = `${devices.length} devices`;
  $("#fleet-grid").innerHTML = devices
    .map((d) => {
      const l = d.latest || {};
      return `
      <div class="device-card" data-id="${d.device_id}">
        <div class="name">${d.name}</div>
        <div class="meta">${d.device_type} · ${d.site}</div>
        <div class="row">
          <span class="${stateBadgeClass(l.device_state)}">${l.device_state || "..."}</span>
          <span class="muted">${l.latency_ms !== undefined ? fmt(l.latency_ms) + " ms" : ""}</span>
        </div>
      </div>`;
    })
    .join("");

  document.querySelectorAll(".device-card").forEach((el) => {
    el.addEventListener("click", () => selectDevice(el.dataset.id, devices));
  });
}

function severityRank(s) {
  return { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 }[s] ?? -1;
}

function renderAlerts(alerts) {
  const active = alerts.filter((a) => a.status === "open" || a.status === "acknowledged");
  $("#alerts-count").textContent = `${active.length} active`;

  if (active.length === 0) {
    $("#alerts-feed").innerHTML = `<div class="empty-state">No predicted failures right now. Fleet is healthy.</div>`;
    return;
  }

  active.sort((a, b) => severityRank(b.severity) - severityRank(a.severity) || new Date(b.updated_at) - new Date(a.updated_at));

  $("#alerts-feed").innerHTML = active
    .map((a) => {
      const causes = (a.causes || [])
        .map((c) => `<li>${c.label} <span class="val">${fmt(c.value)}${c.unit}</span></li>`)
        .join("");
      const actions = (a.recommended_actions || []).map((x) => `<li>${x}</li>`).join("");
      const eta = a.eta_minutes !== null && a.eta_minutes !== undefined ? `~${Math.round(a.eta_minutes)} min to threshold` : "ETA unknown";
      return `
      <div class="alert-card ${a.severity}" data-id="${a.id}">
        <div class="alert-top">
          <div>
            <div class="alert-title sev-${a.severity}">${a.severity} · ${a.device_name}</div>
            <div class="alert-sub">${a.category.replace(/_/g, " ")} · ${(a.probability * 100).toFixed(0)}% risk · ${eta}</div>
          </div>
          <span class="status-pill">${a.status}</span>
        </div>
        <div class="alert-summary">${a.summary}</div>
        <ul class="cause-list">${causes}</ul>
        <ol class="actions-list">${actions}</ol>
        <div class="alert-controls">
          ${a.status === "open" ? `<button class="ack">Acknowledge</button>` : ""}
          <button class="tp">Confirm real issue</button>
          <button class="fp">False alarm</button>
        </div>
      </div>`;
    })
    .join("");

  document.querySelectorAll(".alert-card .ack").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      const id = e.target.closest(".alert-card").dataset.id;
      getJSON(`/api/alerts/${id}/acknowledge`, { method: "POST" }).then(refresh);
    })
  );
  document.querySelectorAll(".alert-card .tp").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      const id = e.target.closest(".alert-card").dataset.id;
      getJSON(`/api/alerts/${id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_true_positive: true }),
      }).then(refresh);
    })
  );
  document.querySelectorAll(".alert-card .fp").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      const id = e.target.closest(".alert-card").dataset.id;
      getJSON(`/api/alerts/${id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_true_positive: false }),
      }).then(refresh);
    })
  );
}

function upsertChart(id, labels, datasets) {
  const ctx = document.getElementById(id).getContext("2d");
  if (charts[id]) {
    charts[id].data.labels = labels;
    charts[id].data.datasets.forEach((ds, i) => (ds.data = datasets[i].data));
    charts[id].update("none");
    return;
  }
  charts[id] = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: true,
      plugins: { legend: { labels: { color: "#8a94aa", boxWidth: 10, font: { size: 10 } } } },
      scales: {
        x: { display: false },
        y: { ticks: { color: "#8a94aa", font: { size: 10 } }, grid: { color: "#232e45" } },
      },
      elements: { point: { radius: 0 }, line: { borderWidth: 2, tension: 0.25 } },
    },
  });
}

async function selectDevice(deviceId, devices) {
  selectedDevice = deviceId;
  const device = devices.find((d) => d.device_id === deviceId);
  $("#device-detail").classList.remove("hidden");
  $("#detail-title").textContent = `${device.name} — ${device.device_type} @ ${device.site}`;
  await refreshDeviceCharts();
}

async function refreshDeviceCharts() {
  if (!selectedDevice) return;
  const rows = await getJSON(`/api/devices/${selectedDevice}/metrics?limit=120`);
  const labels = rows.map((r) => new Date(r.ts).toLocaleTimeString());

  upsertChart("chart-latency", labels, [
    { label: "Latency (ms)", data: rows.map((r) => r.latency_ms), borderColor: "#4da3ff" },
    { label: "Jitter (ms)", data: rows.map((r) => r.jitter_ms), borderColor: "#6fb6ff" },
  ]);
  upsertChart("chart-loss", labels, [
    { label: "Packet loss (%)", data: rows.map((r) => r.packet_loss_pct), borderColor: "#ff5b5b" },
    { label: "Retransmits/min", data: rows.map((r) => r.retransmits_per_min), borderColor: "#ff9a4d" },
  ]);
  upsertChart("chart-cpu-mem", labels, [
    { label: "CPU %", data: rows.map((r) => r.cpu_pct), borderColor: "#f4c94d" },
    { label: "Memory %", data: rows.map((r) => r.memory_pct), borderColor: "#33c281" },
  ]);
  upsertChart("chart-errors", labels, [
    { label: "Interface errors/min", data: rows.map((r) => r.interface_errors_per_min), borderColor: "#ff5b5b" },
    { label: "Temperature (C)", data: rows.map((r) => r.temperature_c), borderColor: "#8a94aa" },
  ]);
}

$("#detail-close").addEventListener("click", () => {
  selectedDevice = null;
  $("#device-detail").classList.add("hidden");
});

async function refresh() {
  try {
    const [devices, alerts, stats] = await Promise.all([
      getJSON("/api/devices"),
      getJSON("/api/alerts"),
      getJSON("/api/stats"),
    ]);
    renderGlobalStats(stats);
    renderFleet(devices);
    renderAlerts(alerts);
    if (selectedDevice) await refreshDeviceCharts();
  } catch (err) {
    console.error("refresh failed", err);
  }
}

refresh();
setInterval(refresh, POLL_MS);
