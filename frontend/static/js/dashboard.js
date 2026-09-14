const POLL_MS = 3000;
const SEVERITY_RANK = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 };

const charts = {};
let selectedDevice = null;
let deviceIndex = {};

const $ = (sel) => document.querySelector(sel);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fmt(n, digits = 1) {
  return n === null || n === undefined ? "-" : Number(n).toFixed(digits);
}

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

function postJSON(url, body) {
  return getJSON(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
}

function renderGlobalStats(stats) {
  const counts = stats.device_state_counts || {};
  const tiles = [
    ["Devices", stats.n_devices],
    ["Normal", counts.normal || 0],
    ["Degrading", (counts.degrading || 0) + (counts.critical || 0)],
    ["Failed/Recovering", (counts.failed || 0) + (counts.recovering || 0)],
    ["Active alerts", stats.active_alerts],
    ["Model", stats.model_available ? "loaded" : "not trained"],
  ];
  $("#global-stats").innerHTML = tiles
    .map(([label, val]) => `<div class="stat"><b>${esc(val)}</b><span>${esc(label)}</span></div>`)
    .join("");

  const fb = stats.feedback_stats || {};
  const precision = fb.precision === null || fb.precision === undefined
    ? "n/a"
    : `${(fb.precision * 100).toFixed(0)}%`;
  $("#trust-panel").innerHTML = `
    <span><b>${esc(stats.alerts_suppressed_total || 0)}</b> repeat alerts suppressed by feedback (fatigue reduction)</span>
    <span>Operator-confirmed precision: <b>${esc(precision)}</b> (${esc(fb.total_rated || 0)} rated)</span>
  `;
}

function renderFleet(devices) {
  deviceIndex = Object.fromEntries(devices.map((d) => [d.device_id, d]));
  $("#fleet-count").textContent = `${devices.length} devices`;
  $("#fleet-grid").innerHTML = devices
    .map((d) => {
      const latest = d.latest || {};
      const state = latest.device_state || "unknown";
      const latency = latest.latency_ms !== undefined ? `${fmt(latest.latency_ms)} ms` : "";
      return `
      <div class="device-card" data-device="${esc(d.device_id)}">
        <div class="name">${esc(d.name)}</div>
        <div class="meta">${esc(d.device_type)} · ${esc(d.site)}</div>
        <div class="row">
          <span class="badge badge-${esc(state)}">${esc(state)}</span>
          <span class="muted">${esc(latency)}</span>
        </div>
      </div>`;
    })
    .join("");
}

function renderAlerts(alerts) {
  const active = alerts.filter((a) => a.status === "open" || a.status === "acknowledged");
  $("#alerts-count").textContent = `${active.length} active`;

  if (active.length === 0) {
    $("#alerts-feed").innerHTML =
      `<div class="empty-state">No predicted failures right now. Fleet is healthy.</div>`;
    return;
  }

  active.sort(
    (a, b) =>
      SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity] ||
      new Date(b.updated_at) - new Date(a.updated_at)
  );

  $("#alerts-feed").innerHTML = active
    .map((a) => {
      const causes = (a.causes || [])
        .map((c) => `<li>${esc(c.label)} <span class="val">${fmt(c.value)}${esc(c.unit)}</span></li>`)
        .join("");
      const actions = (a.recommended_actions || []).map((x) => `<li>${esc(x)}</li>`).join("");
      const eta = a.eta_minutes === null || a.eta_minutes === undefined
        ? "ETA unknown"
        : `~${Math.round(a.eta_minutes)} min to threshold`;
      return `
      <div class="alert-card ${esc(a.severity)}" data-alert="${esc(a.id)}">
        <div class="alert-top">
          <div>
            <div class="alert-title sev-${esc(a.severity)}">${esc(a.severity)} · ${esc(a.device_name)}</div>
            <div class="alert-sub">
              ${esc(a.category.replace(/_/g, " "))} · ${(a.probability * 100).toFixed(0)}% risk · ${esc(eta)}
            </div>
          </div>
          <span class="status-pill">${esc(a.status)}</span>
        </div>
        <div class="alert-summary">${esc(a.summary)}</div>
        <ul class="cause-list">${causes}</ul>
        <ol class="actions-list">${actions}</ol>
        <div class="alert-controls">
          ${a.status === "open" ? `<button data-action="ack">Acknowledge</button>` : ""}
          <button class="tp" data-action="true-positive">Confirm real issue</button>
          <button class="fp" data-action="false-positive">False alarm</button>
        </div>
      </div>`;
    })
    .join("");
}

function chartsAvailable() {
  return typeof Chart !== "undefined";
}

function upsertChart(id, labels, datasets) {
  if (charts[id]) {
    charts[id].data.labels = labels;
    charts[id].data.datasets.forEach((ds, i) => (ds.data = datasets[i].data));
    charts[id].update("none");
    return;
  }
  charts[id] = new Chart(document.getElementById(id).getContext("2d"), {
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

async function refreshDeviceCharts() {
  if (!selectedDevice) return;
  // Chart.js comes from a CDN; a monitoring dashboard may well run on a
  // network with no outbound internet. Degrade to "no charts" rather than
  // breaking the fleet and alert views, which matter more.
  if (!chartsAvailable()) {
    $("#chart-fallback").hidden = false;
    $("#chart-grid").hidden = true;
    return;
  }
  const rows = await getJSON(`/api/devices/${selectedDevice}/metrics?limit=120`);
  const labels = rows.map((r) => new Date(r.timestamp).toLocaleTimeString());

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
    { label: "Temperature (°C)", data: rows.map((r) => r.temperature_c), borderColor: "#8a94aa" },
  ]);
}

$("#fleet-grid").addEventListener("click", async (e) => {
  const card = e.target.closest(".device-card");
  if (!card) return;
  selectedDevice = card.dataset.device;
  const device = deviceIndex[selectedDevice];
  $("#device-detail").hidden = false;
  $("#detail-title").textContent = `${device.name} — ${device.device_type} @ ${device.site}`;
  await refreshDeviceCharts();
});

$("#detail-close").addEventListener("click", () => {
  selectedDevice = null;
  $("#device-detail").hidden = true;
});

$("#alerts-feed").addEventListener("click", async (e) => {
  const button = e.target.closest("button[data-action]");
  if (!button) return;
  const alertId = button.closest(".alert-card").dataset.alert;
  const action = button.dataset.action;

  if (action === "ack") {
    await postJSON(`/api/alerts/${alertId}/acknowledge`);
  } else {
    await postJSON(`/api/alerts/${alertId}/feedback`, {
      is_true_positive: action === "true-positive",
    });
  }
  await refresh();
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
  } catch (err) {
    console.error("refresh failed", err);
  }

  // Isolated: a charting failure must never stop fleet/alert updates.
  try {
    await refreshDeviceCharts();
  } catch (err) {
    console.error("chart refresh failed", err);
  }
}

refresh();
setInterval(refresh, POLL_MS);
