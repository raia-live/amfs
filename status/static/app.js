const STATUS_LABELS = {
  operational: "Operational",
  maintenance: "Under Maintenance",
  degraded: "Degraded Performance",
  partial_outage: "Partial Outage",
  major_outage: "Major Outage",
  no_data: "No Data",
};

const LEGEND = [
  { key: "operational", label: "Operational" },
  { key: "degraded", label: "Degraded" },
  { key: "partial_outage", label: "Partial Outage" },
  { key: "major_outage", label: "Major Outage" },
  { key: "maintenance", label: "Maintenance" },
  { key: "no_data", label: "No Data" },
];

const OVERALL_HEADLINE = {
  operational: "All Systems Operational",
  maintenance: "Scheduled Maintenance",
  degraded: "Degraded Performance",
  partial_outage: "Partial Service Outage",
  major_outage: "Major Service Outage",
  no_data: "Status Unknown",
};

const tooltip = document.getElementById("tooltip");
let pollInterval = 60000;
let timer = null;

/* ---------- theme ---------- */
function initTheme() {
  const saved = localStorage.getItem("sl-status-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("sl-status-theme", next);
  });
}

/* ---------- helpers ---------- */
function fmtRelative(iso) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (secs < 10) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  return `${hrs}h ago`;
}

function fmtDate(iso) {
  return new Date(iso + "T00:00:00Z").toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });
}

/* ---------- rendering ---------- */
function renderOverall(overall, activeIncidents) {
  const el = document.getElementById("overall");
  const status = overall.status;
  el.className = "overall overall--" + status;
  document.getElementById("overall-title").textContent =
    OVERALL_HEADLINE[status] || overall.label;

  let sub = "";
  if (activeIncidents && activeIncidents.length) {
    sub = `${activeIncidents.length} active incident${
      activeIncidents.length > 1 ? "s" : ""
    } — see details below.`;
  } else if (status === "operational") {
    sub = "All SenseLab production services are running normally.";
  } else if (status === "no_data") {
    sub = "Collecting health data from production services…";
  } else {
    sub = "We are aware of an issue and investigating.";
  }
  document.getElementById("overall-sub").textContent = sub;
}

function uptimeBar(day) {
  const bar = document.createElement("div");
  bar.className = "uptime-bar b-" + day.status;
  bar.dataset.date = day.date;
  bar.dataset.status = day.status;
  bar.dataset.uptime = day.uptime === null ? "" : day.uptime;
  return bar;
}

function renderComponent(c) {
  const wrap = document.createElement("div");
  wrap.className = "component";

  const top = document.createElement("div");
  top.className = "component-top";

  const nameCol = document.createElement("div");
  nameCol.className = "component-name";
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = c.name;
  if (c.internal) {
    const badge = document.createElement("span");
    badge.className = "badge-internal";
    badge.textContent = "internal";
    name.appendChild(badge);
  }
  const desc = document.createElement("div");
  desc.className = "desc";
  desc.textContent = c.description;
  nameCol.appendChild(name);
  nameCol.appendChild(desc);

  const statusEl = document.createElement("div");
  statusEl.className = "component-status s-" + c.status;
  const dot = document.createElement("span");
  dot.className = "status-dot dot-" + c.status;
  const label = document.createElement("span");
  let labelText = c.status_label || STATUS_LABELS[c.status];
  if (c.status === "operational" && c.latency_ms != null) {
    labelText += ` · ${Math.round(c.latency_ms)}ms`;
  }
  label.textContent = labelText;
  statusEl.appendChild(dot);
  statusEl.appendChild(label);

  top.appendChild(nameCol);
  top.appendChild(statusEl);

  const uptime = document.createElement("div");
  uptime.className = "uptime";
  const bars = document.createElement("div");
  bars.className = "uptime-bars";
  c.history.forEach((day) => bars.appendChild(uptimeBar(day)));

  const lg = document.createElement("div");
  lg.className = "uptime-legend";
  const left = document.createElement("span");
  left.textContent = "90 days ago";
  const line = document.createElement("span");
  line.className = "line";
  const mid = document.createElement("span");
  mid.className = "pct";
  mid.textContent = c.uptime_90d != null ? `${c.uptime_90d}% uptime` : "no data yet";
  const line2 = document.createElement("span");
  line2.className = "line";
  const right = document.createElement("span");
  right.textContent = "Today";
  lg.append(left, line, mid, line2, right);

  uptime.appendChild(bars);
  uptime.appendChild(lg);

  wrap.appendChild(top);
  wrap.appendChild(uptime);
  return wrap;
}

function renderGroups(components) {
  const container = document.getElementById("groups");
  container.innerHTML = "";

  const order = [];
  const byGroup = {};
  components.forEach((c) => {
    if (!byGroup[c.group]) {
      byGroup[c.group] = [];
      order.push(c.group);
    }
    byGroup[c.group].push(c);
  });

  order.forEach((groupName) => {
    const group = document.createElement("div");
    group.className = "group";
    const label = document.createElement("div");
    label.className = "group-label";
    label.textContent = groupName;
    group.appendChild(label);
    byGroup[groupName].forEach((c) => group.appendChild(renderComponent(c)));
    container.appendChild(group);
  });

  attachTooltips(container);
}

function renderLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = "";
  LEGEND.forEach((item) => {
    const wrap = document.createElement("div");
    wrap.className = "legend-item";
    const sw = document.createElement("span");
    sw.className = "legend-swatch dot-" + item.key;
    const txt = document.createElement("span");
    txt.textContent = item.label;
    wrap.append(sw, txt);
    el.appendChild(wrap);
  });
}

function renderActiveIncidents(incidents) {
  const el = document.getElementById("active-incidents");
  if (!incidents || !incidents.length) {
    el.hidden = true;
    el.innerHTML = "";
    return;
  }
  el.hidden = false;
  el.innerHTML = "";
  incidents.forEach((inc) => {
    const box = document.createElement("div");
    box.className = "active-incident sev-" + (inc.impact || "major_outage");
    const h = document.createElement("h3");
    h.textContent = inc.title;
    const p = document.createElement("p");
    const latest = (inc.updates && inc.updates[0]) || {};
    p.textContent = latest.body || inc.summary || "";
    box.append(h, p);
    el.appendChild(box);
  });
}

function renderIncidents(incidents) {
  const el = document.getElementById("incidents");
  el.innerHTML = "";

  // Group incidents by day (from their latest update or created_at date).
  const days = new Map();
  const today = new Date();
  for (let i = 0; i < 14; i++) {
    const d = new Date(today);
    d.setUTCDate(d.getUTCDate() - i);
    const key = d.toISOString().slice(0, 10);
    days.set(key, []);
  }

  (incidents || []).forEach((inc) => {
    const key = (inc.date || (inc.updates && inc.updates[inc.updates.length - 1]?.at) || "")
      .slice(0, 10);
    if (days.has(key)) days.get(key).push(inc);
  });

  let rendered = 0;
  days.forEach((list, key) => {
    const day = document.createElement("div");
    day.className = "incident-day";
    const date = document.createElement("div");
    date.className = "incident-date";
    date.textContent = fmtDate(key);
    day.appendChild(date);

    if (!list.length) {
      const none = document.createElement("div");
      none.className = "incident-none";
      none.textContent = "No incidents reported.";
      day.appendChild(none);
    } else {
      list.forEach((inc) => day.appendChild(renderIncident(inc)));
    }
    el.appendChild(day);
    rendered++;
  });

  if (!rendered) {
    el.innerHTML = '<div class="incident-none">No incidents reported.</div>';
  }
}

function renderIncident(inc) {
  const box = document.createElement("div");
  const sev = inc.status === "resolved" ? "resolved" : inc.impact || "degraded";
  box.className = "incident sev-" + sev;
  const title = document.createElement("div");
  title.className = "incident-title";
  title.textContent = inc.title;
  box.appendChild(title);

  (inc.updates || []).forEach((u) => {
    const upd = document.createElement("div");
    upd.className = "incident-update";
    const lbl = document.createElement("span");
    lbl.className = "u-label";
    lbl.textContent = (u.label || "Update") + " — ";
    const body = document.createElement("span");
    body.textContent = u.body;
    const time = document.createElement("span");
    time.className = "u-time";
    time.textContent = u.at
      ? new Date(u.at).toLocaleString(undefined, {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          timeZoneName: "short",
        })
      : "";
    upd.append(lbl, body, time);
    box.appendChild(upd);
  });
  return box;
}

/* ---------- tooltips for uptime bars ---------- */
function attachTooltips(container) {
  container.querySelectorAll(".uptime-bar").forEach((bar) => {
    bar.addEventListener("mouseenter", (e) => {
      const status = bar.dataset.status;
      const uptime = bar.dataset.uptime;
      let statusText =
        status === "no_data"
          ? "No data"
          : STATUS_LABELS[status] || status;
      if (status === "operational" && uptime === "1") {
        statusText = "No downtime recorded";
      }
      tooltip.innerHTML = `<div class="tt-date">${fmtDate(bar.dataset.date)}</div>
        <div class="tt-status s-${status}">${statusText}</div>` +
        (uptime && status !== "no_data"
          ? `<div class="tt-pct">${(parseFloat(uptime) * 100).toFixed(2)}% uptime</div>`
          : "");
      tooltip.hidden = false;
      positionTooltip(e);
    });
    bar.addEventListener("mousemove", positionTooltip);
    bar.addEventListener("mouseleave", () => {
      tooltip.hidden = true;
    });
  });
}

function positionTooltip(e) {
  const rect = e.target.getBoundingClientRect();
  tooltip.style.left = rect.left + rect.width / 2 + "px";
  tooltip.style.top = rect.top - 10 + "px";
}

/* ---------- data ---------- */
async function refresh() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) throw new Error("status " + res.status);
    const data = await res.json();

    pollInterval = (data.poll_interval || 60) * 1000;
    document.getElementById("poll-interval").textContent = data.poll_interval || 60;

    renderOverall(data.overall, data.active_incidents);
    renderActiveIncidents(data.active_incidents);
    renderGroups(data.components);
    renderIncidents(data.incidents);

    document.getElementById("last-checked").textContent =
      "Updated " + fmtRelative(data.last_checked_at);
    document.getElementById("live-dot").style.background = "var(--op)";
  } catch (err) {
    document.getElementById("last-checked").textContent = "Reconnecting…";
    document.getElementById("live-dot").style.background = "var(--degraded)";
  }
}

function loop() {
  clearTimeout(timer);
  refresh().finally(() => {
    timer = setTimeout(loop, pollInterval);
  });
}

initTheme();
renderLegend();
loop();

// Refresh "Updated Xs ago" label every 10s without refetching.
setInterval(() => {
  const el = document.getElementById("last-checked");
  if (el && el.textContent.startsWith("Updated")) {
    // Re-fetch is handled by loop(); this keeps the label from going stale
    // between polls is optional — no-op to avoid double state.
  }
}, 10000);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) loop();
});
