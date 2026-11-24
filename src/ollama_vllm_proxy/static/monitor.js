// Live monitoring page: polls /admin/api/live and draws tiles, inline-SVG charts and tables.
// No charting library; every value is written with textContent, never as HTML.
(function () {
  "use strict";
  const { api, el, boot } = window.OVP;
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";

  // ---------- formatting ----------
  const dash = "–";
  const fmtInt = (v) => v == null ? dash : Math.round(v).toLocaleString();
  const fmtNum = (v, d = 1) => v == null ? dash
    : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtPct = (v, d = 0) => v == null ? dash : (v * 100).toFixed(d) + "%";
  function fmtMs(v) {
    if (v == null) return dash;
    if (v < 1000) return Math.round(v) + " ms";
    if (v < 10000) return (v / 1000).toFixed(2) + " s";
    if (v < 120000) return (v / 1000).toFixed(1) + " s";
    return Math.floor(v / 60000) + "m " + Math.round((v % 60000) / 1000) + "s";
  }
  const pad2 = (n) => String(n).padStart(2, "0");
  // kind: "hms" 14:03:09, "hm" 14:03, "dhm" 11-24 14:03, "d" 11-24, "full" 11-24 14:03:09
  function fmtTime(t, kind) {
    const d = new Date(t * 1000);
    const day = pad2(d.getMonth() + 1) + "-" + pad2(d.getDate());
    const hm = pad2(d.getHours()) + ":" + pad2(d.getMinutes());
    if (kind === "d") return day;
    if (kind === "hm") return hm;
    if (kind === "dhm") return day + " " + hm;
    const hms = hm + ":" + pad2(d.getSeconds());
    return kind === "full" ? day + " " + hms : hms;
  }
  function fmtWindow(s) {
    if (s < 3600) return Math.round(s / 60) + " min";
    if (s < 86400) return Math.round(s / 3600) + " h";
    return Math.round(s / 86400) + " d";
  }

  // ---------- DOM helpers ----------
  function svgEl(tag, attrs, parent, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }
  const keyLine = (color) => el("span", { class: "key-line", style: "background:var(" + color + ")" });
  const keyDot = (color) => el("span", { class: "key-dot", style: "background:var(" + color + ")" });
  const textTd = (v) => el("td", { class: "text", text: v == null || v === "" ? dash : String(v) });

  function table(host, headers, rows, emptyText) {
    if (!rows.length) { host.replaceChildren(el("div", { class: "none", text: emptyText })); return; }
    const thead = el("thead", {}, el("tr", {}, headers.map(([label, cls]) => el("th", cls ? { class: cls, text: label } : { text: label }))));
    const tbody = el("tbody", {}, rows.map((cells) => el("tr", {}, cells.map((c) => (c && c.nodeType) ? c : el("td", { text: c })))));
    host.replaceChildren(el("table", {}, [thead, tbody]));
  }
  function outcomeCell(outcome) {
    const [cls, icon] = outcome === "ok" ? ["good", "●"] : outcome === "client_cancelled" ? ["warning", "▲"] : ["critical", "✕"];
    return el("td", { class: "text" }, [el("span", { class: "icon " + cls, "aria-hidden": "true", text: icon }), outcome || dash]);
  }

  // ---------- tooltip ----------
  const tip = el("div", { class: "tooltip hidden", role: "presentation" });
  function showTip(head, rows, cx, cy) {
    tip.replaceChildren(el("div", { class: "tt-head", text: head }));
    for (const r of rows) {
      tip.append(el("div", { class: "tt-row" }, [r.color ? keyLine(r.color) : null,
        el("strong", { text: r.value }), el("span", { class: "lbl", text: r.label })]));
    }
    tip.classList.remove("hidden");
    const w = tip.offsetWidth, ht = tip.offsetHeight;
    let x = cx + 14, y = cy + 14;
    if (x + w > window.innerWidth - 8) x = cx - w - 14;
    if (y + ht > window.innerHeight - 8) y = cy - ht - 14;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }
  function hideTip() { tip.classList.add("hidden"); }

  // ---------- scales ----------
  function niceTicks(max, count = 4) {
    if (!(max > 0)) max = 1;
    const raw = max / count, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((f) => f * mag).find((s) => s >= raw);
    const top = Math.ceil(max / step - 1e-9) * step;
    const ticks = [];
    for (let v = 0; v <= top + step / 2; v += step) ticks.push(+v.toFixed(10));
    return { top, ticks };
  }
  const STEPS = [5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400, 172800, 604800];
  function timeTicks(t0, t1, width) {
    const span = t1 - t0;
    const multiDay = span > 86400;
    const count = Math.max(2, Math.floor(width / (multiDay ? 110 : 80)));
    const step = STEPS.find((s) => span / s <= count) || 1209600;
    const off = -new Date((t0 + t1) * 500).getTimezoneOffset() * 60; // ticks on local-time boundaries
    const ticks = [];
    for (let t = Math.ceil((t0 + off) / step) * step - off; t <= t1; t += step) ticks.push(t);
    return { ticks, kind: step < 60 ? "hms" : !multiDay ? "hm" : step < 86400 ? "dhm" : "d" };
  }
  // Lines are broken where samples are missing for much longer than usual (proxy restart, tunnel down).
  function gapOf(times) {
    if (times.length < 3) return Infinity;
    const d = [];
    for (let i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
    d.sort((a, b) => a - b);
    return Math.max(d[d.length >> 1] * 5, 1);
  }

  // ---------- line chart ----------
  class LineChart {
    constructor(card, opts) {
      this.opts = opts;
      card.append(el("div", { class: "card-head" }, [el("h2", { text: opts.title }),
        opts.sub ? el("span", { class: "sub", text: opts.sub }) : null]));
      if (opts.series.length > 1) {
        card.append(el("div", { class: "legend" }, opts.series.map((s) => el("span", {}, [keyLine(s.color), s.label]))));
      }
      this.plot = el("div", { class: "plot" });
      this.tableHost = el("div", { class: "table-wrap" });
      this.details = el("details", { ontoggle: () => this.renderTable() }, [el("summary", { text: "Show data" }), this.tableHost]);
      card.append(this.plot, this.details);
      this.pointerX = null;
      this.keyIdx = null;
      this.focused = false;
      this.times = [];
      this.values = {};
      this.buildSvg();
      new ResizeObserver(() => this.render()).observe(this.plot);
    }
    // The <svg> element is kept across updates so keyboard focus and the crosshair survive a poll.
    buildSvg() {
      const svg = this.svg = svgEl("svg", { role: "img", tabindex: "0",
        "aria-label": this.opts.title + ". Use the left and right arrow keys to read values." });
      this.static = svgEl("g", {}, svg);
      this.hover = svgEl("g", {}, svg);
      this.hit = svgEl("rect", { fill: "transparent" }, svg);
      this.hit.addEventListener("pointermove", (ev) => {
        const r = svg.getBoundingClientRect();
        this.pointerX = (ev.clientX - r.left) * (this.geom.W / r.width);
        this.keyIdx = null;
        this.drawHover(ev.clientX, ev.clientY);
      });
      this.hit.addEventListener("pointerleave", () => { this.pointerX = null; this.clearHover(); });
      svg.addEventListener("keydown", (ev) => {
        const n = this.times.length;
        if (!n) return;
        if (ev.key === "ArrowLeft" || ev.key === "ArrowRight" || ev.key === "Home" || ev.key === "End") {
          let cur = this.keyIdx == null ? n - 1 : this.keyIdx + (ev.key === "ArrowRight" ? 1 : -1);
          if (ev.key === "Home") cur = 0;
          if (ev.key === "End") cur = n - 1;
          this.keyIdx = Math.max(0, Math.min(n - 1, cur));
          this.pointerX = null;
          this.drawKeyHover();
          ev.preventDefault();
        } else if (ev.key === "Escape") { this.keyIdx = null; this.clearHover(); }
      });
      svg.addEventListener("blur", () => { if (this.pointerX == null) { this.keyIdx = null; this.clearHover(); } });
      this.plot.replaceChildren(svg);
    }
    clearHover() { this.hover.replaceChildren(); hideTip(); }
    update(times, valuesByKey, domain) {
      // Keyboard position follows the sample, not the index, when new samples shift the array.
      const keyT = this.keyIdx != null && this.times[this.keyIdx] != null ? this.times[this.keyIdx] : null;
      this.times = times; this.values = valuesByKey; this.domain = domain;
      if (keyT != null) this.keyIdx = times.length ? this.nearestTime(keyT) : null;
      this.render();
      this.renderTable();
    }
    render() {
      if (!this.domain) return;
      const { series, yMax, minTop = 1, yFmt = fmtNum } = this.opts;
      const W = Math.max(280, this.plot.clientWidth || 600), H = this.opts.height || 190;
      // Direct labels at the line ends need room; on a phone the legend above the plot does the job.
      const multi = series.length > 1 && W >= 480;
      const m = { l: 52, r: multi ? 104 : 14, t: 10, b: 24 };
      const [t0, t1] = this.domain;
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const x = (t) => m.l + (t - t0) / (t1 - t0) * pw;
      let max = 0;
      for (const s of series) for (const v of this.values[s.key] || []) if (v != null && v > max) max = v;
      const nt = niceTicks(yMax != null ? yMax : Math.max(max, minTop));
      const y = (v) => m.t + ph - (Math.min(v, nt.top) / nt.top) * ph;
      this.geom = { x, y, m, W, H, t0, t1, pw };
      this.svg.setAttribute("viewBox", "0 0 " + W + " " + H);
      const g = this.static;
      g.replaceChildren();
      for (const tv of nt.ticks) {
        svgEl("line", { class: tv === 0 ? "baseline" : "gridline", x1: m.l, x2: W - m.r, y1: y(tv), y2: y(tv) }, g);
        svgEl("text", { class: "tick", x: m.l - 8, y: y(tv) + 4, "text-anchor": "end" }, g, yFmt(tv, true));
      }
      const tt = timeTicks(t0, t1, pw);
      for (const t of tt.ticks) svgEl("text", { class: "tick", x: x(t), y: H - 6, "text-anchor": "middle" }, g, fmtTime(t, tt.kind));
      const n = this.times.length;
      const gap = gapOf(this.times);
      const ends = [];
      let any = false;
      for (const s of series) {
        const vals = this.values[s.key] || [];
        let d = "", run = 0, last = null, prevT = null, lone = [];
        for (let i = 0; i < n; i++) {
          const v = vals[i];
          if (v == null || this.times[i] < t0) { if (run === 1) lone.push(last); run = 0; continue; }
          if (run && this.times[i] - prevT > gap) { if (run === 1) lone.push(last); run = 0; }
          d += (run ? "L" : "M") + x(this.times[i]).toFixed(1) + "," + y(v).toFixed(1);
          run++; last = i; prevT = this.times[i];
        }
        if (run === 1) lone.push(last);
        if (d) {
          any = true;
          svgEl("path", { d, fill: "none", stroke: "var(" + s.color + ")", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, g);
        }
        for (const i of lone) svgEl("circle", { cx: x(this.times[i]), cy: y(vals[i]), r: 2, fill: "var(" + s.color + ")" }, g);
        if (multi && last != null) ends.push({ s, y: y(vals[last]) });
      }
      if (!any) {
        svgEl("text", { class: "empty", x: m.l + pw / 2, y: m.t + ph / 2, "text-anchor": "middle" }, g, this.opts.emptyText || "No samples in this window");
      }
      // direct labels at the line ends, pushed apart when they would overlap
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
      for (const e of ends) svgEl("text", { class: "dlabel", x: W - m.r + 8, y: e.y + 4 }, g, e.s.label);
      for (const [k, v] of Object.entries({ x: m.l, y: m.t, width: pw, height: ph })) this.hit.setAttribute(k, v);
      if (this.keyIdx != null) this.drawKeyHover();
      else if (this.pointerX != null && this.lastClient) this.drawHover(...this.lastClient);
    }
    nearestTime(t) {
      let best = -1, bd = Infinity;
      for (let i = 0; i < this.times.length; i++) { const d = Math.abs(this.times[i] - t); if (d < bd) { bd = d; best = i; } }
      return best;
    }
    drawKeyHover() {
      const { x, m, W } = this.geom;
      const r = this.svg.getBoundingClientRect();
      this.drawHover(r.left + x(this.times[this.keyIdx]) * r.width / W, r.top + m.t);
    }
    drawHover(cx, cy) {
      this.lastClient = [cx, cy];
      if (!this.times.length) return;
      const { x, y, m, H, t0, t1, pw } = this.geom;
      const i = this.keyIdx != null ? this.keyIdx : this.nearestTime(t0 + (this.pointerX - m.l) / pw * (t1 - t0));
      if (i < 0 || this.times[i] < t0) { this.clearHover(); return; }
      const xi = x(this.times[i]);
      this.hover.replaceChildren();
      svgEl("line", { class: "crosshair", x1: xi, x2: xi, y1: m.t, y2: H - m.b }, this.hover);
      const rows = [];
      for (const s of this.opts.series) {
        const v = (this.values[s.key] || [])[i];
        if (v != null) svgEl("circle", { cx: xi, cy: y(v), r: 4, fill: "var(" + s.color + ")", stroke: "var(--surface-1)", "stroke-width": 2 }, this.hover);
        rows.push({ color: s.color, value: v == null ? dash : (this.opts.yFmt || fmtNum)(v), label: s.label });
      }
      showTip(fmtTime(this.times[i], t1 - t0 > 86400 ? "full" : "hms"), rows, cx, cy);
    }
    renderTable() {
      if (!this.details.open) return;
      const { series, yFmt = fmtNum } = this.opts;
      const kind = this.domain && this.domain[1] - this.domain[0] > 86400 ? "full" : "hms";
      const rows = [];
      for (let i = this.times.length - 1; i >= Math.max(0, this.times.length - 40); i--) {
        rows.push([fmtTime(this.times[i], kind)].concat(series.map((s) => {
          const v = (this.values[s.key] || [])[i];
          return v == null ? dash : yFmt(v);
        })));
      }
      table(this.tableHost, [["Time"]].concat(series.map((s) => [s.label])), rows, "No samples in this window.");
    }
  }

  // ---------- scatter (time to first content per request, log scale) ----------
  const COLORS = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6"];
  const OTHER = "--text-muted";
  // One colour per key; beyond six keys the least active ones share a muted colour.
  function keyColors(keys) {
    const ranked = keys.slice().sort((a, b) => b.n - a.n || (a.key < b.key ? -1 : 1));
    const limit = ranked.length <= COLORS.length ? COLORS.length : COLORS.length - 1;
    const out = new Map();
    ranked.forEach((k, i) => out.set(k.key, i < limit ? COLORS[i] : OTHER));
    return out;
  }
  class Scatter {
    constructor(card) {
      this.sub = el("span", { class: "sub" });
      card.append(el("div", { class: "card-head" }, [el("h2", { text: "Time to first content per request" }), this.sub]));
      this.legend = el("div", { class: "legend" });
      this.plot = el("div", { class: "plot" });
      card.append(this.legend, this.plot);
      this.rows = [];
      this.colors = new Map();
      new ResizeObserver(() => this.render()).observe(this.plot);
    }
    update(rows, keys, domain, total) {
      this.rows = rows; this.domain = domain;
      this.colors = keyColors(keys);
      const items = [];
      let other = false;
      for (const [name, color] of this.colors) { if (color === OTHER) other = true; else items.push(el("span", {}, [keyDot(color), name])); }
      if (other) items.push(el("span", {}, [keyDot(OTHER), "Other keys"]));
      items.push(el("span", {}, [el("span", { class: "key-hollow" }), "Not completed OK"]));
      this.legend.replaceChildren(...items);
      this.sub.textContent = "proxy clock, log scale; one dot per finished request, coloured by API key"
        + (total > rows.length ? " (the newest " + fmtInt(rows.length) + " of " + fmtInt(total) + ")" : "");
      this.render();
    }
    render() {
      if (!this.domain) return;
      const W = Math.max(280, this.plot.clientWidth || 600), H = 230;
      const m = { l: 60, r: 14, t: 10, b: 24 };
      const [t0, t1] = this.domain;
      const pw = W - m.l - m.r, ph = H - m.t - m.b;
      const pts = this.rows.filter((r) => r.first_payload_ms != null && r.first_payload_ms > 0 && r.t_arrival_wall >= t0);
      let lo = 100, hi = 10000;
      for (const p of pts) { lo = Math.min(lo, p.first_payload_ms); hi = Math.max(hi, p.first_payload_ms); }
      const l0 = Math.floor(Math.log10(lo)), l1 = Math.ceil(Math.log10(hi));
      const x = (t) => m.l + (t - t0) / (t1 - t0) * pw;
      const y = (v) => m.t + ph - (Math.log10(v) - l0) / (l1 - l0) * ph;
      const svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
        "aria-label": "Time to first content per request; the same values are in the requests table below" });
      for (let e = l0; e <= l1; e++) {
        const v = Math.pow(10, e);
        svgEl("line", { class: e === l0 ? "baseline" : "gridline", x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, svg);
        svgEl("text", { class: "tick", x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, svg, fmtMs(v));
      }
      const tt = timeTicks(t0, t1, pw);
      for (const t of tt.ticks) svgEl("text", { class: "tick", x: x(t), y: H - 6, "text-anchor": "middle" }, svg, fmtTime(t, tt.kind));
      if (!pts.length) svgEl("text", { class: "empty", x: m.l + pw / 2, y: m.t + ph / 2, "text-anchor": "middle" }, svg, "No finished requests in this window");
      const placed = [];
      for (const p of pts) {
        const color = this.colors.get(p.actor_id || "(none)") || OTHER;
        const cx = x(p.t_arrival_wall), cy = y(p.first_payload_ms);
        svgEl("circle", p.outcome === "ok"
          ? { cx, cy, r: 4, fill: "var(" + color + ")", stroke: "var(--surface-1)", "stroke-width": 2 }
          : { cx, cy, r: 3.5, fill: "var(--surface-1)", stroke: "var(" + color + ")", "stroke-width": 2 }, svg);
        placed.push({ p, cx, cy, color });
      }
      const hl = svgEl("g", {}, svg);
      svg.addEventListener("pointermove", (ev) => {
        const r = svg.getBoundingClientRect(), k = W / r.width;
        const px = (ev.clientX - r.left) * k, py = (ev.clientY - r.top) * k;
        let best = null, bd = 24 * k;
        for (const q of placed) { const d = Math.hypot(q.cx - px, q.cy - py); if (d < bd) { bd = d; best = q; } }
        hl.replaceChildren();
        if (!best) { hideTip(); return; }
        svgEl("circle", { cx: best.cx, cy: best.cy, r: 7, fill: "none", stroke: "var(--text-primary)", "stroke-width": 1.5 }, hl);
        const p = best.p;
        showTip(fmtTime(p.t_arrival_wall, t1 - t0 > 86400 ? "full" : "hms") + " · " + (p.actor_id || dash)
          + (p.session_id ? " · " + p.session_id : ""), [
          { color: best.color, value: fmtMs(p.first_payload_ms), label: "first content" },
          { value: fmtMs(p.first_text_ms), label: "first text" },
          { value: fmtMs(p.first_tool_ms), label: "first tool call" },
          { value: fmtMs(p.end_ms), label: "end" },
          { value: fmtInt(p.prompt_tokens) + " / " + fmtInt(p.cached_tokens), label: "prompt / cached tok" },
          { value: fmtInt(p.completion_tokens), label: "output tok" },
          { value: fmtNum(p.decode_tps), label: "decode tok/s" },
          { value: p.outcome || dash, label: "outcome" },
        ], ev.clientX, ev.clientY);
      });
      svg.addEventListener("pointerleave", () => { hl.replaceChildren(); hideTip(); });
      this.plot.replaceChildren(svg);
    }
  }

  // ---------- tiles ----------
  const TILES = [
    ["open", "Open requests", ""],
    ["running", "Engine running", ""],
    ["waiting", "Engine waiting", ""],
    ["kv", "KV cache used", "%"],
    ["output", "Output", "tok/s"],
    ["prompt", "Prompt", "tok/s"],
    ["completed", "Completed", ""],
    ["errors", "Errors", ""],
    ["gpu_util", "GPU utilization", "%"],
    ["gpu_mem", "GPU memory used", "GiB"],
  ];
  const tileEls = {};
  function buildTiles() {
    for (const [key, label, unit] of TILES) {
      const value = el("span", { text: dash });
      const flag = el("span", { class: "flag" }, [el("span", { class: "icon", "aria-hidden": "true", text: "✕ " }), "attention"]);
      const sub = el("div", { class: "sub", text: " " });
      const tile = el("div", { class: "tile" }, [el("div", { class: "label", text: label }),
        el("div", { class: "value" }, [value, unit ? el("span", { class: "unit", text: unit }) : null, flag]), sub]);
      $("tiles").append(tile);
      tileEls[key] = { tile, value, sub };
    }
  }
  function setTile(key, value, sub, alert) {
    tileEls[key].value.textContent = value;
    tileEls[key].sub.textContent = sub || " ";
    tileEls[key].tile.classList.toggle("alert", !!alert);
  }

  // ---------- state & charts ----------
  const state = { windowS: 300, paused: false, timer: null, lastOk: 0, snap: null, busy: false };
  try { const w = +localStorage.getItem("ovp-monitor-window"); if (w > 0) state.windowS = w; } catch (e) { /* default */ }

  let charts = null;
  function buildCharts() {
    const rate = (v, axis) => axis && Number.isInteger(v) ? fmtInt(v) : fmtNum(v);
    charts = {
      concurrency: new LineChart($("c-concurrency"), {
        title: "Concurrency", sub: "requests, all keys",
        series: [
          { key: "running", label: "Running", color: "--series-1" },
          { key: "waiting", label: "Waiting", color: "--series-2" },
          { key: "open", label: "Open at proxy", color: "--series-3" },
        ],
        yFmt: (v) => fmtInt(v), minTop: 4,
      }),
      kv: new LineChart($("c-kv"), {
        title: "KV cache used", sub: "percent of GPU cache blocks",
        series: [{ key: "kv", label: "KV cache used", color: "--series-1" }],
        yMax: 100, yFmt: (v, axis) => axis ? Math.round(v) + "%" : fmtNum(v) + "%",
      }),
      output: new LineChart($("c-output"), {
        title: "Output tokens/s", sub: "engine-wide, from counter deltas",
        series: [{ key: "output_tps", label: "Output tok/s", color: "--series-1" }], yFmt: rate, minTop: 10,
      }),
      prompt: new LineChart($("c-prompt"), {
        title: "Prompt tokens/s", sub: "engine-wide, includes cached prompt tokens",
        series: [{ key: "prompt_tps", label: "Prompt tok/s", color: "--series-1" }], yFmt: rate, minTop: 10,
      }),
      prefix: new LineChart($("c-prefix"), {
        title: "Prefix cache hit rate", sub: "share of looked-up prompt tokens served from cache",
        series: [{ key: "prefix_hit", label: "Hit rate", color: "--series-1" }],
        yMax: 100, yFmt: (v) => Math.round(v) + "%",
      }),
      first: new Scatter($("c-first")),
    };
  }

  // ---------- GPU charts (one line per GPU, rebuilt when the set of GPUs changes) ----------
  let gpuCharts = null, gpuKey = "";
  function buildGpuCharts(ids) {
    const host = $("gpu-charts");
    host.replaceChildren();
    const card = () => { const c = el("section", { class: "card" }); host.append(c); return c; };
    const per = (field, scale) => ids.map((g, i) => ({ key: "g" + g + "_" + field, label: "GPU " + g, color: COLORS[i % 6], scale }));
    const pcie = [];
    ids.forEach((g, i) => {
      pcie.push({ key: "g" + g + "_rx", label: "GPU " + g + " RX", color: COLORS[(2 * i) % 6] });
      pcie.push({ key: "g" + g + "_tx", label: "GPU " + g + " TX", color: COLORS[(2 * i + 1) % 6] });
    });
    const rate = (v, axis) => axis && Number.isInteger(v) ? fmtInt(v) : fmtNum(v);
    gpuCharts = {
      util: new LineChart(card(), { title: "GPU utilization", sub: "percent of time a kernel was running, per GPU", series: per("util"), yMax: 100, yFmt: (v) => Math.round(v) + "%" }),
      mem: new LineChart(card(), { title: "GPU memory used", sub: "GiB per GPU (vLLM reserves most of it at startup)", series: per("mem_used", 1 / 1024), yFmt: rate }),
      power: new LineChart(card(), { title: "GPU power", sub: "watts per GPU; axis top is the power limit", series: per("power"), minTop: 50, yFmt: (v) => Math.round(v) + " W" }),
      temp: new LineChart(card(), { title: "GPU temperature", sub: "°C per GPU", series: per("temp"), minTop: 90, yFmt: (v) => Math.round(v) + "°" }),
      clock: new LineChart(card(), { title: "SM clock", sub: "MHz per GPU; drops under power or thermal limits", series: per("sm_clock"), minTop: 500, yFmt: (v) => fmtInt(v) }),
      pcie: new LineChart(card(), { title: "PCIe traffic", sub: "MB/s per GPU; with tensor parallelism this includes GPU-to-GPU traffic", series: pcie, minTop: 10, yFmt: (v, axis) => axis ? fmtInt(v) : fmtNum(v, 0) }),
    };
  }
  function updateGpus(s, times, domain) {
    const gpus = s.gpus || [];
    $("gpu-charts").classList.toggle("hidden", !gpus.length);
    $("gpu-summary").classList.toggle("hidden", !gpus.length);
    if (!gpus.length) return;
    const ids = gpus.map((g) => g.gpu);
    if (ids.join(",") !== gpuKey) { gpuKey = ids.join(","); buildGpuCharts(ids); }
    gpuCharts.mem.opts.yMax = Math.max(0, ...gpus.map((g) => (g.mem_total_mib || 0) / 1024)) || null;
    gpuCharts.power.opts.yMax = Math.max(0, ...gpus.map((g) => g.power_limit || 0)) || null;
    for (const chart of Object.values(gpuCharts)) {
      const values = {};
      for (const sr of chart.opts.series) {
        values[sr.key] = s.series.map((p) => p[sr.key] == null ? null : p[sr.key] * (sr.scale || 1));
      }
      chart.update(times, values, domain);
    }
    const pctv = (v) => v == null ? dash : fmtInt(v) + "%";
    $("gpu-sub").textContent = "over the last " + fmtWindow(s.window_s);
    table($("t-gpus"),
      [["GPU"], ["Util mean"], ["Util p90"], ["Memory peak"], ["Power mean / max"], ["Temp max"], ["SM clock min"],
        ["PCIe RX mean / max MB/s"], ["PCIe TX mean / max MB/s"], ["Samples"]],
      gpus.map((g) => ["GPU " + g.gpu, pctv(g.util_mean), pctv(g.util_p90),
        g.mem_used_max_mib == null ? dash : fmtNum(g.mem_used_max_mib / 1024, 1) + " / " + fmtNum((g.mem_total_mib || 0) / 1024, 1) + " GiB",
        g.power_mean == null ? dash : fmtInt(g.power_mean) + " / " + fmtInt(g.power_max) + " W",
        g.temp_max == null ? dash : fmtInt(g.temp_max) + "°",
        g.sm_clock_min == null ? dash : fmtInt(g.sm_clock_min) + " MHz",
        g.rx_mean == null ? dash : fmtInt(g.rx_mean) + " / " + fmtInt(g.rx_max),
        g.tx_mean == null ? dash : fmtInt(g.tx_mean) + " / " + fmtInt(g.tx_max),
        fmtInt(g.samples)]), "");
  }

  // ---------- hardware panel ----------
  const LINK_TEXT = {
    SYS: "PCIe across CPU sockets (crosses the NUMA interconnect)", NODE: "PCIe across host bridges, same NUMA node",
    PHB: "PCIe through the CPU host bridge", PXB: "PCIe through several PCIe switches", PIX: "PCIe through one PCIe switch",
  };
  const linkText = (l) => /^NV\d+/.test(l) ? "NVLink (" + l + ")" : l + " · " + (LINK_TEXT[l] || "see topology");
  function facts(pairs) {
    const kids = [];
    for (const [k, v] of pairs) if (v != null && v !== "") kids.push(el("dt", { text: k }), el("dd", { text: String(v) }));
    return el("dl", { class: "facts" }, kids);
  }
  const digits = (v) => /^\d+$/.test(String(v == null ? "" : v)) ? +v : null;
  let hwKey = null;
  function renderHardware(s) {
    const hw = s.hardware || {}, b = s.backend || {};
    const inv = hw.gpu_inventory, cache = (hw["vllm:cache_config_info"] || [])[0], ver = (hw.vllm_version || {}).version;
    const key = JSON.stringify([inv, cache, ver, s.gpu_ok, b.gpu_helper, b.state, b.served_model, b.context_length]);
    if (key === hwKey) return; // static data: only redraw when it changes
    hwKey = key;
    $("hw-sub").textContent = !b.gpu_helper ? "no GPU helper connected"
      : s.gpu_ok ? "GPU helper connected" : s.gpu_ok === false ? "GPU helper not reachable" : "";
    const gpus = inv && inv.gpus && inv.gpus.length ? inv.gpus : null;
    if (!gpus && !cache && !ver) {
      $("hw-body").replaceChildren(el("div", { class: "none", text: b.state === "ready"
        ? "No vLLM configuration read yet; it appears within 30 seconds of the backend becoming reachable."
        : "Nothing to show until a backend is running. GPU details come from the helper that Deploy starts on the pod." }));
      return;
    }
    const body = [], left = [];
    if (gpus) {
      const g0 = gpus[0];
      const links = (inv.topology && inv.topology.links) || [];
      left.push(facts([
        ["GPUs", gpus.length + " × " + g0.name],
        ["Driver / CUDA", (g0.driver_version || dash) + " / " + (inv.cuda_version || dash)],
        ["GPU-to-GPU", links.length ? links.map((l) => l.a + "↔" + l.b + ": " + linkText(l.link)).join("; ") : (gpus.length === 1 ? "single GPU" : null)],
        ["NVLink", inv.nvlink_active ? "active" : "inactive"],
        ["Host CPU", inv.host ? (inv.host.cpu_model || dash) + " · " + fmtInt(inv.host.logical_cpus) + " logical CPUs" : null],
        ["Host RAM", inv.host && inv.host.memory_gib ? fmtNum(inv.host.memory_gib, 0) + " GiB" : null],
      ]));
    } else {
      left.push(el("div", { class: "none", text: b.gpu_helper
        ? "No GPU inventory yet; it appears within 30 seconds of the helper starting."
        : "No GPU helper is connected, so GPU details and GPU charts are not shown." }));
    }
    const c = cache || {};
    const block = digits(c.block_size), blocks = digits(c.num_gpu_blocks);
    const kvTokens = digits(c.kv_cache_size_tokens) || (block && blocks ? block * blocks : null);
    const conc = +c.kv_cache_max_concurrency || (kvTokens && b.context_length ? kvTokens / b.context_length : null);
    const right = facts([
      ["vLLM", ver],
      ["Served model", b.served_model],
      ["Context length", b.context_length ? fmtInt(b.context_length) + " tokens" : null],
      ["KV cache size", kvTokens ? fmtInt(kvTokens) + " tokens" : null],
      ["Max concurrency", conc && b.context_length ? fmtNum(conc, 2) + "× requests at " + fmtInt(b.context_length) + " tokens each" : null],
      ["KV cache dtype", c.cache_dtype],
      ["Block size / blocks", block && blocks ? block + " tokens / " + fmtInt(blocks) : null],
      ["GPU memory share", c.gpu_memory_utilization ? fmtPct(+c.gpu_memory_utilization, 0) : null],
      ["Prefix caching", c.enable_prefix_caching === "True" ? "on" : c.enable_prefix_caching === "False" ? "off" : null],
    ]);
    body.push(el("div", { class: "hw-grid" }, [el("div", {}, left), el("div", {}, right)]));
    if (gpus) {
      const host = el("div", { class: "table-wrap" });
      table(host, [["GPU"], ["Name", "text"], ["Memory"], ["ECC", "text"], ["Power limit"], ["PCIe max"], ["Max SM clock"], ["Bus ID", "text"], ["VBIOS", "text"]],
        gpus.map((g) => [String(g.index == null ? "" : g.index), textTd(g.name),
          +g["memory.total"] ? fmtNum(+g["memory.total"] / 1024, 1) + " GiB" : dash, textTd(g["ecc.mode.current"]),
          +g["power.limit"] ? Math.round(+g["power.limit"]) + " W" : dash,
          g["pcie.link.gen.max"] ? "Gen" + g["pcie.link.gen.max"] + " x" + g["pcie.link.width.max"] : dash,
          g["clocks.max.sm"] ? g["clocks.max.sm"] + " MHz" : dash, textTd(g["pci.bus_id"]), textTd(g.vbios_version)]), "");
      body.push(host);
    }
    const logLines = inv && inv.vllm_log ? Object.values(inv.vllm_log).flat() : [];
    if (logLines.length) {
      body.push(el("details", {}, [el("summary", { text: "vLLM startup lines (" + (inv.vllm_log_path || "log") + ")" }),
        el("pre", { class: "raw", text: logLines.join("\n") })]));
    }
    if (inv && inv.topology_text) {
      body.push(el("details", {}, [el("summary", { text: "nvidia-smi topo -m" }), el("pre", { class: "raw", text: inv.topology_text })]));
    }
    if (inv && inv.errors && Object.keys(inv.errors).length) {
      body.push(el("div", { class: "none", text: "GPU helper errors: " + JSON.stringify(inv.errors) }));
    }
    $("hw-body").replaceChildren(...body);
  }

  // ---------- backend state ----------
  const STATE_TEXT = {
    no_pod: "No backend yet: no pod is connected. Connect one on the Pod page; monitoring starts by itself once vLLM answers.",
    connected: "A pod is connected but nothing is deployed yet. Deploy a profile on the Pod page.",
    deploying: "The backend is deploying. GPU data appears as soon as the helper is up, engine data once vLLM answers.",
    stopped: "The backend is stopped. Charts show what was recorded before.",
    failed: "The last deployment failed. See the Pod page for the log.",
    tunnel_down: "The SSH tunnel to the pod is down, so nothing can be sampled. The proxy keeps trying to reconnect.",
  };
  function renderBackend(s) {
    const b = s.backend || {};
    const note = $("backend-note");
    let text = "", cls = "notice warn";
    if (b.state !== "ready") {
      text = (STATE_TEXT[b.state] || "Backend state: " + b.state + ".") + (b.detail ? " (" + b.detail + ")" : "");
      if (b.state === "failed" || b.state === "tunnel_down") cls = "notice bad";
    } else if (s.metrics_ok === false) {
      text = "The backend is ready but its /metrics endpoint could not be read (" + (s.metrics_error || "error")
        + "), so the engine charts are empty. Requests are still recorded.";
    }
    note.className = text ? cls : "notice hidden";
    note.textContent = text;
    $("meta").textContent = [
      b.label,
      b.served_model ? "model " + b.served_model : null,
      b.context_length ? "context " + fmtInt(b.context_length) : null,
      b.mode ? b.mode + " backend" : null,
    ].filter(Boolean).join(" · ");
  }

  function windowMax(series, key) {
    let m = null;
    for (const p of series) if (p[key] != null && (m == null || p[key] > m)) m = p[key];
    return m;
  }

  function render(s) {
    const ser = s.series || [], t = s.totals || {}, cur = s.current || {}, win = fmtWindow(s.window_s);
    renderBackend(s);

    const flightKeys = new Set(s.in_flight.map((f) => f.key)).size;
    const maxSub = (v) => v == null ? "no samples in " + win : "max " + fmtInt(v) + " in " + win;
    const meanSub = (v, f) => v == null ? "no samples in " + win : "mean " + f(v) + " in " + win;
    setTile("open", fmtInt(cur.open), s.in_flight.length ? "from " + flightKeys + (flightKeys === 1 ? " key" : " keys") : maxSub(windowMax(ser, "open")));
    setTile("running", fmtInt(cur.running), maxSub(t.running_max));
    setTile("waiting", fmtInt(cur.waiting), maxSub(t.waiting_max), (cur.waiting || 0) > 0);
    setTile("kv", cur.kv == null ? dash : (cur.kv * 100).toFixed(1), fmtInt(t.preemptions) + " preemptions in " + win, (t.preemptions || 0) > 0);
    setTile("output", fmtNum(cur.output_tps), meanSub(t.output_tps_mean, fmtNum));
    setTile("prompt", fmtInt(cur.prompt_tps), meanSub(t.prompt_tps_mean, fmtInt));
    setTile("completed", fmtInt(t.completed), fmtInt(t.ok) + " ok · " + fmtInt(t.cancelled) + " cancelled in " + win);
    const st = s.store || { dropped: 0, failed: 0 };
    setTile("errors", fmtInt(t.errors), "writer: " + st.dropped + " dropped, " + st.failed + " failed · " + fmtInt(s.rejected) + " rejected keys",
      (t.errors || 0) > 0 || st.dropped > 0 || st.failed > 0);
    const gp = cur.gpus || [];
    tileEls.gpu_util.tile.classList.toggle("hidden", !gp.length);
    tileEls.gpu_mem.tile.classList.toggle("hidden", !gp.length);
    if (gp.length) {
      const utils = gp.map((g) => g.util).filter((v) => v != null);
      setTile("gpu_util", utils.length ? fmtInt(utils.reduce((a, b) => a + b, 0) / utils.length) : dash,
        gp.map((g) => "GPU" + g.gpu + " " + fmtInt(g.util) + "%").join(" · "));
      const used = gp.reduce((a, g) => a + (g.mem_used || 0), 0) / 1024, total = gp.reduce((a, g) => a + (g.mem_total || 0), 0) / 1024;
      setTile("gpu_mem", fmtNum(used, 1), "of " + fmtNum(total, 1) + " GiB across " + gp.length + (gp.length > 1 ? " GPUs" : " GPU"));
    }

    table($("t-flight"),
      [["Key"], ["Session", "text"], ["API", "text"], ["Phase", "text"], ["Elapsed"], ["First content"], ["Content events"], ["Messages"], ["Max tokens"]],
      s.in_flight.map((f) => [textTd(f.key), textTd(f.session), textTd(f.api), textTd(f.phase), fmtMs(f.elapsed_ms), fmtMs(f.first_ms),
        fmtInt(f.content_events), fmtInt(f.messages), fmtInt(f.max_tokens)]),
      "Nothing in flight.");

    renderHardware(s);

    const domain = [s.now - s.window_s, s.now];
    const times = ser.map((p) => p.t);
    const col = (k, scale) => ser.map((p) => p[k] == null ? null : p[k] * (scale || 1));
    charts.concurrency.update(times, { running: col("running"), waiting: col("waiting"), open: col("open") }, domain);
    charts.kv.update(times, { kv: col("kv", 100) }, domain);
    charts.output.update(times, { output_tps: col("output_tps") }, domain);
    charts.prompt.update(times, { prompt_tps: col("prompt_tps") }, domain);
    charts.prefix.update(times, { prefix_hit: col("prefix_hit", 100) }, domain);
    updateGpus(s, times, domain);
    charts.first.update(s.requests || [], s.keys || [], domain, t.completed || 0);

    $("keys-sub").textContent = "requests that arrived in the last " + win + "; latencies on the proxy clock";
    table($("t-keys"),
      [["Key"], ["Requests"], ["OK"], ["Cancelled"], ["Errors"], ["In flight"], ["First p50"], ["First p90"], ["Text p50"], ["Tool p50"],
        ["End p50"], ["End p90"], ["Gap p90"], ["Decode p50 tok/s"], ["Delivered p50 tok/s"], ["Cached"], ["Output tok"], ["Context max"]],
      (s.keys || []).map((a) => [textTd(a.key), fmtInt(a.n), fmtInt(a.ok), fmtInt(a.cancelled), fmtInt(a.errors), fmtInt(a.in_flight),
        fmtMs(a.first_p50), fmtMs(a.first_p90), fmtMs(a.text_p50), fmtMs(a.tool_p50), fmtMs(a.end_p50), fmtMs(a.end_p90), fmtMs(a.gap_p90),
        fmtNum(a.decode_p50), fmtNum(a.delivered_p50), fmtPct(a.cached_ratio), fmtInt(a.completion_tokens), fmtPct(a.ctx_max, 1)]),
      "No finished requests in this window.");

    const kind = s.window_s > 86400 ? "full" : "hms";
    const recent = (s.requests || []).slice(-30).reverse();
    table($("t-recent"),
      [["Time"], ["Key", "text"], ["Session", "text"], ["API", "text"], ["Outcome", "text"], ["Finish", "text"], ["Msgs"], ["Prompt tok"],
        ["Cached"], ["Output tok"], ["Reasoning tok"], ["Open at start"], ["First content"], ["First text"], ["First tool"], ["End"],
        ["Max gap"], ["Decode tok/s"], ["Delivered tok/s"], ["Context"]],
      recent.map((r) => [fmtTime(r.t_arrival_wall, kind), textTd(r.actor_id), textTd(r.session_id), textTd(r.api), outcomeCell(r.outcome),
        textTd(r.finish_reason), fmtInt(r.messages_count), fmtInt(r.prompt_tokens),
        r.prompt_tokens && r.cached_tokens != null ? fmtPct(r.cached_tokens / r.prompt_tokens) : dash,
        fmtInt(r.completion_tokens), fmtInt(r.reasoning_tokens), fmtInt(r.open_requests_at_start), fmtMs(r.first_payload_ms),
        fmtMs(r.first_text_ms), fmtMs(r.first_tool_ms), fmtMs(r.end_ms), fmtMs(r.max_gap_ms), fmtNum(r.decode_tps),
        fmtNum(r.delivered_tps), fmtPct(r.context_fraction, 1)]),
      "No finished requests in this window.");
  }

  // ---------- window selector ----------
  const WINDOWS = [[120, "2 min"], [300, "5 min"], [900, "15 min"], [3600, "1 h"], [21600, "6 h"], [86400, "24 h"], [604800, "7 d"], [2592000, "30 d"]];
  let windowsFor = null;
  function buildWindows(retentionDays) {
    if (windowsFor === retentionDays) return;
    windowsFor = retentionDays;
    const limit = retentionDays > 0 ? retentionDays * 86400 : Infinity;
    const list = WINDOWS.filter(([w]) => w <= limit);
    if (retentionDays > 0 && !list.some(([w]) => w === limit)) list.push([limit, retentionDays + " d"]);
    if (!list.some(([w]) => w === state.windowS)) state.windowS = 300;
    $("windows").replaceChildren(...list.map(([w, label]) => el("button", {
      type: "button", "aria-pressed": String(w === state.windowS), text: label,
      onclick: () => {
        state.windowS = w;
        try { localStorage.setItem("ovp-monitor-window", String(w)); } catch (e) { /* not remembered */ }
        for (const b of $("windows").children) b.setAttribute("aria-pressed", String(b.textContent === label));
        poll();
      },
    })));
  }

  // ---------- status & polling ----------
  function setStatus(kind, text) {
    const node = $("status");
    node.dataset.state = kind;
    node.querySelector(".icon").textContent = kind === "live" ? "●" : kind === "warn" ? "▲" : "✕";
    $("status-text").textContent = text;
  }
  async function poll() {
    clearTimeout(state.timer);
    if (state.paused) return;
    if (state.busy) { state.again = true; return; }
    state.busy = true;
    let delay = 2000;
    try {
      const snap = await api("GET", "/live?window=" + state.windowS);
      state.snap = snap;
      state.lastOk = Date.now();
      buildWindows(snap.retention_days);
      render(snap);
      setStatus("live", "Live");
      $("updated").textContent = "Updated " + fmtTime(snap.now, "hms") + (snap.source === "database" ? " · from the database, refreshed every " + Math.round(snap.refresh_s) + " s" : "");
      delay = Math.max(1000, (snap.refresh_s || 2) * 1000);
    } catch (err) {
      const age = state.lastOk ? Math.round((Date.now() - state.lastOk) / 1000) : null;
      setStatus("down", age == null ? "Proxy unreachable" : "Proxy unreachable, last data " + age + " s ago");
    }
    state.busy = false;
    if (state.again) { state.again = false; poll(); return; }
    if (!state.paused && !document.hidden) state.timer = setTimeout(poll, delay);
  }

  // ---------- report download ----------
  const RANGES = [["window", ""], ["900", "Last 15 minutes"], ["3600", "Last hour"], ["21600", "Last 6 hours"], ["86400", "Last 24 hours"],
    ["604800", "Last 7 days"], ["all", "Everything recorded"], ["custom", "Custom range…"]];
  let exOptions = null;
  function exStatus(text, err) {
    $("ex-status").textContent = text || "";
    $("ex-status").classList.toggle("err", !!err);
  }
  function fillSelect(id, options, selected) {
    $(id).replaceChildren(...options.map(([value, label]) => el("option", { value, selected: value === selected, text: label })));
  }
  const localInput = (t) => {
    const d = new Date(t * 1000);
    return d.getFullYear() + "-" + pad2(d.getMonth() + 1) + "-" + pad2(d.getDate()) + "T" + pad2(d.getHours()) + ":" + pad2(d.getMinutes()) + ":" + pad2(d.getSeconds());
  };
  function exRange() {
    const choice = $("ex-range").value, now = Date.now() / 1000;
    if (choice === "custom") {
      const from = new Date($("ex-from").value).getTime() / 1000, to = new Date($("ex-to").value).getTime() / 1000;
      if (!isFinite(from) || !isFinite(to)) throw new Error("Enter both a start and an end time.");
      if (to <= from) throw new Error("The end time must be later than the start time.");
      return [from, to];
    }
    if (choice === "all") {
      const span = (exOptions && exOptions.span) || {};
      if (span.first == null) throw new Error("Nothing has been recorded yet.");
      return [Math.floor(span.first) - 1, now];
    }
    return [now - (choice === "window" ? state.windowS : +choice), now];
  }
  async function loadOptions() {
    exStatus("Loading…");
    try {
      const d = exOptions = await api("GET", "/export/options");
      const keep = { key: $("ex-key").value, session: $("ex-session").value, range: $("ex-range").value || "window" };
      fillSelect("ex-range", RANGES.map(([v, label]) => [v, v === "window" ? "Shown window (last " + fmtWindow(state.windowS) + ")" : label]), keep.range);
      fillSelect("ex-key", [["", "All keys"]].concat(d.keys.map((k) => [k.key, k.key + " · " + fmtInt(k.requests) + " req"])), keep.key);
      fillSelect("ex-session", [["", "All sessions"]].concat(d.sessions.map((x) =>
        [x.session, x.session + " · " + fmtInt(x.requests) + " req · " + fmtTime(x.first, "dhm") + " to " + fmtTime(x.last, "dhm")])), keep.session);
      if (!$("ex-from").value) {
        $("ex-from").value = localInput(Math.max(d.span.first || 0, d.now - 3600));
        $("ex-to").value = localInput(d.now);
      }
      exStatus(d.span.first == null ? "Nothing has been recorded yet; a report would be empty."
        : "Data recorded from " + fmtTime(d.span.first, "dhm") + " to " + fmtTime(d.span.last, "dhm") + " · " + fmtInt(d.requests) + " requests"
          + (d.unnamed_session_requests ? " (" + fmtInt(d.unnamed_session_requests) + " without an X-Session-Id)" : "")
          + (d.deployments.length ? " · deployments: " + d.deployments.map((x) => x.label).join(", ") : ""));
    } catch (err) { exStatus("Could not load the choices: " + err.message, true); }
  }
  async function download() {
    const btn = $("ex-go");
    try {
      const [from, to] = exRange();
      const p = new URLSearchParams({ from: from.toFixed(3), to: to.toFixed(3), format: $("ex-format").value,
        tz_offset: String(-new Date().getTimezoneOffset()) });
      if ($("ex-key").value) p.set("key", $("ex-key").value);
      if ($("ex-session").value) p.set("session", $("ex-session").value);
      btn.disabled = true;
      exStatus("Building the report…");
      // The token travels in a header, never in the URL, so the file is fetched and saved from memory.
      const resp = await window.OVP.fetch("/export?" + p);
      if (!resp.ok) {
        let msg = "Request failed (" + resp.status + ")";
        try { msg = (await resp.json()).error || msg; } catch (e) { /* keep the status text */ }
        throw new Error(resp.status === 401 ? "The admin token was not accepted." : msg);
      }
      const m = /filename="([^"]+)"/.exec(resp.headers.get("content-disposition") || "");
      const name = m ? m[1] : "ovp-report." + p.get("format");
      const url = URL.createObjectURL(await resp.blob());
      const a = el("a", { href: url, download: name });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      exStatus("Saved " + name);
    } catch (err) {
      exStatus("Download failed: " + err.message, true);
    } finally {
      btn.disabled = false;
    }
  }

  function initControls() {
    $("export-toggle").addEventListener("click", () => {
      const open = $("export").classList.toggle("hidden") === false;
      $("export-toggle").setAttribute("aria-expanded", String(open));
      if (open) loadOptions();
    });
    $("ex-range").addEventListener("change", () => {
      const custom = $("ex-range").value === "custom";
      for (const node of document.querySelectorAll(".ex-custom")) node.classList.toggle("hidden", !custom);
    });
    $("export-form").addEventListener("submit", (ev) => { ev.preventDefault(); download(); });
    $("pause").addEventListener("click", () => {
      state.paused = !state.paused;
      $("pause").setAttribute("aria-pressed", String(state.paused));
      $("pause").textContent = state.paused ? "Resume" : "Pause";
      if (state.paused) { clearTimeout(state.timer); setStatus("warn", "Paused"); } else poll();
    });
    document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  }

  boot("/admin/monitor", () => {
    document.body.append(tip);
    buildTiles();
    buildCharts();
    buildWindows(0);
    initControls();
    poll();
  });
})();
