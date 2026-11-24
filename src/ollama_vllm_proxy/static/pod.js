// Pod page: connection, Hugging Face token, profiles, fit plan, deployment progress and logs.
// All values are rendered as text nodes; nothing from the server is inserted as HTML.
(function () {
  "use strict";
  const { api, stream, el, boot, fmtTime } = window.OVP;
  const $ = (id) => document.getElementById(id);
  const show = (node, on) => node.classList.toggle("hidden", !on);

  let status = null;        // last /deploy/status snapshot
  let profiles = [];
  let template = null;      // default profile, for "New"
  let remoteRoot = "/workspace";
  let plan = null;          // fit plan of the selected profile on the connected pod
  let planKey = "";
  let chosenVariant = null; // radio selection in the plan table
  let editingId = null;     // profile being edited (null = new)
  let pendingFingerprint = null;
  let events = null;
  let actionBusy = false;

  // ---------------------------------------------------------------- helpers

  function fail(err) {
    const box = $("message");
    box.textContent = err && err.message ? err.message : String(err);
    show(box, true);
    box.scrollIntoView({ block: "nearest" });
  }
  function clearError() { show($("message"), false); }

  async function act(fn) {
    if (actionBusy) return;
    actionBusy = true;
    clearError();
    render();
    try { await fn(); } catch (err) { fail(err); }
    actionBusy = false;
    render();
  }

  function gb(bytes) {
    if (bytes == null) return "–";
    return (bytes / 1e9).toFixed(bytes >= 1e11 ? 0 : 1) + " GB";
  }
  function num(n) { return n == null ? "–" : Number(n).toLocaleString("en-US"); }
  function duration(seconds) {
    if (seconds < 0 || !isFinite(seconds)) return "";
    if (seconds < 60) return seconds.toFixed(seconds < 10 ? 1 : 0) + " s";
    const m = Math.floor(seconds / 60);
    return m + " min " + String(Math.round(seconds - m * 60)).padStart(2, "0") + " s";
  }
  function pill(text, cls) { return el("span", { class: "pill" + (cls ? " " + cls : ""), text }); }

  const external = () => !!(status && status.external);
  const connected = () => !!(status && status.pod && status.pod.connected);
  const busy = () => !!(status && status.busy);
  const backendState = () => (status ? status.backend.state : "");
  const selectedProfile = () => profiles.find((p) => String(p.id) === $("profile-select").value) || null;

  // ---------------------------------------------------------------- state line

  const STATE_LABELS = {
    no_pod: ["no pod", ""], connected: ["connected", ""], deploying: ["deploying", "warn"], ready: ["ready", "good"],
    failed: ["failed", "bad"], tunnel_down: ["tunnel down", "bad"], stopped: ["stopped", "warn"],
  };

  function renderState() {
    const b = status.backend;
    const [label, cls] = STATE_LABELS[b.state] || [b.state, ""];
    const pillNode = $("state-pill");
    pillNode.className = "pill" + (cls ? " " + cls : "");
    pillNode.textContent = label + (b.state === "deploying" && b.detail ? " (" + b.detail + ")" : "");
    let text = "";
    if (b.state === "ready") text = (b.served_model || "") + (b.label ? " · " + b.label : "") + (b.context_length ? " · context " + num(b.context_length) : "");
    else if (b.state !== "deploying") text = b.detail || "";
    $("state-text").textContent = text;
  }

  // ---------------------------------------------------------------- connection

  function renderParsed(parsed) {
    const box = $("parsed");
    if (!parsed) { show(box, false); return; }
    const item = (name, value) => el("span", {}, [el("b", { text: name }), el("code", { text: String(value) })]);
    const nodes = [item("Host", parsed.host), item("Port", parsed.port), item("User", parsed.user), item("Key", parsed.key_path)];
    if (parsed.key_problem) nodes.push(el("span", { class: "warn", text: "⚠ " + parsed.key_problem }));
    box.replaceChildren(...nodes);
    show(box, true);
  }

  function connError(text) {
    $("conn-error").textContent = text || "";
    show($("conn-error"), !!text);
  }

  function renderConnection() {
    const pod = status.pod;
    const input = $("ssh-command");
    if (pod && document.activeElement !== input && !input.dataset.dirty) input.value = pod.ssh_command;
    if (pod && !input.dataset.dirty) renderParsed(pod);
    const up = connected();
    const linked = !!(pod && pod.state !== "saved" && pod.state !== "closed");
    input.disabled = external() || linked;
    $("connect-btn").disabled = external() || actionBusy || linked;
    show($("connect-btn"), !linked);
    show($("disconnect-btn"), linked);
    $("disconnect-btn").disabled = actionBusy;
    show($("connected-box"), linked);
    if (linked) {
      const facts = pod.facts || {};
      const parts = [];
      if (up) parts.push(pill("connected", "good"));
      else if (pod.state === "hostkey_changed") parts.push(pill("host key changed", "bad"));
      else parts.push(pill("reconnecting", "bad"));
      parts.push(" " + pod.user + "@" + pod.host + ":" + pod.port);
      if (facts.gpu_summary) parts.push(" · " + facts.gpu_summary);
      if (facts.gpu_error) parts.push(" · GPUs unreadable: " + facts.gpu_error);
      if (facts.runpod && facts.runpod.RUNPOD_POD_ID) parts.push(" · pod " + facts.runpod.RUNPOD_POD_ID);
      if (!up && pod.detail) parts.push(" · " + pod.detail);
      $("pod-summary").replaceChildren(...parts);
      const more = el("p", { class: "small muted" }, ["Host key ", el("code", { text: pod.fingerprint || "not confirmed" })]);
      $("pod-summary").append(more);
    }
    if (linked || external()) show($("fingerprint-box"), false);
  }

  async function saveConnection() {
    const input = $("ssh-command");
    const value = input.value.trim();
    connError("");
    if (!value) { renderParsed(null); return false; }
    try {
      const data = await api("PUT", "/pod", { ssh_command: value });
      delete input.dataset.dirty;
      status = data;
      renderParsed(data.parsed);
      return true;
    } catch (err) {
      renderParsed(null);
      connError(err.message);
      return false;
    }
  }

  async function connect(accept) {
    connError("");
    let result;
    try {
      result = await api("POST", "/pod/connect", accept ? { accept_fingerprint: accept } : {});
    } catch (err) { connError(err.message); return; }
    if (result.state === "confirm") {
      pendingFingerprint = result.fingerprint;
      $("fp-value").textContent = result.fingerprint;
      $("fp-host").textContent = status && status.pod ? status.pod.host + ":" + status.pod.port : "the pod";
      show($("fp-changed"), !!result.changed);
      show($("fingerprint-box"), true);
      $("fp-accept").focus();
      return;
    }
    pendingFingerprint = null;
    show($("fingerprint-box"), false);
    if (result.status) { status = result.status; render(); }
    loadPlan(true);
  }

  // ---------------------------------------------------------------- Hugging Face token

  function renderToken() {
    const set = !!status.hf_token_set;
    const node = $("hf-status");
    node.textContent = set ? "a token is set" : "no token set";
    node.className = "pill" + (set ? " good" : "");
    $("hf-save").textContent = set ? "Replace token" : "Save token";
    show($("hf-remove"), set);
    $("hf-save").disabled = $("hf-remove").disabled = $("hf-token").disabled = external();
  }

  // ---------------------------------------------------------------- profiles

  function argsToText(args) { return (args || []).map((a) => a.flag + (a.value ? " " + a.value : "")).join("\n"); }
  function textToArgs(text) {
    return text.split("\n").map((l) => l.trim()).filter(Boolean).map((line) => {
      const i = line.search(/\s/);
      return i < 0 ? { flag: line, value: "" } : { flag: line.slice(0, i), value: line.slice(i + 1).trim() };
    });
  }
  function envToText(env) { return Object.entries(env || {}).map(([k, v]) => k + "=" + v).join("\n"); }
  function textToEnv(text) {
    const out = {};
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      if (!line) continue;
      const i = line.indexOf("=");
      if (i < 0) throw new Error("Environment line '" + line + "' must look like NAME=value");
      out[line.slice(0, i).trim()] = line.slice(i + 1);
    }
    return out;
  }

  function renderProfileSummary() {
    const p = selectedProfile();
    const dl = $("profile-summary");
    if (!p) { dl.replaceChildren(); return; }
    const d = p.data;
    const row = (k, v) => [el("dt", { text: k }), el("dd", {}, v)];
    const variants = el("div", {}, (d.variants || []).map((v, i) => el("div", {}, [
      (i + 1) + ". ", el("code", { text: v.repo + (v.revision ? "@" + v.revision : "") }), " (" + v.label + ") → ",
      el("code", { text: remoteRoot + "/" + v.dir }),
      v.args && v.args.length ? " · " + argsToText(v.args).replace(/\n/g, " ") : "",
    ])));
    dl.replaceChildren(
      ...row("Served model name", el("code", { text: d.served_model_name })),
      ...row("Variants", variants),
      ...row("Context", num(d.context_wanted) + " wanted, " + num(d.context_min) + " minimum"),
      ...row("vLLM arguments", el("code", { text: argsToText(d.args).replace(/\n/g, " ") || "none" })),
      ...row("GPU memory utilization", d.gpu_memory_utilization == null ? "vLLM default" : String(d.gpu_memory_utilization)),
      ...row("vLLM / Python", d.vllm_version + " / " + d.python_version),
      ...row("Environment", el("code", { text: envToText(d.env).replace(/\n/g, " ") || "none" })),
    );
  }

  function renderProfiles() {
    const select = $("profile-select");
    const current = select.value || (status && status.deployment ? String(status.deployment.profile_id) : "");
    select.replaceChildren(...profiles.map((p) => el("option", { value: String(p.id), text: p.name + " — serves " + p.data.served_model_name })));
    if (profiles.some((p) => String(p.id) === current)) select.value = current;
    const none = profiles.length === 0;
    $("profile-dup").disabled = $("profile-edit").disabled = $("profile-del").disabled = none || external();
    $("profile-new").disabled = external();
    renderProfileSummary();
  }

  async function loadProfiles() {
    const data = await api("GET", "/profiles");
    profiles = data.profiles;
    template = data.template;
    remoteRoot = data.remote_root;
    renderProfiles();
  }

  function variantRow(v) {
    const field = (cls, label, value, placeholder) => el("div", {}, [
      el("label", { text: label }),
      el("input", { type: "text", class: cls, value: value || "", placeholder: placeholder || "", spellcheck: "false", "aria-label": label }),
    ]);
    const node = el("div", { class: "variant" }, [
      field("v-label", "Label", v.label, "BF16"),
      field("v-repo", "Hugging Face repo", v.repo, "owner/name"),
      field("v-rev", "Revision", v.revision, "main"),
      field("v-dir", "Directory", v.dir, "model-dir"),
      field("v-args", "Extra arguments", argsToText(v.args).replace(/\n/g, "  "), "--kv-cache-dtype fp8"),
      el("button", { type: "button", class: "danger", text: "Remove", onclick: () => node.remove() }),
    ]);
    return node;
  }

  function openEditor(profile, title, id) {
    editingId = id;
    const d = profile.data;
    $("profile-form-title").textContent = title;
    $("pf-name").value = profile.name;
    $("pf-served").value = d.served_model_name;
    $("pf-ctx").value = d.context_wanted;
    $("pf-ctx-min").value = d.context_min;
    $("pf-util").value = d.gpu_memory_utilization == null ? "" : d.gpu_memory_utilization;
    $("pf-vllm").value = d.vllm_version;
    $("pf-python").value = d.python_version;
    $("pf-args").value = argsToText(d.args);
    $("pf-env").value = envToText(d.env);
    $("pf-variants").replaceChildren(
      el("p", { class: "small muted" }, ["Each variant is one Hugging Face repo, downloaded into its directory under ", el("code", { text: remoteRoot }), "."]),
      ...(d.variants || []).map(variantRow));
    show($("profile-error"), false);
    show($("profile-form"), true);
    $("pf-name").focus();
  }

  function readEditor() {
    // "--a x  --b y" on one line: split extra arguments at each " --"
    const variantArgs = (text) => textToArgs(text.trim().replace(/\s+(?=--)/g, "\n"));
    const variants = [...$("pf-variants").querySelectorAll(".variant")].map((row) => ({
      label: row.querySelector(".v-label").value.trim(),
      repo: row.querySelector(".v-repo").value.trim(),
      revision: row.querySelector(".v-rev").value.trim(),
      dir: row.querySelector(".v-dir").value.trim(),
      args: variantArgs(row.querySelector(".v-args").value),
    }));
    const util = $("pf-util").value.trim();
    return {
      name: $("pf-name").value.trim(),
      data: {
        served_model_name: $("pf-served").value.trim(),
        variants,
        context_wanted: parseInt($("pf-ctx").value, 10),
        context_min: parseInt($("pf-ctx-min").value, 10),
        args: textToArgs($("pf-args").value),
        gpu_memory_utilization: util === "" ? null : parseFloat(util),
        vllm_version: $("pf-vllm").value.trim(),
        python_version: $("pf-python").value.trim(),
        env: textToEnv($("pf-env").value),
      },
    };
  }

  async function saveProfile(ev) {
    ev.preventDefault();
    const box = $("profile-error");
    show(box, false);
    try {
      const body = readEditor();
      const saved = editingId == null ? await api("POST", "/profiles", body) : await api("PUT", "/profiles/" + editingId, body);
      show($("profile-form"), false);
      await loadProfiles();
      $("profile-select").value = String(saved.id);
      renderProfileSummary();
      loadPlan(true);
      loadSizing();
    } catch (err) {
      box.textContent = err.message;
      show(box, true);
    }
  }

  // ---------------------------------------------------------------- plan

  function chosenRow() {
    if (!plan) return null;
    return plan.rows.find((r) => r.variant === chosenVariant) || null;
  }

  function overrides() {
    const out = {};
    const row = chosenRow();
    if (!row) return out;
    if (plan.auto !== row.variant) out.variant = row.variant;
    const ctx = parseInt($("ov-context").value, 10);
    if (Number.isFinite(ctx) && ctx !== row.context) { out.context = ctx; out.variant = row.variant; }
    const tp = $("ov-tp").value;
    if (tp && Number(tp) !== row.tensor_parallel) { out.tensor_parallel = Number(tp); out.variant = row.variant; }
    if ($("ov-force").checked) { out.force = true; out.variant = row.variant; }
    return out;
  }

  function renderCommand() {
    const row = chosenRow();
    let text = row ? row.command || "" : "";
    const o = overrides();
    if (text && o.context) text = text.replace(/--max-model-len \d+/, "--max-model-len " + o.context);
    if (text && o.tensor_parallel) text = text.replace(/--tensor-parallel-size \d+/, "--tensor-parallel-size " + o.tensor_parallel);
    $("command").textContent = text || "No command: this variant cannot run on the pod's GPUs.";
  }

  function verdict(row) {
    if (row.status === "fits") return pill("fits", "good");
    if (row.status === "untested") return pill("fits · untested", "warn");
    if (row.status === "unknown") return pill("unknown", "warn");
    return pill("does not fit", "bad");
  }

  function renderPlan() {
    const up = connected();
    show($("plan-body"), up && !!plan);
    show($("plan-note"), !(up && plan));
    if (!up) {
      $("plan-note").textContent = external() ? "Pod management is disabled in external mode." : "Connect a pod to see how this profile fits its GPUs.";
    } else if (!plan) {
      $("plan-note").textContent = selectedProfile() ? "Reading the pod's GPUs and the model repositories…" : "Create a profile first.";
    }
    if (!(up && plan)) return;
    $("plan-gpus").replaceChildren("This pod: ", el("strong", { text: plan.gpu_summary }),
      plan.gpu_error ? " — " + plan.gpu_error : "",
      plan.auto ? "" : " — no variant fits by the estimate; choose one and tick “Deploy anyway” to try regardless.");
    if (!plan.rows.some((r) => r.variant === chosenVariant)) chosenVariant = plan.auto || (plan.rows[0] && plan.rows[0].variant);
    const body = $("plan-table").querySelector("tbody");
    body.replaceChildren(...plan.rows.map((row) => {
      const radio = el("input", { type: "radio", name: "variant", value: row.variant, "aria-label": "Deploy " + row.variant,
        checked: row.variant === chosenVariant, onchange: () => { chosenVariant = row.variant; resetOverrides(); renderPlan(); } });
      const reason = [row.reason].concat(row.warnings || []).filter(Boolean).join(" · ");
      return el("tr", { class: row.variant === chosenVariant ? "chosen" : "" }, [
        el("td", {}, radio),
        el("td", {}, [el("strong", { text: row.variant }), row.variant === plan.auto ? " " : "", row.variant === plan.auto ? pill("automatic") : "",
          el("div", { class: "small muted", text: row.repo })]),
        el("td", { class: "num", text: gb(row.weights_bytes) }),
        el("td", { class: "num", text: row.tensor_parallel == null ? "–" : String(row.tensor_parallel) }),
        el("td", { class: "num", text: row.kv_bytes == null ? "–" : (row.kv_bytes > 0 ? gb(row.kv_bytes) : "none") }),
        el("td", { class: "num", text: row.max_context === 0 ? "–" : num(row.context) }),
        el("td", { class: "num", text: row.disk_request_gb == null ? "–" : row.disk_request_gb + " GB" }),
        el("td", { class: "wrap" }, [verdict(row), el("div", { class: "small", text: reason })]),
      ]);
    }));
    const row = chosenRow();
    const tp = $("ov-tp");
    const keep = tp.value;
    tp.replaceChildren(el("option", { value: "", text: "automatic" + (row && row.tensor_parallel ? " (" + row.tensor_parallel + ")" : "") }),
      ...(plan.tensor_parallel_options || []).map((n) => el("option", { value: String(n), text: String(n) })));
    if ([...tp.options].some((o) => o.value === keep)) tp.value = keep;
    $("ov-context").placeholder = row ? "automatic (" + num(row.context) + ")" : "automatic";
    renderCommand();
  }

  function resetOverrides() {
    $("ov-context").value = "";
    $("ov-tp").value = "";
    $("ov-force").checked = false;
  }

  async function loadPlan(force) {
    const p = selectedProfile();
    const key = connected() && p ? p.id + ":" + p.updated_at + ":" + (status.pod.fingerprint || "") + ":" + status.hf_token_set : "";
    if (!force && key === planKey) return;
    planKey = key;
    if (!key) { plan = null; renderPlan(); renderButtons(); return; }
    plan = null;
    renderPlan();
    try {
      const data = await api("GET", "/profiles/" + p.id + "/plan");
      if (planKey !== key) return;
      plan = data;
      chosenVariant = plan.auto;
      resetOverrides();
    } catch (err) {
      if (planKey === key) { planKey = ""; $("plan-note").textContent = err.message; }
    }
    renderPlan();
    renderButtons();
  }

  // ---------------------------------------------------------------- sizing table and catalog

  async function loadSizing() {
    const p = selectedProfile();
    const body = $("sizing-table").querySelector("tbody");
    if (!p) { body.replaceChildren(); return; }
    try {
      const data = await api("GET", "/profiles/" + p.id + "/sizing");
      const errors = data.errors || [];
      $("sizing-error").textContent = errors.join(" ");
      show($("sizing-error"), errors.length > 0);
      $("sizing-disk").replaceChildren("Container disk to request (model + 25 GB): ",
        ...data.variants.map((v, i) => el("span", {}, [i ? " · " : "", el("strong", { text: v.variant }), " ",
          v.disk_request_gb == null ? "unknown" : v.disk_request_gb + " GB (" + gb(v.download_bytes) + " download)"])));
      const known = data.variants.map((v) => v.disk_request_gb).filter((n) => n != null);
      if (known.length) $("disk-hint").textContent = "For the selected profile: " + Math.max(...known) + " GB covers every variant" +
        (known.length > 1 ? ", " + Math.min(...known) + " GB the smallest." : ".");
      body.replaceChildren(...data.rows.map((r) => el("tr", { class: r.variant ? "" : "no" }, [
        el("td", { text: r.gpu + " (" + r.memory_gb + " GB)" }),
        el("td", { text: r.generation }),
        el("td", { class: "num", text: String(r.count) }),
        el("td", {}, r.variant ? [el("strong", { text: r.variant }), r.status === "untested" ? " " : "", r.status === "untested" ? pill("untested", "warn") : ""] : pill("does not fit", "")),
        el("td", { class: "num", text: r.tensor_parallel == null ? "–" : String(r.tensor_parallel) }),
        el("td", { class: "num", text: r.variant ? num(r.context) : "–" }),
        el("td", { class: "num", text: r.disk_request_gb == null ? "–" : r.disk_request_gb + " GB" }),
        el("td", { class: "wrap small", text: r.reason }),
      ])));
    } catch (err) {
      $("sizing-error").textContent = err.message;
      show($("sizing-error"), true);
    }
  }

  async function loadCatalog() {
    const data = await api("GET", "/gpu-catalog");
    $("catalog-text").value = "[\n" + data.catalog.map((e) => "  " + JSON.stringify(e)).join(",\n") + "\n]";
  }

  async function saveCatalog(reset) {
    const box = $("catalog-error");
    show(box, false);
    try {
      const value = reset ? null : JSON.parse($("catalog-text").value);
      await api("PUT", "/gpu-catalog", { catalog: value });
      await loadCatalog();
      await loadSizing();
    } catch (err) {
      box.textContent = err.message;
      show(box, true);
    }
  }

  // ---------------------------------------------------------------- deployment progress

  const ICONS = { pending: "○", running: "◐", done: "✓", skipped: "✓", failed: "✕", cancelled: "–" };
  const openSteps = {};   // step id -> user toggled the output open/closed

  function stepNode(step, depState) {
    const now = Date.now() / 1000;
    let when = step.status;
    if (step.started_at && step.status !== "pending") {
      const end = step.ended_at || (step.status === "running" ? now : step.started_at);
      when = (step.status === "running" ? "running · " : step.status === "skipped" ? "skipped · " : step.status === "failed" ? "failed · " : step.status === "cancelled" ? "cancelled · " : "") + duration(end - step.started_at);
    }
    const more = [];
    const p = step.progress;
    if (p && p.total) {
      const pct = Math.max(0, Math.min(100, (100 * p.done) / p.total));
      more.push(el("div", { class: "bar", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(pct)), "aria-label": step.title + " progress" },
        el("span", { style: "width:" + pct.toFixed(1) + "%" })));
      more.push(el("div", { class: "bar-label", text: gb(Math.min(p.done, p.total)) + " of " + gb(p.total) + " (" + pct.toFixed(0) + "%)" }));
    }
    if (step.output) {
      const auto = step.status === "running" || step.status === "failed";
      const open = step.id in openSteps ? openSteps[step.id] : auto;
      const lines = step.output.split("\n");
      const details = el("details", { open }, [
        el("summary", { text: lines[lines.length - 1].slice(0, 140) }),
        el("pre", { text: step.output }),
      ]);
      details.addEventListener("toggle", () => { if (details.open !== auto) openSteps[step.id] = details.open; else delete openSteps[step.id]; });
      more.push(details);
    }
    return el("li", { class: step.status, "data-step": step.id }, [
      el("span", { class: "icon", "aria-hidden": "true", text: ICONS[step.status] || "○" }),
      el("span", { class: "name", text: step.title }),
      el("span", { class: "when", text: when }),
      el("div", { class: "more" }, more),
    ]);
  }

  let lastStepsJson = "";
  function renderDeployment() {
    const dep = status.deployment;
    show($("dep-none"), !dep);
    show($("dep-body"), !!dep);
    if (!dep) return;
    const states = { running: ["running", "warn"], ready: ["serving", "good"], failed: ["failed", "bad"], cancelled: ["cancelled", ""], stopped: ["stopped", "warn"] };
    const [label, cls] = states[dep.state] || [dep.state, ""];
    const plan_ = dep.plan || {};
    const head = [pill(label, cls), " ", el("strong", { text: (dep.kind === "restart" ? "Restart" : "Deployment") + " #" + dep.id }), " · profile ", el("code", { text: dep.profile_name || "?" })];
    if (plan_.variant) head.push(" · " + plan_.variant + " on " + plan_.gpu_count + "× " + plan_.gpu_name + ", tensor-parallel " + plan_.tensor_parallel + ", context " + num(plan_.context));
    if (dep.label) head.push(el("div", { class: "small muted" }, ["Label ", el("code", { text: dep.label }), " · started " + fmtTime(dep.started_at),
      dep.ended_at ? " · " + duration(dep.ended_at - dep.started_at) : ""]));
    $("dep-head").replaceChildren(...head);
    $("dep-error").textContent = dep.error || "";
    show($("dep-error"), !!dep.error && dep.state !== "ready");
    const offer = dep.offer && dep.state === "failed" ? dep.offer : null;
    show($("dep-offer"), !!offer);
    if (offer) {
      $("dep-offer-text").textContent = "vLLM can serve a context of " + num(offer.context) + " tokens on this pod. ";
      $("dep-offer-btn").dataset.context = String(offer.context);
      $("dep-offer-btn").disabled = busy() || !connected();
    }
    // Rebuild the list only when something changed, so open outputs and scroll positions are kept.
    const json = JSON.stringify(dep.steps) + dep.state;
    const running = dep.steps.some((s) => s.status === "running");
    if (json !== lastStepsJson || running) {
      const scroll = {};
      $("steps").querySelectorAll("li").forEach((li) => { const pre = li.querySelector("pre"); if (pre) scroll[li.dataset.step] = [pre.scrollTop, pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4]; });
      $("steps").replaceChildren(...dep.steps.map((s) => stepNode(s, dep.state)));
      $("steps").querySelectorAll("li").forEach((li) => {
        const pre = li.querySelector("pre");
        if (!pre) return;
        const saved = scroll[li.dataset.step];
        pre.scrollTop = !saved || saved[1] ? pre.scrollHeight : saved[0];
      });
      lastStepsJson = json;
    }
  }

  function renderButtons() {
    const up = connected();
    const dep = status ? status.deployment : null;
    const idle = up && !busy() && !actionBusy && !external();
    const row = chosenRow();
    $("deploy-btn").disabled = !(idle && selectedProfile() && plan && row);
    $("cancel-btn").disabled = !(busy() && !actionBusy);
    $("stop-btn").disabled = !(idle && ["ready", "failed", "stopped", "connected", "tunnel_down"].includes(backendState()) && dep);
    $("restart-btn").disabled = !(idle && dep && dep.plan && dep.plan.variant);
    $("delete-btn").disabled = !idle;
    $("deploy-btn").textContent = busy() ? "Deploying…" : "Deploy";
  }

  function render() {
    if (!status) return;
    show($("external"), external());
    renderState();
    renderConnection();
    renderToken();
    renderDeployment();
    renderPlan();
    renderButtons();
  }

  // ---------------------------------------------------------------- logs

  const MAX_LOG_LINES = 5000;
  let logLines = [];
  let logHeld = [];      // lines received while paused
  let logPaused = false;
  let logStream = null;
  let logDirty = false;
  let logKey = "";

  function flushLog() {
    logDirty = false;
    const view = $("log-view");
    view.textContent = logLines.join("\n");
    if ($("log-autoscroll").checked) view.scrollTop = view.scrollHeight;
  }
  function scheduleLog() {
    if (!logDirty) { logDirty = true; requestAnimationFrame(flushLog); }
  }
  function pushLog(lines) {
    const target = logPaused ? logHeld : logLines;
    for (const line of lines) target.push(line);
    if (target.length > MAX_LOG_LINES) target.splice(0, target.length - MAX_LOG_LINES);
    if (logPaused) $("log-info").textContent = logHeld.length + " new lines while paused";
    else scheduleLog();
  }

  function startLog(force) {
    const source = $("log-source").value;
    const key = connected() ? source + ":" + status.pod.fingerprint : "";
    if (!force && key === logKey) return;
    logKey = key;
    if (logStream) { logStream.close(); logStream = null; }
    logLines = [];
    logHeld = [];
    scheduleLog();
    if (!key) { $("log-info").textContent = "No pod connected."; return; }
    $("log-info").textContent = "";
    logStream = stream("/logs/stream?source=" + encodeURIComponent(source), (event) => {
      if (logKey !== key || typeof event !== "object" || !event) return;
      if (event.reset) { logLines = []; logHeld = []; scheduleLog(); $("log-info").textContent = event.path || ""; }
      if (event.info) $("log-info").textContent = event.info;
      if (event.lines) pushLog(event.lines);
    }, () => {
      if (logKey !== key) return;
      logStream = null;
      logKey = "";
      setTimeout(() => { if (connected()) startLog(false); }, 3000);
    });
  }

  async function downloadLog() {
    const source = $("log-source").value;
    const data = await api("GET", "/logs/download?format=json&source=" + encodeURIComponent(source));
    const url = URL.createObjectURL(new Blob([data.text], { type: "text/plain" }));
    const link = el("a", { href: url, download: source + ".log" });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }

  // ---------------------------------------------------------------- live status

  function onStatus(data) {
    if (!data || typeof data !== "object" || !data.backend) return;
    status = data;
    render();
    loadPlan(false);
    startLog(false);
  }

  function listen() {
    if (events) events.close();
    events = stream("/deploy/events", onStatus, () => {
      events = null;
      setTimeout(listen, 2000);   // the proxy restarted or the connection dropped
    });
  }

  // ---------------------------------------------------------------- actions

  function deploy() {
    const p = selectedProfile();
    const row = chosenRow();
    if (!p || !row) return;
    const body = Object.assign({ profile_id: p.id }, overrides());
    if (!row.fits && !body.force) {
      fail(new Error("The estimate says " + row.variant + " does not fit: " + row.reason + ". Tick “Deploy anyway” to try regardless."));
      return;
    }
    const dep = status.deployment;
    if (backendState() === "ready" && dep && (dep.profile_id !== p.id || dep.plan.variant !== row.variant || body.context || body.tensor_parallel)) {
      const ok = confirm("vLLM is serving " + (dep.label || dep.profile_name) + ".\n\nDeploying " + p.name + " (" + row.variant +
        ") stops it first: requests in flight will be cut and the proxy answers 503 until the new deployment is ready.\n\nContinue?");
      if (!ok) return;
    }
    act(async () => { onStatus(await api("POST", "/deploy", body)); $("progress").scrollIntoView({ behavior: "smooth", block: "start" }); });
  }

  boot("/admin/pod", () => {
    const input = $("ssh-command");
    input.addEventListener("input", () => { input.dataset.dirty = "1"; });
    input.addEventListener("change", () => { if (input.dataset.dirty) saveConnection(); });
    input.addEventListener("paste", () => setTimeout(() => { if (input.dataset.dirty) saveConnection(); }, 0));
    $("conn-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      act(async () => {
        if (input.dataset.dirty || !(status && status.pod)) { if (!(await saveConnection())) return; }
        await connect(null);
      });
    });
    $("fp-accept").addEventListener("click", () => act(() => connect(pendingFingerprint)));
    $("fp-cancel").addEventListener("click", () => { pendingFingerprint = null; show($("fingerprint-box"), false); });
    $("disconnect-btn").addEventListener("click", () => {
      const serving = backendState() === "ready" ? "\n\nThe proxy will answer 503 until you connect again; vLLM keeps running on the pod." : "";
      if (confirm("Close the SSH connection? The pod and everything on it keep running (and billing)." + serving)) {
        act(async () => onStatus(await api("POST", "/pod/disconnect")));
      }
    });

    $("hf-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const value = $("hf-token").value.trim();
      if (!value) return;
      act(async () => { await api("PUT", "/settings/hf-token", { token: value }); $("hf-token").value = ""; status.hf_token_set = true; loadPlan(true); loadSizing(); });
    });
    $("hf-remove").addEventListener("click", () => {
      if (confirm("Remove the stored Hugging Face token? Gated repositories will not download without it.")) {
        act(async () => { await api("DELETE", "/settings/hf-token"); status.hf_token_set = false; loadPlan(true); loadSizing(); });
      }
    });

    $("profile-select").addEventListener("change", () => { show($("profile-form"), false); renderProfileSummary(); loadPlan(true); loadSizing(); });
    $("profile-new").addEventListener("click", () => openEditor({ name: "", data: JSON.parse(JSON.stringify(template)) }, "New profile", null));
    $("profile-dup").addEventListener("click", () => {
      const p = selectedProfile();
      if (p) openEditor({ name: p.name + "-copy", data: JSON.parse(JSON.stringify(p.data)) }, "Duplicate of " + p.name, null);
    });
    $("profile-edit").addEventListener("click", () => { const p = selectedProfile(); if (p) openEditor(p, "Edit " + p.name, p.id); });
    $("profile-del").addEventListener("click", () => {
      const p = selectedProfile();
      if (p && confirm("Delete the profile '" + p.name + "'? Deployments made with it keep their own copy.")) {
        act(async () => { await api("DELETE", "/profiles/" + p.id); $("profile-select").value = ""; await loadProfiles(); loadPlan(true); loadSizing(); });
      }
    });
    $("profile-form").addEventListener("submit", saveProfile);
    $("pf-cancel").addEventListener("click", () => show($("profile-form"), false));
    $("pf-add-variant").addEventListener("click", () => $("pf-variants").append(variantRow({ label: "", repo: "", revision: "", dir: "", args: [] })));

    for (const id of ["ov-context", "ov-tp", "ov-force"]) $(id).addEventListener("input", renderCommand);
    $("deploy-btn").addEventListener("click", deploy);
    $("cancel-btn").addEventListener("click", () => {
      if (confirm("Cancel the deployment? The step that is running is stopped and the remote process it started is killed.")) {
        act(async () => onStatus(await api("POST", "/deploy/cancel")));
      }
    });
    $("stop-btn").addEventListener("click", () => {
      if (confirm("Stop vLLM on the pod? Requests in flight are cut and the proxy answers 503.\n\nThe pod itself keeps running and is still billed.")) {
        act(async () => onStatus(await api("POST", "/vllm/stop")));
      }
    });
    $("restart-btn").addEventListener("click", () => {
      if (confirm("Restart vLLM with the last deployment's profile and plan? Nothing is downloaded again; requests in flight are cut.")) {
        act(async () => onStatus(await api("POST", "/vllm/restart", {})));
      }
    });
    $("dep-offer-btn").addEventListener("click", () => {
      const context = Number($("dep-offer-btn").dataset.context);
      act(async () => onStatus(await api("POST", "/vllm/restart", { context })));
    });
    $("delete-btn").addEventListener("click", () => {
      if (!confirm("Delete the downloaded model files of all profiles from the pod? vLLM must be stopped; the next Deploy downloads them again.")) return;
      act(async () => {
        const data = await api("POST", "/pod/delete-models", {});
        const freed = data.deleted.reduce((sum, d) => sum + d.bytes, 0);
        alert(data.deleted.length ? "Deleted " + data.deleted.map((d) => d.dir).join(", ") + " (" + gb(freed) + " on disk)." : "There were no model files to delete.");
        loadPlan(true);
      });
    });

    $("log-source").addEventListener("change", () => startLog(true));
    $("log-pause").addEventListener("click", () => {
      logPaused = !logPaused;
      $("log-pause").textContent = logPaused ? "Resume" : "Pause";
      $("log-pause").setAttribute("aria-pressed", String(logPaused));
      if (!logPaused) { const held = logHeld; logHeld = []; $("log-info").textContent = ""; pushLog(held); }
    });
    $("log-autoscroll").addEventListener("change", () => { if ($("log-autoscroll").checked) scheduleLog(); });
    $("log-download").addEventListener("click", () => act(downloadLog));
    $("catalog-save").addEventListener("click", () => saveCatalog(false));
    $("catalog-reset").addEventListener("click", () => { if (confirm("Replace the catalog with the built-in list?")) saveCatalog(true); });
    $("catalog-details").addEventListener("toggle", () => { if ($("catalog-details").open) loadCatalog().catch(fail); });

    setInterval(() => { if (status && status.deployment && status.deployment.state === "running") renderDeployment(); }, 1000);

    (async () => {
      try {
        status = await api("GET", "/deploy/status");
        await loadProfiles();
        render();
        loadPlan(true);
        loadSizing();
        startLog(true);
        listen();
      } catch (err) { fail(err); }
    })();
  });
})();
