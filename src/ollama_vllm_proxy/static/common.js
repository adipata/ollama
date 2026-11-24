// Shared by the management pages: admin-token handling, API helper, header.
// The token is taken from the URL fragment (never sent to the server) or a
// prompt, and is kept only in this tab's sessionStorage.
(function () {
  "use strict";
  const KEY = "ovp-admin-token";
  let token = "";

  function takeFragment() {
    const frag = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (frag) {
      try { sessionStorage.setItem(KEY, frag); } catch (e) { /* private mode */ }
      history.replaceState(null, "", location.pathname + location.search);
      return frag;
    }
    try { return sessionStorage.getItem(KEY) || ""; } catch (e) { return ""; }
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of [].concat(children || [])) {
      if (child == null) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  async function api(method, path, body) {
    const opts = { method, headers: { Authorization: "Bearer " + token } };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch("/admin/api" + path, opts);
    if (resp.status === 401) { logout(); throw new Error("The admin token was not accepted."); }
    let data = null;
    try { data = await resp.json(); } catch (e) { /* empty body */ }
    if (!resp.ok) throw new Error((data && (data.error || data.detail)) || ("Request failed (" + resp.status + ")"));
    return data;
  }

  // Raw authenticated GET for downloads; returns the Response.
  function rawFetch(path) {
    return fetch("/admin/api" + path, { headers: { Authorization: "Bearer " + token }, cache: "no-store" });
  }

  // Streaming GET (server-sent events over fetch, so the token stays in a header).
  // onEvent receives each parsed "data:" payload. Returns an object with close().
  function stream(path, onEvent, onEnd) {
    const ctl = new AbortController();
    (async () => {
      try {
        const resp = await fetch("/admin/api" + path, { headers: { Authorization: "Bearer " + token }, signal: ctl.signal });
        if (!resp.ok || !resp.body) throw new Error("stream failed (" + resp.status + ")");
        const reader = resp.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\n\n")) >= 0) {
            const event = buf.slice(0, idx);
            buf = buf.slice(idx + 2);
            const data = event.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).replace(/^ /, "")).join("\n");
            if (!data) continue;
            let parsed = data;
            try { parsed = JSON.parse(data); } catch (e) { /* plain text */ }
            onEvent(parsed);
          }
        }
        if (onEnd) onEnd(null);
      } catch (err) {
        if (err.name !== "AbortError" && onEnd) onEnd(err);
      }
    })();
    return { close: () => ctl.abort() };
  }

  function logout() {
    try { sessionStorage.removeItem(KEY); } catch (e) { /* ignore */ }
    location.reload();
  }

  function fmtTime(ts) {
    if (!ts) return "";
    return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  function header(current) {
    const pages = [["/admin", "API keys"], ["/admin/pod", "Pod"], ["/admin/monitor", "Monitoring"]];
    const state = el("span", { class: "state", id: "backend-state", text: "" });
    const top = el("header", { class: "top" }, [
      el("span", { class: "brand", text: "Ollama → vLLM proxy" }),
      el("nav", { "aria-label": "Sections" }, pages.map(([href, label]) =>
        el("a", { href, "aria-current": href === current ? "page" : false, text: label }))),
      el("span", { class: "spacer" }),
      state,
      el("button", { class: "link", type: "button", text: "Sign out", onclick: logout }),
    ]);
    document.body.prepend(top);
    const refresh = async () => {
      try {
        const s = await api("GET", "/status");
        const b = s.backend;
        state.textContent = "Backend: " + b.state + (b.served_model && b.state === "ready" ? " · " + b.served_model : "");
      } catch (e) { /* shown elsewhere */ }
    };
    refresh();
    setInterval(refresh, 5000);
  }

  // Shows a token prompt until a token is available, then calls start().
  function boot(current, start) {
    token = takeFragment();
    const login = document.getElementById("login");
    const main = document.querySelector("main");
    if (!token) {
      main.classList.add("hidden");
      login.classList.remove("hidden");
      login.querySelector("form").addEventListener("submit", (ev) => {
        ev.preventDefault();
        const value = login.querySelector("input").value.trim();
        if (!value) return;
        try { sessionStorage.setItem(KEY, value); } catch (e) { /* ignore */ }
        location.reload();
      });
      return;
    }
    login.classList.add("hidden");
    main.classList.remove("hidden");
    header(current);
    start();
  }

  window.OVP = { api, fetch: rawFetch, stream, el, boot, fmtTime, logout };
})();
