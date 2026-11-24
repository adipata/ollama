(function () {
  "use strict";
  const { api, el, boot, fmtTime } = window.OVP;
  const $ = (id) => document.getElementById(id);

  function fail(err) {
    const box = $("message");
    box.textContent = err.message || String(err);
    box.classList.remove("hidden");
  }
  function clearError() { $("message").classList.add("hidden"); }

  function statusPill(key) {
    const cls = key.status === "active" ? "good" : key.status === "disabled" ? "warn" : "bad";
    return el("span", { class: "pill " + cls, text: key.status });
  }

  async function act(fn) {
    clearError();
    try { await fn(); await load(); } catch (err) { fail(err); }
  }

  function actions(key) {
    if (key.status === "revoked") return el("span", { class: "muted", text: "revoked " + fmtTime(key.revoked_at) });
    return el("div", { class: "row" }, [
      el("button", { type: "button", text: "Rename", onclick: () => {
        const name = prompt("New name for this key", key.name);
        if (name && name !== key.name) act(() => api("PATCH", "/keys/" + key.id, { name }));
      } }),
      el("button", { type: "button", text: key.enabled ? "Disable" : "Enable",
        onclick: () => act(() => api("PATCH", "/keys/" + key.id, { enabled: !key.enabled })) }),
      el("button", { type: "button", class: "danger", text: "Revoke", onclick: () => {
        if (confirm("Revoke '" + key.name + "' permanently? Clients using it will be rejected.")) {
          act(() => api("DELETE", "/keys/" + key.id));
        }
      } }),
    ]);
  }

  async function load() {
    const data = await api("GET", "/keys");
    const body = $("keys");
    body.replaceChildren(...data.keys.map((key) => el("tr", {}, [
      el("td", { text: key.name }),
      el("td", {}, el("code", { text: key.prefix + "…" })),
      el("td", {}, statusPill(key)),
      el("td", { text: fmtTime(key.created_at) }),
      el("td", {}, actions(key)),
    ])));
    $("empty").classList.toggle("hidden", data.keys.length > 0);
  }

  function showCreated(key) {
    const base = location.origin;
    $("created-key").textContent = key.key;
    $("snippet").textContent =
      "# OpenAI SDK (Python)\n" +
      "from openai import OpenAI\n" +
      "client = OpenAI(base_url=\"" + base + "/v1\", api_key=\"" + key.key + "\")\n\n" +
      "# ollama-python\n" +
      "from ollama import Client\n" +
      "client = Client(host=\"" + base + "\", headers={\"Authorization\": \"Bearer " + key.key + "\"})\n\n" +
      "# curl\n" +
      "curl " + base + "/api/tags -H \"Authorization: Bearer " + key.key + "\"";
    $("created").classList.remove("hidden");
    $("created").scrollIntoView({ block: "nearest" });
  }

  boot("/admin", () => {
    $("create").addEventListener("submit", async (ev) => {
      ev.preventDefault();
      clearError();
      try {
        const key = await api("POST", "/keys", { name: $("key-name").value.trim() });
        $("key-name").value = "";
        showCreated(key);
        await load();
      } catch (err) { fail(err); }
    });
    $("copy-key").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText($("created-key").textContent);
        $("copy-key").textContent = "Copied";
      } catch (e) { fail(new Error("Copy failed; select the key and copy it manually.")); }
    });
    $("dismiss-key").addEventListener("click", () => {
      $("created-key").textContent = "";
      $("snippet").textContent = "";
      $("copy-key").textContent = "Copy";
      $("created").classList.add("hidden");
    });
    load().catch(fail);
  });
})();
