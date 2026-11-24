(() => {
  const el = id => document.getElementById(id);
  const select = el("previous-wedge");
  let current;
  async function api(path, method = "GET", data) {
    const options = { method };
    if (data) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(data);
    }
    const response = await fetch("/api/opacity-wedges" + path, options);
    const body = await response.json();
    if (!response.ok) throw Error(typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    return body;
  }

  function fields(text) {
    return text.trim() ? text.trim().split(/\n/).map(line => line.trim().split(/[\s,]+/)) : [];
  }
  function scans(text) {
    const pairs = fields(text);
    if (!pairs.length) return null;
    if (pairs.length !== 5 || pairs.some(pair => pair.length !== 2))
      throw Error("Enter five TD and hex pairs, one per line.");
    return pairs.map(([td, hex]) => ({ td_mm: Number(td), hex }));
  }
  function interval([lo, hi]) {
    if (lo === 0 && hi === null) return "unbounded";
    if (hi === null) return "> " + lo.toPrecision(4);
    if (lo === 0) return "≤ " + hi.toPrecision(4);
    return lo.toPrecision(4) + "–" + hi.toPrecision(4);
  }
  function show(w) {
    current = w.id;
    if (!Array.from(select.options).some(option => option.value === w.id)) {
      const label = w.top.spool_id + " over " + w.backing.spool_id + " · " + w.id.slice(0, 8);
      select.add(new Option(label, w.id), 1);
    }
    select.value = w.id;
    el("history-picker").hidden = false;
    const count = select.options.length - 1;
    el("history-count").textContent = count + " recent test print" + (count === 1 ? "." : "s.");
    history.replaceState(null, "", "/opacity-wedges?wedge=" + encodeURIComponent(w.id));
    el("wedge-detail").hidden = false;
    el("pair-title").textContent = w.top.name + " (" + w.top.spool_id + ") over " + w.backing.name + " (" + w.backing.spool_id + ")";
    el("divisor").textContent = w.top.hex + " (" + (w.top.color_level || "legacy") + ") over " + w.backing.hex + " (" + (w.backing.color_level || "legacy") + ") · A = " + w.divisor.toFixed(5) + " · 31 positions including bare backing";
    const backingLayers = Math.round(w.swap_z_mm / w.layer_mm);
    el("print-estimate").textContent = "Slice the wedge with your printer settings to estimate material and time.";
    el("swap-instructions").textContent = "Change filament after layer " + backingLayers
      + " (" + w.swap_z_mm.toFixed(2) + " mm), before layer " + (backingLayers + 1)
      + " at Z=" + (w.swap_z_mm + w.layer_mm).toFixed(2) + " mm. Compare the backing and "
      + w.rail_mm.toFixed(2) + " mm rail with thicker controls of their respective spools."
      + (w.backing_target_capped ? " The backing reached the 4 mm ceiling before its opacity target. Choose a different backing if the control differs." : "");
    el("wedge-download").href = "/api/opacity-wedges/" + w.id + "/parts.zip";
    el("wedge-reading").reset();
    for (const input of el("wedge-reading").querySelectorAll(".pass")) {
      input.hidden = false;
      input.nextElementSibling.hidden = true;
    }
    el("wedge-reading").hidden = Boolean(w.measurement);
    el("wedge-result").hidden = !w.measurement;
    if (w.measurement) {
      const m = w.measurement;
      let summary = "Library opacity " + w.top.library_td_mm + " mm (" + (w.top.opacity_level || "legacy") + ") · Crossing " + interval(m.crossing_mm) + " mm · Hiding distance " + interval(m.hiding_mm) + " mm";
      if (m.scale) {
        summary += " · Median meter TD " + m.median_td_mm + " mm · Scale " + interval(m.scale);
        summary += " (including meter uncertainty " + interval(m.scale_with_accuracy) + ")";
      }
      summary += m.ranking_hiding_mm === null ? " · Retained as a limit or flagged observation." : " · Ranking hiding distance " + m.ranking_hiding_mm.toPrecision(4) + " mm.";
      el("intervals").textContent = summary;
      el("raw-result").textContent = JSON.stringify(w, null, 2);
    }
    el("pair-title").focus();
    el("wedge-status").textContent = w.measurement ? "Saved " + w.measured_at : "Wedge ready.";
  }

  function submit(form, handler) {
    form.addEventListener("submit", async event => {
      event.preventDefault();
      const button = event.submitter;
      const errorLine = form.querySelector('[role="alert"]');
      errorLine.textContent = "";
      button.disabled = true;
      try { await handler(new FormData(form)); }
      catch (error) { errorLine.textContent = error.message; }
      finally {
        button.disabled = false;
        if (document.activeElement === document.body) button.focus();
      }
    });
  }
  for (const input of el("wedge-reading").querySelectorAll(".pass")) {
    const reveal = input.nextElementSibling;
    input.addEventListener("change", () => {
      input.hidden = Boolean(input.value);
      reveal.hidden = !input.value;
      if (input.value) reveal.focus();
    });
    reveal.addEventListener("click", () => {
      reveal.hidden = true;
      input.hidden = false;
      input.focus();
    });
  }
  submit(el("generate-wedge"), async f => {
    const spool = role => ({ filament_id: Number(f.get(role)), spool_id: f.get(role + "_lot"), scans: scans(f.get(role + "_scans")) });
    const w = await api("", "POST", { top: spool("top"), backing: spool("backing") });
    if (w) show(w);
  });
  async function openRecord(id) {
    current = null;
    el("wedge-status").textContent = "";
    el("wedge-detail").hidden = true;
    if (!id) {
      history.replaceState(null, "", "/opacity-wedges");
      return;
    }
    current = id;
    try { const w = await api("/" + encodeURIComponent(id)); if (current === id) show(w); }
    catch (error) { el("wedge-status").textContent = error.message; }
  }
  select.addEventListener("change", () => openRecord(select.value));
  const linked = new URLSearchParams(location.search).get("wedge");
  if (linked) openRecord(linked);

  submit(el("wedge-reading"), async f => {
    const measured = {};
    for (const pair of fields(f.get("heights"))) {
      if (pair.length !== 2 || pair[0] in measured) throw Error("Enter each measured patch once with its height.");
      measured[pair[0]] = Number(pair[1]);
    }
    const id = current;
    const w = await api("/" + id + "/reading", "PUT", {
      passes: [1,2,3].map(i => f.get("pass" + i) === "none" ? null : Number(f.get("pass" + i))),
      top_scans: scans(f.get("late_scans")), measured_heights_mm: measured,
      conditions: f.get("conditions"), controls_passed: f.has("controls")
    });
    if (w && current === id) show(w);
  });
})();
