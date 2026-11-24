// Keep the native file controls and checkboxes mounted while changing search scope.
(() => {
  const form = document.querySelector("#suggestion-form");
  if (!form) return;

  const get = (id) => document.getElementById(id);
  const links = [...form.querySelectorAll("[data-mode]")];
  const spools = [...form.querySelectorAll("input[name=filament_ids]")];
  const inventory = form.elements.inventory;
  const slots = form.elements.slots;
  const submit = form.querySelector("[type=submit]");
  const status = get("comparison-status");
  const output = get("comparison-output");
  const filter = get("spool-filter");
  let mode = form.action.split("/").pop();
  let pending;

  function clearResult() {
    pending?.abort();
    pending = null;
    submit.disabled = false;
    output.replaceChildren();
    status.textContent = "";
  }

  function updateSelection() {
    const chosen = spools.filter(input => input.checked);
    const count = chosen.length;
    const uploaded = inventory.files.length > 0;
    const full = mode === "full-set";
    const settings = get("comparison-settings").dataset;
    const report = mode === "loadouts" && !count ? settings.ownedReport : settings.catalogueReport;
    showInventoryMaterial(report === "true" || (!full && uploaded));
    get("spool-list").disabled = uploaded;
    get("filter-field").hidden = uploaded;
    get("spool-list").hidden = uploaded;
    get("selection-summary").hidden = uploaded;
    get("clear-inventory").hidden = !uploaded;
    get("selected-count").textContent = count + (count === 1 ? " spool selected" : " spools selected");
    const tags = chosen.map(input => {
      const tag = document.createElement("li");
      tag.className = "max-w-full wrap-anywhere rounded-md border border-border bg-surface-2 px-2 py-1 text-xs";
      tag.append(input.closest("label").querySelector("[data-spool-tag]").cloneNode(true));
      return tag;
    });
    get("selected-spools").replaceChildren(...tags);
    const parked = uploaded ? inventory.files[0].name : count + (count === 1 ? " selected spool" : " selected spools");
    get("parked-spools").textContent = full && (uploaded || count)
      ? "Switch back to use your selection again (" + parked + ")." : "";
    let hint = "";
    if (!full && uploaded) {
      hint = mode === "next" ? "Using the file as your loaded spools." : "Using owned spools from the file.";
    } else if (mode === "next" && count > 7) {
      hint = "All your choices are still selected. Remove " + (count - 7) + " to compare an addition.";
    } else if (mode === "loadouts" && count && Number(slots.value) > count) {
      hint = "Choose more spools or reduce Spools per set to " + count + " or fewer.";
    }
    get("selection-hint").textContent = hint;
    slots.setCustomValidity(mode === "loadouts" && !uploaded && count && Number(slots.value) > count ? hint : "");
  }

  function changeMode(next) {
    mode = next;
    clearResult();
    form.action = "/suggestions/" + mode;
    for (const link of links) {
      const active = link.dataset.mode === mode;
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    const full = mode === "full-set";
    get("spool-fields").hidden = get("spool-fields").disabled = full;
    get("library-scope").hidden = !full;
    get("slot-field").hidden = get("slot-field").disabled = mode === "next";
    get("spool-title").textContent = mode === "next" ? "Spools already loaded" : "Spools to choose from";
    form.querySelectorAll("[data-help]").forEach(el => { el.hidden = el.dataset.help !== mode; });
    submit.textContent = mode === "next" ? "Compare additions" : "Compare sets";
    updateSelection();
  }

  links.forEach(link => link.addEventListener("click", event => {
    if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    event.preventDefault();
    if (link.dataset.mode !== mode) {
      history.pushState(null, "", link.href);
      changeMode(link.dataset.mode);
    }
  }));
  window.addEventListener("popstate", () => changeMode(new URL(location.href).searchParams.get("mode") || "next"));

  filter.addEventListener("input", () => {
    const query = filter.value.trim().toLowerCase();
    let matches = 0;
    for (const spool of spools) {
      const row = spool.closest("label");
      row.hidden = !row.textContent.toLowerCase().includes(query);
      if (!row.hidden) matches++;
    }
    get("no-matches").hidden = !spools.length || matches > 0;
  });
  filter.addEventListener("keydown", event => {
    if (event.key === "Enter") event.preventDefault();
  });
  get("clear-inventory").addEventListener("click", () => {
    inventory.value = "";
    clearResult();
    updateSelection();
    filter.focus();
  });
  form.addEventListener("invalid", event => {
    const firstInvalid = form.querySelector("input:invalid, select:invalid");
    if (event.target !== firstInvalid) return;
    if (firstInvalid === form.elements.image) get("image-help").hidden = false;
    firstInvalid.scrollIntoView({block: "center"});
  }, true);
  form.addEventListener("input", event => {
    if (event.target === form.elements.image) get("image-help").hidden = true;
    if (event.target !== filter) {
      clearResult();
      updateSelection();
    }
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (pending) return;
    const data = new FormData(form);
    const count = data.getAll("filament_ids").length;
    if (mode === "next" && !inventory.files.length && (count < 1 || count > 7)) {
      status.textContent = "Choose 1–7 loaded spools or upload their inventory file.";
      filter.focus();
      return;
    }
    const request = new AbortController();
    pending = request;
    submit.disabled = true;
    status.textContent = "Comparing predicted prints…";
    output.replaceChildren();
    try {
      const response = await fetch(form.action, {
        method: "POST", body: data, signal: request.signal, headers: {"X-Requested-With": "fetch"}
      });
      const body = await response.text();
      if (pending !== request) return;
      if (!response.ok && response.headers.get("Content-Type")?.startsWith("text/plain")) {
        status.textContent = "Comparison could not finish: " + body + ". Check your inputs and try again.";
        return;
      }
      if (!response.ok) throw new Error("Comparison unavailable");
      const sheet = new DOMParser().parseFromString(body, "text/html").getElementById("suggestion-sheet");
      if (!sheet) throw new Error("Missing comparison");
      output.replaceChildren(sheet);
      status.textContent = "Comparison ready.";
      const title = get("sheet-title");
      title.focus();
      title.scrollIntoView({block: "center"});
    } catch (error) {
      if (pending === request) status.textContent = "Comparison unavailable. Your choices are still here. Try again.";
    } finally {
      if (pending === request) {
        pending = null;
        submit.disabled = false;
      }
    }
  });
  updateSelection();
})();
