import { bindRegions } from "./flat-regions.js";
import { bindProjectIo } from "./project-io.js";
import { bindAccuracyReport } from "./report.js";
import { readBoot } from "./state.js";

const boot = readBoot();
const el = id => document.getElementById(id);
const selectors = Array.from(document.querySelectorAll(".flat-slot"));
const profile = el("flat-profile");
const inventory = boot.inventory;
const notice = message => { el("flat-status").textContent = message; };
let state = null;
let selectedPlan = "full";
let pending = null;
let matrix = null;
let sliceReport = null;
let source = {};
let editProjectId = null;
let ditherScope = null;
let trial = null;
const regions = bindRegions(() => {
  selectedPlan = "full";
  ditherScope = null;
  invalidate();
  notice("Generate a preview with these region palettes.");
}, notice);
document.body.dataset.projectType = "flat";

const invalidateReport = bindAccuracyReport(() => ({ state, factor: 1 }));
function invalidate() {
  trial = null;
  el("dither-trial").hidden = true;
  regions.busy(false);
  pending?.abort();
  pending = null;
  el("flat-inputs").disabled = false;
  el("flat-result").hidden = true;
  el("studio-save").disabled = true;
  invalidateReport();
}

const material = document.querySelector('[name="material_group"]');
function setSpools(saved, savedOnly = false) {
  const group = material.value === "auto" ? boot.default_material : material.value;
  const eligible = savedOnly ? [] : inventory.filter(s => group === "mixed" || boot.spool_materials[s.id] === group);
  selectors.forEach((select, i) => {
    select.replaceChildren();
    const choices = saved ? [saved[i], ...eligible.filter(s => JSON.stringify(s) !== JSON.stringify(saved[i]))] : eligible;
    for (const spool of choices) {
      const label = [spool.brand, spool.material, spool.name, spool.spool_id].filter(Boolean).join(" ");
      const { input_distance_mm, input_distance_source, hiding_distance_mm, color_source, ...fields } = spool;
      const td = input_distance_mm ?? fields.td;
      const option = new Option(label + " · " + spool.hex + " · TD " + td, JSON.stringify({ ...fields, td }));
      select.add(option);
    }
    if (!saved) {
      select.selectedIndex = Math.min(i, choices.length - 1);
    }
  });
}

function draw(run, canvas = el("flat-canvas")) {
  canvas.width = run.geometry.columns;
  canvas.height = run.geometry.rows;
  const ctx = canvas.getContext("2d");
  const pixels = ctx.createImageData(canvas.width, canvas.height);
  let offset = 0;
  for (const row of run.cell_recipe_ids) {
    for (const id of row) {
      pixels.data.set(run.recipe_table[id].rgb, offset);
      pixels.data[offset + 3] = 255;
      offset += 4;
    }
  }
  ctx.putImageData(pixels, 0, 0);
}

function planLabel(plan) {
  if (plan === "full") return "All four slots available";
  const pair = /^(middle|all)-(\d+)-(\d+)$/.exec(plan);
  if (!pair) return plan;
  const layers = pair[1] === "middle" ? "middle layers" : "all picture layers";
  return "Slots " + pair[2] + " and " + pair[3] + " only in " + layers;
}

function costText(plan) {
  const range = plan.minutes_range?.map(v => v.toFixed(1)).join("–");
  const kind = plan.time_kind || "saved estimate (previous model)";
  return kind + " · " + plan.change_count + " changes · " + (plan.minutes == null ? "time unavailable" : plan.minutes.toFixed(1) + " min") + (range ? " (planning range " + range + " min)" : "") + " · " +
    (plan.waste_g == null ? "waste unavailable" : plan.waste_g.toFixed(2) + " g waste") + " · cell colour error (average / 95th percentile) " +
    plan.mean_delta_e.toFixed(2) + " / " + plan.p95_delta_e.toFixed(2) + " ΔE00";
}

function detailWarning(run) {
  const detail = run.printability;
  if (!detail) return "";
  const messages = [];
  if (run.regions.empty_regions.length) messages.push("Empty or covered regions: " + run.regions.empty_regions.join(", ") + ".");
  if (detail.isolated_layer_cells || detail.thin_layer_cells) {
    messages.push(detail.isolated_layer_cells + " isolated and " + detail.thin_layer_cells +
      " one-cell-wide positions across picture layers. A bead can miss a cell centre. Check these in the slice before printing.");
  }
  return messages.join(" ");
}

function tradeText(run) {
  const d = run.dithering;
  if (!d) return "Dithering is off.";
  const change = d.dithered_changes - d.direct_changes;
  return "Mean ΔE00 at one-cell blur, direct " + d.direct_error.toFixed(2) + " → dithered " +
    d.dithered_error.toFixed(2) + ". Planned filament changes " + d.direct_changes + " → " +
    d.dithered_changes + " (" + (change > 0 ? "+" : "") + change + ")." +
    (d.dithered_error >= d.direct_error ? " This setting gives no predicted improvement." : "");
}

function showState(next) {
  invalidate();
  state = next;
  selectedPlan = state.flat.run.selected_plan;
  source = state.image;
  if (source.project_id) editProjectId = source.project_id;
  ditherScope = state.options.dither_scope ?? null;
  regions.load(state);
  sliceReport = state.options.slice_report || null;
  el("flat-slice").value = "";
  matrix = state.options.slicer_settings || null;
  el("flat-settings").value = "";
  el("flat-image").value = "";
  profile.value = state.options.profile;
  el("a1-settings").hidden = profile.value !== "a1";
  el("flat-edge").value = state.options.long_edge_cells;
  el("flat-pitch").value = state.options.cell_mm;
  el("slot-help").hidden = true;
  const savedGroups = new Set(state.palette.filaments.map(spool => {
    const row = inventory.find(s => s.id === spool.id && s.material === spool.material && s.brand === spool.brand);
    return row && boot.spool_materials[row.id];
  }));
  const savedGroup = savedGroups.size === 1 ? [...savedGroups][0] : null;
  const selected = state.material_group || savedGroup;
  const note = el("material-note");
  let savedOnly = selected !== boot.default_material;
  if (note) {
    material.value = selected === boot.default_material ? "auto" : selected || "auto";
    savedOnly = !selected || !material.value;
    if (savedOnly) {
      material.value = "auto";
      note.textContent = "Keeping the saved slots. Choose a material to replace them.";
    } else {
      note.textContent = material.value === "mixed" ? material.dataset.mixed :
        material.value === "auto" ? material.dataset.auto : "Using " + material.value + " from this print.";
    }
  }
  setSpools(state.palette.filaments, savedOnly);
  const { run, cost, downloads, run_id, printer_settings_note } = state.flat;
  draw(run);
  const g = run.geometry;
  notice(source.filename + " · " + g.width_mm + " × " + g.height_mm + " mm · " + planLabel(run.selected_plan));
  el("flat-detail-warning").textContent = detailWarning(run);
  el("dither-summary").textContent = tradeText(run);
  el("direct-match").hidden = ditherScope === null;
  const scopes = [new Option("Whole plate", "-1"), new Option("Background", "0"),
    ...(state.options.regions || []).map((r, i) => new Option(r.name, String(i + 1)))];
  el("dither-scope").replaceChildren(...scopes);
  el("dither-scope").value = String(ditherScope ?? -1);
  el("flat-selected").textContent = costText(cost.selected);
  const budget = cost.budget_100mm;
  const exceeded = cost.selected.over_time || cost.selected.over_waste;
  el("flat-budget").textContent = !cost.selected.time_kind ? "Generate a preview to check this saved plan against the calibrated budget." : (exceeded ? "Over budget" : cost.selected.over_time === null || cost.selected.over_waste === null ? "Budget comparison incomplete" : "Within budget") +
    " using " + (cost.selected.time_kind || "saved estimate") + " · 100 × 100 mm reference: " + budget.minutes + " min and " + budget.waste_g + " g.";
  el("flat-basis").textContent = cost.basis + (cost.selected.minutes_range ?
    " Range covers the calibration fixtures. Dense detail, different presets and physical delays can fall outside it." : "");
  const alt = cost.selected.time_kind && cost.two_slot_alternative;
  el("accept-plan").hidden = !alt;
  el("flat-alternative").textContent = alt ? "Alternative: " + planLabel(alt.plan) + " · " + costText(alt) : "";
  const links = el("flat-downloads");
  links.replaceChildren();
  const artifacts = [["Print geometry (3MF)", "geometry.3mf"], ["Slot map", "expected.json"],
                     ["Run record", "run.json"], ["Preview PNG", "preview.png"]];
  if (downloads.printer_project) artifacts.push(["Printer project with settings", "flat.3mf"]);
  for (const [title, name] of artifacts) {
    const link = document.createElement("a");
    link.className = "btn btn-quiet";
    link.href = "/artifacts/" + encodeURIComponent(run_id) + "/" + name + "?download=1";
    link.textContent = title;
    links.append(link);
  }
  el("settings-note").textContent = printer_settings_note;
  el("flat-result").hidden = false;
  el("studio-save").disabled = false;
  el("studio-report").disabled = false;
}

async function generate(plan, scope = ditherScope, compare = false) {
  const file = el("flat-image").files[0];
  if (!file && !source.project_id && !source.upload_token) {
    notice("Choose a source image.");
    return;
  }
  invalidate();
  const controller = new AbortController();
  pending = controller;
  regions.busy(true);
  el("flat-inputs").disabled = true;
  notice("Matching cells and preparing the print files…");
  try {
    const options = { profile: profile.value, long_edge_cells: Number(el("flat-edge").value),
                      cell_mm: Number(el("flat-pitch").value), plan, ...regions.options() };
    if (scope !== null) options.dither_scope = scope;
    if (profile.value === "a1") {
      const settings = el("flat-settings").files[0];
      if (settings) {
        if (settings.size > 4 * 1024 * 1024) throw new Error("Settings must fit in 4 MiB.");
        const parsed = JSON.parse(await settings.text());
        const keys = ["printer_model", "filament_colour", "filament_density", "flush_multiplier", "flush_volumes_matrix"];
        options.slicer_settings = Object.fromEntries(keys.map(k => [k, parsed[k]]));
      } else if (matrix) options.slicer_settings = matrix;
    }
    const slice = el("flat-slice").files[0];
    if (slice) {
      if (slice.size > 65536) throw new Error("Slice report must fit in 64 KiB.");
      options.slice_report = JSON.parse(await slice.text());
    } else if (sliceReport) options.slice_report = sliceReport;
    const form = new FormData();
    const profileFile = el("flat-project-settings").files[0];
    if (profileFile) form.append("project_settings", profileFile);
    if (file) form.append("image", file);
    form.append("source", JSON.stringify(source));
    form.append("material_group", material.value);
    form.append("palette", JSON.stringify({ filaments: selectors.map(s => JSON.parse(s.value)) }));
    form.append("options", JSON.stringify(options));
    const response = await fetch("/studio/flat/run", { method: "POST", body: form, signal: controller.signal });
    const next = await response.json();
    if (!response.ok) throw new Error(next.detail || "Flat run failed");
    if (pending !== controller) return;
    if (compare) {
      trial = next;
      const run = next.flat.run;
      draw(run, el("trial-canvas"));
      el("trial-trade").textContent = tradeText(run);
      el("trial-warning").textContent = detailWarning(run);
      el("trial-cost").textContent = costText(next.flat.cost.selected);
      el("dither-trial").hidden = false;
      el("trial-title").focus();
      notice("Compare the trial before accepting it.");
    } else {
      showState(next);
      el("flat-preview-title").focus();
    }
  } catch (err) {
    if (pending === controller) {
      if (compare) finishComparison(state);
      notice("Could not make the preview. " + (err instanceof SyntaxError
        ? "Check that your settings file contains valid JSON." : err.message));
    }
  } finally {
    if (pending === controller) {
      pending = null;
      regions.busy(false);
      el("flat-inputs").disabled = false;
    }
  }
}

el("flat-form").addEventListener("input", event => {
  selectedPlan = "full";
  ditherScope = null;
  if (event.target === el("flat-image")) {
    regions.clear();
    sliceReport = null;
    el("flat-slice").value = "";
  }
  invalidate();
  if (event.target === material) setSpools(null);
  if (event.target === material || event.target === profile || selectors.includes(event.target)) {
    matrix = null;
    el("flat-settings").value = "";
  }
  el("a1-settings").hidden = profile.value !== "a1";
  notice("Generate a preview with these settings.");
});
el("flat-form").addEventListener("submit", event => { event.preventDefault(); generate(selectedPlan); });
el("accept-plan").addEventListener("click", () => generate(state.flat.cost.two_slot_alternative.plan));
el("compare-dither").onclick = () => generate(selectedPlan, Number(el("dither-scope").value), true);
function finishComparison(next) {
  showState(next);
  el("compare-dither").focus();
}
el("accept-dither").onclick = () => { if (trial) finishComparison(trial); };
el("discard-dither").onclick = () => finishComparison(state);
el("direct-match").onclick = () => generate(selectedPlan, null);
bindProjectIo({ getState: () => state, getProjectId: () => editProjectId, setState: showState, onNotice: notice });
if (boot.state) showState(boot.state);
else {
  setSpools(null);
  if (!inventory.length) notice("Add your printer’s spools to the spool library to start a flat print.");
}
