import { evenSwapSchedule } from "./state.js";

function showPaletteMaterial(state) {
  const status = document.getElementById("stack-material");
  const group = state.palette.material_group;
  status.hidden = !group;
  status.textContent = group === "mixed" ? status.dataset.warning : group ? "Current palette: " + group : "";
}

export async function bindPresets(state, onChange, onNotice, restore = true) {
  const select = document.getElementById("palette-preset");
  const material = document.getElementById("print-material");
  if (restore) {
    showPaletteMaterial(state);
    if (material) {
      const group = state.palette.material_group;
      const matches = group && group !== "mixed" && Array.from(material.options).some(o => o.value === group);
      material.value = matches ? group : "auto";
      document.getElementById("material-note").textContent = matches ? "Presets use " + group + " to match this palette." : material.dataset.auto;
    }
  }
  const requested = material?.value || "auto";
  const request = Symbol();
  select.presetRequest = request;
  if (material) material.onchange = () => bindPresets(state, onChange, onNotice, false);
  select.disabled = true;
  select.length = 1;
  select.options[0].text = "Choose from available filaments";
  try {
    const response = await fetch(requested === "auto" ? "/studio/presets" :
      "/studio/presets?material_group=" + encodeURIComponent(requested));
    if (!response.ok) throw new Error();
    const { presets, material_group } = await response.json();
    if (select.presetRequest !== request) return;
    for (const [index, preset] of presets.entries()) {
      const missing = preset.missing.length ? " · missing " + preset.missing.join(", ") : "";
      const option = new Option(preset.name + missing, String(index));
      option.disabled = preset.filaments.length < 2;
      select.add(option);
    }
    select.disabled = false;
    if (!presets.some(preset => preset.filaments.length >= 2)) {
      select.options[0].text = "No preset has two matching colours";
    }
    select.onchange = () => {
      if (select.value === "") return;
      const preset = presets[Number(select.value)];
      select.value = "";
      if (preset.filaments.length > state.layer.total_layers - state.layer.base_layers) {
        onNotice("Increase the layer count to fit this palette.", true);
        return;
      }
      state.palette = { filaments: structuredClone(preset.filaments), score: 0, notes: preset.name };
      if (state.version === 2) {
        for (const spool of state.palette.filaments) {
          spool.input_distance_mm = spool.td;
          spool.input_distance_source = spool.opacity_level || "unknown";
          spool.color_source = spool.color_level || "unknown";
          spool.hiding_distance_mm = spool.td * state.front_lit_scale;
          delete spool.td;
        }
      }
      if (material_group) state.palette.material_group = material_group;
      showPaletteMaterial(state);
      state.swap_schedule = evenSwapSchedule(state.palette.filaments.length, state.layer);
      onChange("palette");
      onNotice(preset.name + (preset.missing.length ? " · omitted " + preset.missing.join(", ") : ""));
    };
  } catch {
    if (select.presetRequest === request) select.options[0].text = "Filament table unavailable";
  }
}

export function bindPrintOptions(state, onChange) {
  const inputs = document.querySelectorAll("[data-print-option]");
  for (const input of inputs) {
    input.addEventListener("input", () => {
      if (Array.from(inputs).some(item => !item.validity.valid)) {
        document.getElementById("print-grid-status").textContent = "Enter a valid number for each print option.";
        onChange("invalid");
        return;
      }
      state.print_options = Object.fromEntries(Array.from(inputs, item => [
        item.dataset.printOption, item.type === "checkbox" ? item.checked : item.valueAsNumber,
      ]));
      onChange("finish");
    });
  }
  return () => {
    for (const input of inputs) {
      const value = state.print_options?.[input.dataset.printOption];
      if (input.type === "checkbox") input.checked = Boolean(value);
      else input.value = value || 0;
    }
  };
}

export function hasPrintOptions(state) {
  return Object.entries(state.print_options || {}).some(([key, value]) =>
    key !== "hole_x" && key !== "hole_y" && Boolean(value));
}

export async function loadPrintGrid(state, baseFactor, signal) {
  const response = await fetch(`/studio/print-grid?base_factor=${baseFactor}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state), signal,
  });
  const grid = await response.json();
  if (!response.ok) throw new Error(grid.detail || "Print preview unavailable");
  grid.field = Uint8Array.from(atob(grid.cells), character => character.charCodeAt(0));
  return grid;
}
