import { clampSwapSchedule, evenSwapSchedule, setBoolean, setNumber } from "./state.js";

const NUMBER_CONTROLS = [
  ["layer-height", "layer.layer_height_mm", "layer"],
  ["total-layers", "layer.total_layers", "layer"],
  ["base-layers", "layer.base_layers", "layer"],
  ["max-dim-mm", "mesh.max_dim_mm", "field"],
  ["xy-pixel-mm", "mesh.xy_pixel_mm", "field"],
  ["max-triangles", "mesh.max_triangles", "field"],
  ["black-point", "luminance.black_point", "luminance"],
  ["white-point", "luminance.white_point", "luminance"],
  ["gamma", "luminance.gamma", "luminance"],
  ["backlight-strength", "view.backlight_strength", "view"],
  ["backlight-color-k", "view.backlight_color_k", "view"],
];

export function bindScalarControls(state, onChange, constraints = {}) {
  applyControlConstraints(constraints);
  for (const [id, path, changeType] of NUMBER_CONTROLS) {
    const input = document.getElementById(id);
    if (!input) {
      continue;
    }
    input.addEventListener("input", () => {
      setNumber(state, path, input.value);
      if (path.startsWith("layer.")) {
        clampSwapSchedule(state);
      }
      onChange(changeType);
    });
  }
  const orientation = document.getElementById("orientation");
  orientation?.addEventListener("change", () => {
    state.mesh.orientation = orientation.value;
    onChange("field");
  });
  const flatten = document.getElementById("flatten-transparent");
  flatten?.addEventListener("change", () => {
    setBoolean(state, "mesh.flatten_transparent", flatten.checked);
    onChange("field");
  });
  syncScalarControls(state);
}

function applyControlConstraints(constraints) {
  const mapping = [
    // Layer-count bounds come from the server's single source (boot.layer_constraints),
    // so the input's min/max can't drift from the Python policy.
    ["total-layers", constraints.layer],
    ["black-point", constraints.luminance?.black_point],
    ["white-point", constraints.luminance?.white_point],
    ["gamma", constraints.luminance?.gamma],
    ["backlight-strength", constraints.view?.backlight_strength],
    ["backlight-color-k", constraints.view?.backlight_color_k],
  ];
  for (const [id, rule] of mapping) {
    const input = document.getElementById(id);
    if (!input || !rule) {
      continue;
    }
    if (rule.min !== undefined) {
      input.min = String(rule.min);
    }
    if (rule.max !== undefined) {
      input.max = String(rule.max);
    }
  }
}

export function syncScalarControls(state) {
  setValue("layer-height", state.layer.layer_height_mm);
  setValue("total-layers", state.layer.total_layers);
  setValue("base-layers", state.layer.base_layers);
  setValue("max-dim-mm", state.mesh.max_dim_mm);
  setValue("xy-pixel-mm", state.mesh.xy_pixel_mm);
  setValue("max-triangles", state.mesh.max_triangles);
  setValue("black-point", state.luminance.black_point);
  setValue("white-point", state.luminance.white_point);
  setValue("gamma", state.luminance.gamma);
  setValue("backlight-strength", state.view.backlight_strength);
  setValue("backlight-color-k", state.view.backlight_color_k);
  const orientation = document.getElementById("orientation");
  if (orientation) {
    orientation.value = state.mesh.orientation || "auto";
  }
  const flatten = document.getElementById("flatten-transparent");
  if (flatten) {
    flatten.checked = Boolean(state.mesh.flatten_transparent);
  }
}

export function renderFilamentStack(container, state, onChange, onNotice) {
  container.replaceChildren();
  state.palette.filaments.forEach((filament, index) => {
    const row = document.createElement("div");
    row.className = "filament-editor-row";
    row.draggable = true;
    row.dataset.index = String(index);

    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = filament.hex;

    const identity = document.createElement("div");
    identity.className = "filament-identity";
    const strong = document.createElement("strong");
    strong.textContent = `${filament.brand || "Filament"} ${filament.name || ""}`.trim();
    const hex = document.createElement("input");
    hex.type = "text";
    hex.value = filament.hex;
    hex.pattern = "#[0-9a-fA-F]{6}";
    const colorSource = document.createTextNode(filament.color_level || "matched");
    hex.addEventListener("input", () => {
      filament.color_level = "estimated";
      if (state.version === 2) filament.color_source = "estimated";
      colorSource.textContent = "estimated";
      filament.hex = hex.value;
      swatch.style.background = filament.hex;
      onChange("table");
    });
    identity.append(strong, hex, colorSource);

    const td = document.createElement("label");
    td.className = "td-control";
    const opacitySource = document.createTextNode(`Opacity (mm, ${filament.opacity_level || "matched"})`);
    td.append(opacitySource);
    const tdInput = document.createElement("input");
    tdInput.type = "number";
    tdInput.min = "0.01";
    tdInput.step = "0.01";
    const distanceField = state.version === 2 ? "input_distance_mm" : "td";
    tdInput.value = String(filament[distanceField]);
    tdInput.addEventListener("input", () => {
      const result = applyPositiveNumberDraft(filament, distanceField, tdInput.value);
      if (result.valid) {
        filament.opacity_level = "estimated";
        if (state.version === 2) {
          filament.input_distance_source = "estimated";
          filament.hiding_distance_mm = filament.input_distance_mm * state.front_lit_scale;
        }
        opacitySource.textContent = "Opacity (mm, estimated)";
      }
      onChange(result.valid ? "table" : "invalid");
    });
    td.append(tdInput);

    const actions = document.createElement("div");
    actions.className = "row-actions";
    const up = moveButton("Move up", strong.textContent);
    const down = moveButton("Move down", strong.textContent);
    up.disabled = index === 0;
    down.disabled = index === state.palette.filaments.length - 1;
    up.addEventListener("click", () => reorder(state, index, index - 1, onChange, onNotice));
    down.addEventListener("click", () => reorder(state, index, index + 1, onChange, onNotice));
    actions.append(up, down);

    row.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", String(index));
      event.dataTransfer.effectAllowed = "move";
    });
    row.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      const from = Number.parseInt(event.dataTransfer.getData("text/plain"), 10);
      const to = Number.parseInt(row.dataset.index, 10);
      reorder(state, from, to, onChange, onNotice);
    });

    row.append(swatch, identity, td, actions);
    container.append(row);
  });
}

export function applyPositiveNumberDraft(target, field, rawValue) {
  if (String(rawValue).trim() === "") {
    return { valid: false };
  }
  const value = Number(rawValue);
  if (!Number.isFinite(value) || value <= 0) {
    return { valid: false };
  }
  target[field] = value;
  return { valid: true };
}

function reorder(state, from, to, onChange, onNotice) {
  if (from === to || from < 0 || to < 0 || from >= state.palette.filaments.length || to >= state.palette.filaments.length) {
    return;
  }
  const [item] = state.palette.filaments.splice(from, 1);
  state.palette.filaments.splice(to, 0, item);
  state.swap_schedule = evenSwapSchedule(state.palette.filaments.length, state.layer);
  onNotice("Swap points reset after filament reorder.");
  onChange("palette");
}

function moveButton(direction, spool) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "icon-button secondary";
  const label = `${direction}: ${spool}`;
  button.title = label;
  button.setAttribute("aria-label", label);
  button.textContent = direction;
  return button;
}

function setValue(id, value) {
  const input = document.getElementById(id);
  if (input) {
    input.value = String(value);
  }
}
