export function renderSwapEditor(container, state, onChange) {
  container.replaceChildren();
  const expected = Math.max(0, state.palette.filaments.length - 1);
  if (expected === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "Single filament";
    container.append(empty);
    return;
  }
  const status = document.createElement("p");
  status.className = "validation-message";
  status.id = "swap-validation";
  status.setAttribute("role", "status");
  status.hidden = true;

  for (let index = 0; index < expected; index += 1) {
    const row = document.createElement("label");
    row.className = "swap-row";
    const from = state.palette.filaments[index];
    const to = state.palette.filaments[index + 1];
    const text = document.createElement("span");
    text.textContent = `${from.name || from.hex} -> ${to.name || to.hex}`;
    const input = document.createElement("input");
    input.type = "number";
    input.min = String(Number(state.layer.base_layers) + 1);
    input.max = String(Number(state.layer.total_layers) - 1);
    input.step = "1";
    input.value = String(state.swap_schedule[index] || "");
    input.addEventListener("input", () => {
      const result = applySwapDraft(state, index, input.value);
      const message = result.message;
      status.textContent = message || "";
      status.hidden = !message;
      onChange(message ? "invalid" : "swap");
    });
    row.append(text, input);
    container.append(row);
  }
  container.append(status);
  const message = validateSwapSchedule(state);
  status.textContent = message || "";
  status.hidden = !message;
}

export function validateSwapSchedule(state) {
  const expected = Math.max(0, state.palette.filaments.length - 1);
  if (!Array.isArray(state.swap_schedule) || state.swap_schedule.length !== expected) {
    return "Swap count must match filament count.";
  }
  const base = Number.parseInt(state.layer.base_layers, 10);
  const total = Number.parseInt(state.layer.total_layers, 10);
  let previous = base;
  for (const raw of state.swap_schedule) {
    const layer = Number.parseInt(raw, 10);
    if (!Number.isInteger(layer) || layer <= base || layer >= total) {
      return `Swap layers must stay between ${base + 1} and ${total - 1}.`;
    }
    if (layer <= previous) {
      return "Swap layers must be strictly increasing.";
    }
    previous = layer;
  }
  return "";
}

export function applySwapDraft(state, index, rawValue) {
  if (String(rawValue).trim() === "") {
    return { valid: false, message: "Swap layers must be integers." };
  }
  const value = Number(rawValue);
  if (!Number.isInteger(value)) {
    return { valid: false, message: "Swap layers must be integers." };
  }
  const next = [...state.swap_schedule];
  next[index] = value;
  const candidate = { ...state, swap_schedule: next };
  const message = validateSwapSchedule(candidate);
  if (message) {
    return { valid: false, message };
  }
  state.swap_schedule = next;
  return { valid: true, message: "" };
}
