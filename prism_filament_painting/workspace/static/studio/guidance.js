import { imageUrlForState } from "./state.js";
import { previewCost, costText } from "./guidance-cost.js";

export function bindGuidance(state, onNotice, getTable, invalidateExport) {
  const panel = document.getElementById("guidance-panel");
  if (!panel) return { restore() {}, updateCost() {} };
  const canvas = document.getElementById("focus-canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  const source = document.getElementById("focus-source");
  const strength = document.getElementById("focus-strength");
  const tool = document.getElementById("focus-tool");
  const depthFile = document.getElementById("depth-file");
  const height = document.getElementById("height-strength");
  const column = document.getElementById("focus-column");
  const row = document.getElementById("focus-row");
  const applyBrush = document.getElementById("focus-apply");
  let pointer = null;
  let previous;
  let revision = 0;
  let mapRevision = 0;
  let initialized = false;

  const undo = document.getElementById("undo-focus-clear");
  let cleared = null;
  function discardUndo() {
    cleared = null;
    undo.hidden = true;
  }

  const cost = document.getElementById("depth-colour-cost");
  let costRevision = 0;
  let map = null;
  async function updateCost() {
    const current = ++costRevision;
    if (!state.height_strength) {
      cost.textContent = "Depth off. No added colour error.";
      return;
    }
    cost.textContent = "Comparing colours with depth off…";
    if (!initialized || !getTable() || !state.depth_map) return;
    try {
      if (map?.src !== state.depth_map) {
        map = new Image();
        map.src = state.depth_map;
      }
      await map.decode();
      if (current === costRevision) cost.textContent = previewCost(state, source, map, canvas, getTable());
    } catch {
      if (current === costRevision) cost.textContent = "Colour comparison unavailable.";
    }
  }

  function changed() {
    invalidateExport();
    updateCost();
  }
  function clear() {
    if (!initialized || !state.focus_mask) return;
    cleared = { mask: state.focus_mask, pixels: context.getImageData(0, 0, canvas.width, canvas.height) };
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
    delete state.focus_mask;
    undo.hidden = false;
    onNotice("Focus mask cleared.");
    undo.focus();
    changed();
  }
  undo.onclick = () => {
    state.focus_mask = cleared.mask;
    context.putImageData(cleared.pixels, 0, 0);
    changed();
    discardUndo();
    onNotice("Focus mask restored.");
    document.getElementById("focus-clear").focus();
  };

  function drawAt(x, y) {
    const radius = Math.max(1, Math.min(canvas.width, canvas.height) * 0.035);
    const weight = Math.max(2, Math.min(100, Number(strength.value)));
    const shade = tool.value === "erase" ? 0 : Math.round(255 * (weight - 1) / 99);
    context.fillStyle = context.strokeStyle = `rgb(${shade},${shade},${shade})`;
    context.lineWidth = 2 * radius;
    context.lineCap = "round";
    if (previous) {
      context.beginPath();
      context.moveTo(...previous);
      context.lineTo(x, y);
      context.stroke();
    } else {
      context.beginPath();
      context.arc(x, y, radius, 0, Math.PI * 2);
      context.fill();
    }
    previous = [x, y];
  }

  function draw(event) {
    const rect = canvas.getBoundingClientRect();
    drawAt((event.clientX - rect.left) * canvas.width / rect.width,
           (event.clientY - rect.top) * canvas.height / rect.height);
  }

  canvas.addEventListener("pointerdown", event => {
    if (!initialized || pointer !== null || event.button !== 0) return;
    discardUndo();
    pointer = event.pointerId;
    previous = null;
    canvas.setPointerCapture(pointer);
    draw(event);
  });
  canvas.addEventListener("pointermove", event => {
    if (event.pointerId === pointer) draw(event);
  });
  canvas.addEventListener("lostpointercapture", event => {
    if (event.pointerId !== pointer) return;
    pointer = null;
    saveBrush();
  });
  function saveBrush() {
    state.focus_mask = canvas.toDataURL("image/png");
    if (tool.value === "erase") {
      const bytes = context.getImageData(0, 0, canvas.width, canvas.height).data;
      if (!bytes.some((value, index) => index % 4 === 0 && value)) delete state.focus_mask;
    }
    changed();
  }
  applyBrush.addEventListener("click", () => {
    if (!initialized || !column.reportValidity() || !row.reportValidity()) return;
    discardUndo();
    previous = null;
    drawAt(column.valueAsNumber - 0.5, row.valueAsNumber - 0.5);
    saveBrush();
    onNotice("Brush applied at column " + column.value + ", row " + row.value + ".");
  });
  document.getElementById("focus-clear").addEventListener("click", clear);
  height.addEventListener("change", () => { state.height_strength = Number(height.value); changed(); });
  depthFile.addEventListener("change", async () => {
    const file = depthFile.files[0];
    if (!file) return;
    const current = ++mapRevision;
    const url = URL.createObjectURL(file);
    try {
      const img = new Image();
      img.src = url;
      await img.decode();
      if (current !== mapRevision) return;
      if (img.naturalWidth !== canvas.width || img.naturalHeight !== canvas.height) {
        throw new Error("Depth map must match the source image dimensions.");
      }
      const buffer = document.createElement("canvas");
      buffer.width = canvas.width;
      buffer.height = canvas.height;
      buffer.getContext("2d").drawImage(img, 0, 0);
      state.depth_map = buffer.toDataURL("image/png");
      changed();
      height.disabled = false;
      onNotice("Depth map loaded. Choose a height strength for export.");
    } catch (error) {
      if (current === mapRevision) onNotice(String(error.message || error), true);
    } finally {
      URL.revokeObjectURL(url);
    }
  });

  async function restore() {
    discardUndo();
    const current = ++revision;
    mapRevision++;
    pointer = null;
    initialized = false;
    applyBrush.disabled = true;
    if (!panel.open) return;
    source.src = imageUrlForState(state) + "?raster_axes=true";
    await source.decode();
    if (current !== revision) return;
    canvas.width = source.naturalWidth;
    canvas.height = source.naturalHeight;
    depthFile.value = "";
    height.value = String(state.height_strength || 0);
    height.disabled = !state.depth_map;
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (state.focus_mask) {
      const mask = new Image();
      mask.src = state.focus_mask;
      await mask.decode();
      if (current !== revision) return;
      context.drawImage(mask, 0, 0);
    }
    column.max = canvas.width;
    row.max = canvas.height;
    initialized = true;
    applyBrush.disabled = false;
    updateCost();
  }
  panel.addEventListener("toggle", () => {
    if (panel.open && !initialized) restore().catch(error => onNotice(String(error), true));
  });
  function exported(report) {
    if (Number.isFinite(report?.baseline_colour_mse) && Number.isFinite(report?.source_colour_mse)) {
      ++costRevision;
      cost.textContent = costText(report.baseline_colour_mse, report.source_colour_mse, "Optimized export before finishing");
    }
  }
  return { restore, updateCost, exported };
}
