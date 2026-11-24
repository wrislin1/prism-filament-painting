import { buildForwardTable, buildHeightField, configureStudioDefaults, renderToCanvas } from "./engine.js";
import { bindScalarControls, renderFilamentStack, syncScalarControls } from "./controls.js";
import { invalidatePreviewCaches } from "./cache.js";
import { bindAccuracyReport } from "./report.js";
import { bindPresets, bindPrintOptions, hasPrintOptions, loadPrintGrid } from "./finishing.js";
import { bindGuidance } from "./guidance.js";
import { bindExport } from "./export.js";
import { bindProjectIo } from "./project-io.js";
import { renderSwapEditor, validateSwapSchedule } from "./swap-editor.js";
import { cloneState, imageUrlForState, loadImageRgba, previewState, readBoot } from "./state.js";

const boot = readBoot();

// Seed the engine's default layer geometry from the server's single source (no client literal).
configureStudioDefaults(boot.layer_constraints);

let editorState = boot.state ? cloneState(boot.state) : null;
let sourceImage = null;
let guidance;
let invalidateExport;
let syncPrintOptions;
let previewRequest;
let previewCache = {
  forwardTable: null,
  heightField: null,
  downsampleFactor: null,
  downsampleCache: new Map(),
};
let frameRequested = false;
let previewRevision = 0;
let reportNeedsPreview = true;

const canvas = document.getElementById("preview-canvas");
const stackContainer = document.getElementById("filament-stack");
const swapContainer = document.getElementById("swap-editor");
const notice = document.getElementById("swap-notice");

const invalidateReport = bindAccuracyReport(() => ({
  state: editorState,
  factor: previewCache.downsampleFactor,
}));

if (editorState) {
  initialize().catch((error) => setNotice(String(error.message || error), true));
}

async function initialize() {
  sourceImage = await loadImageRgba(imageUrlForState(editorState));
  bindScalarControls(editorState, handleChange, {
    ...(boot.constraints || {}),
    layer: boot.layer_constraints,
  });
  bindPresets(editorState, handleChange, setNotice);
  syncPrintOptions = bindPrintOptions(editorState, handleChange);
  guidance = bindGuidance(editorState, setNotice, () => previewCache.forwardTable, () => invalidateExport?.());
  await guidance.restore();
  invalidateExport = bindExport(() => editorState, () => hasPrintOptions(editorState) ? previewCache.downsampleFactor : undefined, guidance.exported);
  bindProjectIo({
    getState: () => editorState,
    setState: replaceState,
    onNotice: setNotice,
  });
  renderDynamicControls();
  setButtonsEnabled(false);
  schedulePreview();
}

function handleChange(changeType) {
  previewRequest?.abort();
  if (document.querySelector("[data-print-option]:invalid")) changeType = "invalid";
  const valid = changeType !== "invalid" && validateSwapSchedule(editorState) === "";
  setButtonsEnabled(valid && !hasPrintOptions(editorState));
  if (changeType !== "view") {
    invalidateExport?.();
    invalidateReport();
    reportNeedsPreview = valid;
  }
  previewRevision++;
  if (changeType === "invalid") {
    cancelAnimationFrame(frameRequested);
    frameRequested = false;
    return;
  }
  if (changeType === "palette" || changeType === "layer") {
    renderDynamicControls();
  }
  previewCache = invalidatePreviewCaches(previewCache, changeType);
  if (changeType !== "finish" && changeType !== "view") previewCache.baseField = null;
  if (changeType === "finish") previewCache.heightField = null;
  schedulePreview();
}

function renderDynamicControls() {
  renderFilamentStack(stackContainer, editorState, handleChange, setNotice);
  renderSwapEditor(swapContainer, editorState, handleChange);
  syncScalarControls(editorState);
  syncPrintOptions();
}

async function replaceState(nextState) {
  const image = await loadImageRgba(imageUrlForState(nextState));
  invalidateExport?.();
  invalidateReport();
  reportNeedsPreview = true;
  // Scalar inputs retain the state object they were bound to on initialization.
  previewRevision++;
  previewRequest?.abort();
  delete editorState.print_options;
  delete editorState.focus_mask;
  delete editorState.depth_map;
  delete editorState.height_strength;
  for (const field of Object.keys(editorState)) delete editorState[field];
  Object.assign(editorState, cloneState(nextState));
  bindPresets(editorState, handleChange, setNotice);
  await guidance.restore();
  sourceImage = image;
  previewCache = {
    forwardTable: null,
    heightField: null,
    downsampleFactor: null,
    downsampleCache: new Map(),
  };
  renderDynamicControls();
  setButtonsEnabled(false);
  schedulePreview();
}

function schedulePreview() {
  if (frameRequested) {
    return;
  }
  frameRequested = requestAnimationFrame(async () => {
    frameRequested = false;
    try {
      await renderPreview(++previewRevision);
    } catch (error) {
      if (error.name === "AbortError") return;
      setButtonsEnabled(false);
      canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
      document.getElementById("print-grid-status").textContent = String(error.message || error);
    }
  });
}

async function renderPreview(revision) {
  if (!sourceImage || !editorState) {
    return;
  }
  if (!previewCache.forwardTable) {
    if (boot.calibration_enabled) {
      const response = await fetch("/studio/prediction-table", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(editorState),
      });
      if (revision !== previewRevision) return;
      if (!response.ok) throw new Error("Preview colour table unavailable");
      const table = await response.json();
      if (revision !== previewRevision) return;
      previewCache.forwardTable = table;
    } else {
      previewCache.forwardTable = buildForwardTable(editorState.palette.filaments, editorState.swap_schedule, editorState.layer, editorState.front_lit_scale);
    }
  }
  const finished = hasPrintOptions(editorState);
  const status = document.getElementById("print-grid-status");
  if (!previewCache.baseField) {
    const state = previewState(editorState, sourceImage);
    previewCache.baseField = buildHeightField(state.image, previewCache.forwardTable, {
      layer: state.layer,
      mesh: state.mesh,
      luminance: state.luminance,
      downsampleFactorHint: previewCache.downsampleFactor,
      downsampleCache: previewCache.downsampleCache,
    });
    previewCache.downsampleFactor = previewCache.baseField.downsample_factor;
  }
  if (finished && !previewCache.heightField) {
    setButtonsEnabled(false);
    status.textContent = "Updating print grid…";
    const controller = new AbortController();
    previewRequest = controller;
    const grid = await loadPrintGrid(editorState, previewCache.downsampleFactor, controller.signal);
    if (controller.signal.aborted || revision !== previewRevision) return;
    previewCache.heightField = grid;
    status.textContent = `${grid.width} × ${grid.height} cells · ${grid.pitch.toFixed(3)} mm per cell`;
  } else if (!finished) {
    status.textContent = "";
    previewCache.heightField = previewCache.baseField;
  }
  renderToCanvas(canvas, previewCache.heightField, previewCache.forwardTable, editorState.view);
  if (boot.calibration_enabled) {
    const heights = new Set(previewCache.heightField.field);
    heights.delete(0);
    const sources = new Set(Array.from(heights, h => previewCache.forwardTable.layer_sources[h]));
    document.getElementById("preview-sources").textContent = "Prediction source: " + [...sources].sort().join(", ");
  }
  setButtonsEnabled(validateSwapSchedule(editorState) === "");
  if (reportNeedsPreview) {
    guidance?.updateCost();
    reportNeedsPreview = false;
    document.getElementById("studio-report").disabled = false;
  }
}

function setNotice(message, error = false) {
  if (!notice) {
    return;
  }
  notice.textContent = message;
  notice.hidden = !message;
  notice.dataset.kind = error ? "error" : "info";
}

function setButtonsEnabled(enabled) {
  for (const id of ["studio-save", "studio-export"]) {
    const button = document.getElementById(id);
    if (button) {
      button.disabled = !enabled;
    }
  }
}

export function currentStudioState() {
  return editorState;
}
