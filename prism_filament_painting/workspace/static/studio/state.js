export function readBoot() {
  const node = document.getElementById("studio-boot");
  if (!node) {
    throw new Error("studio boot JSON is missing");
  }
  return JSON.parse(node.textContent || "{}");
}

export function cloneState(state) {
  return JSON.parse(JSON.stringify(state));
}

export function imageUrlForState(state) {
  if (state.image?.project_id) {
    return `/studio/projects/${encodeURIComponent(state.image.project_id)}/image`;
  }
  if (state.image?.upload_token) {
    return `/studio/image/${encodeURIComponent(state.image.upload_token)}`;
  }
  throw new Error("state has no image source");
}

export async function loadImageRgba(url) {
  const img = new Image();
  img.decoding = "async";
  img.src = url;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("could not load image"));
  });
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const context = canvas.getContext("2d");
  context.drawImage(img, 0, 0);
  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  return {
    width: canvas.width,
    height: canvas.height,
    rgba: Array.from(imageData.data),
  };
}

export function previewState(editorState, sourceImage) {
  return {
    ...editorState,
    image: sourceImage,
  };
}

export function evenSwapSchedule(paletteSize, layer) {
  if (paletteSize <= 1) {
    return [];
  }
  const base = Number.parseInt(layer.base_layers, 10);
  const total = Number.parseInt(layer.total_layers, 10);
  const span = total - base;
  return Array.from({ length: paletteSize - 1 }, (_, index) => roundHalfEven(base + ((index + 1) * span) / paletteSize));
}

export function clampSwapSchedule(state) {
  const expected = Math.max(0, state.palette.filaments.length - 1);
  if (!Array.isArray(state.swap_schedule) || state.swap_schedule.length !== expected) {
    state.swap_schedule = evenSwapSchedule(state.palette.filaments.length, state.layer);
    return;
  }
  let previous = Number.parseInt(state.layer.base_layers, 10);
  const total = Number.parseInt(state.layer.total_layers, 10);
  state.swap_schedule = state.swap_schedule.map((raw) => {
    let value = Number.parseInt(raw, 10);
    value = Math.max(previous + 1, Math.min(total - 1, value));
    previous = value;
    return value;
  });
}

export function setNumber(target, path, value) {
  const parts = path.split(".");
  let cursor = target;
  for (const part of parts.slice(0, -1)) {
    cursor = cursor[part];
  }
  cursor[parts[parts.length - 1]] = Number(value);
}

export function setBoolean(target, path, value) {
  const parts = path.split(".");
  let cursor = target;
  for (const part of parts.slice(0, -1)) {
    cursor = cursor[part];
  }
  cursor[parts[parts.length - 1]] = Boolean(value);
}

function roundHalfEven(value) {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction > 0.5) {
    return floor + 1;
  }
  if (fraction < 0.5) {
    return floor;
  }
  return floor % 2 === 0 ? floor : floor + 1;
}
