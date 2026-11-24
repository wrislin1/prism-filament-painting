// Live default layer geometry for the normalizeLayer fallback. Seeded from the server's
// single source via configureStudioDefaults() (called from main.js with the boot
// payload's layer_constraints object) so the browser never hardcodes a divergent value.
let defaultTotalLayers = 27;
let defaultBaseLayers = 2;
let defaultLayerHeightMm = 0.08;
// A pre-scale boot payload used raw TD. Current servers supply their render scale below.
let tdScale = 1;

export function configureStudioDefaults(constraints) {
  if (!constraints || typeof constraints !== "object") return;
  const scale = Number(constraints.td_scale);
  if (Number.isFinite(scale) && scale > 0) tdScale = scale;
  const t = Number(constraints.default);
  if (Number.isFinite(t) && t > 0) defaultTotalLayers = t;
  const b = Number(constraints.base_layers);
  if (Number.isFinite(b) && b > 0) defaultBaseLayers = b;
  const h = Number(constraints.layer_height_mm);
  if (Number.isFinite(h) && h > 0) defaultLayerHeightMm = h;
}

const DEFAULT_LAYER = Object.freeze({
  get layer_height_mm() {
    return defaultLayerHeightMm;
  },
  get base_layers() {
    return defaultBaseLayers;
  },
  get total_layers() {
    return defaultTotalLayers;
  },
});

const DEFAULT_MESH = Object.freeze({
  max_triangles: 1_500_000,
  flatten_transparent: true,
  orientation: "auto",
});

const DEFAULT_LUMINANCE = Object.freeze({
  black_point: 0.0,
  white_point: 1.0,
  gamma: 1.0,
});

const XY_UNITS = 1_000_000;
const RELIEF_UNITS = 10_000;
const NW = 0;
const NE = 1;
const SE = 2;
const SW = 3;
const NORTH = 0;
const EAST = 1;
const SOUTH = 2;
const WEST = 3;

export function roundHalfEven(value) {
  if (!Number.isFinite(value)) {
    throw new Error("value must be finite");
  }
  const sign = value < 0 ? -1 : 1;
  const abs = Math.abs(value);
  const floor = Math.floor(abs);
  const fraction = abs - floor;
  let rounded;
  if (fraction > 0.5) {
    rounded = floor + 1;
  } else if (fraction < 0.5) {
    rounded = floor;
  } else {
    rounded = floor % 2 === 0 ? floor : floor + 1;
  }
  return sign * rounded;
}

export function clampChannel(value) {
  return Math.max(0, Math.min(255, roundHalfEven(value)));
}

export function hexToRgb(value) {
  const raw = String(value || "").trim().replace(/^#/, "");
  if (!/^[0-9a-fA-F]{6}$/.test(raw)) {
    throw new Error("hex color must be #RRGGBB");
  }
  return [
    Number.parseInt(raw.slice(0, 2), 16),
    Number.parseInt(raw.slice(2, 4), 16),
    Number.parseInt(raw.slice(4, 6), 16),
  ];
}

export function beerLambertBlend(overRgb, underRgb, thicknessMm, hidingDistanceMm) {
  if (!Number.isFinite(thicknessMm) || thicknessMm < 0) {
    throw new Error("thickness_mm must be finite and non-negative");
  }
  if (!Number.isFinite(hidingDistanceMm) || hidingDistanceMm <= 0) {
    throw new Error("hiding distance must be finite and positive");
  }
  const transmission = 10 ** (-thicknessMm / hidingDistanceMm);
  return overRgb.map((over, index) => clampChannel(over * (1 - transmission) + underRgb[index] * transmission));
}

export function relativeLuminance(rgb) {
  const r = linearize(rgb[0] / 255.0);
  const g = linearize(rgb[1] / 255.0);
  const b = linearize(rgb[2] / 255.0);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function linearize(value) {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return ((value + 0.055) / 1.055) ** 2.4;
}

export function buildForwardTable(filaments, swapSchedule, layerParams = DEFAULT_LAYER, frontLitScale = tdScale) {
  const layer = normalizeLayer(layerParams);
  if (!Array.isArray(filaments) || filaments.length === 0) {
    throw new Error("filaments must not be empty");
  }
  if (!Array.isArray(swapSchedule) || swapSchedule.length !== filaments.length - 1) {
    throw new Error("swap_schedule length must be one less than filament count");
  }
  const filamentRgb = filaments.map((filament) => hexToRgb(filament.hex));
  const layerRgb = [];
  const baseRgb = filamentRgb[0];
  let current = baseRgb;
  const schedule = swapSchedule.map((value) => Number.parseInt(value, 10));
  const firstSwap = schedule.length > 0 ? schedule[0] : layer.total_layers + 1;
  for (let index = 0; index < Math.min(firstSwap, layer.total_layers + 1); index += 1) {
    layerRgb.push([...baseRgb]);
  }
  for (let scheduleIndex = 0; scheduleIndex < schedule.length; scheduleIndex += 1) {
    const filamentIndex = scheduleIndex + 1;
    const startLayer = schedule[scheduleIndex];
    const nextStart = scheduleIndex + 1 < schedule.length ? schedule[scheduleIndex + 1] : layer.total_layers + 1;
    for (let currentLayer = startLayer; currentLayer < Math.min(nextStart, layer.total_layers + 1); currentLayer += 1) {
      const printedLayers = currentLayer - startLayer + 1;
      const filament = filaments[filamentIndex];
      layerRgb.push(
        beerLambertBlend(
          filamentRgb[filamentIndex],
          current,
          printedLayers * layer.layer_height_mm,
          Number(filament.input_distance_mm ?? filament.td) * frontLitScale,
        ),
      );
    }
    if (layerRgb.length > 0) {
      current = layerRgb[layerRgb.length - 1];
    }
  }
  if (layerRgb.length !== layer.total_layers + 1) {
    throw new Error("forward table layer count mismatch");
  }
  return { layer_rgb: layerRgb };
}

export function buildHeightField(image, table, options = {}) {
  const layer = normalizeLayer(options.layer || options.layerParams || DEFAULT_LAYER);
  const mesh = normalizeMesh(options.mesh || options.meshParams || DEFAULT_MESH);
  const luminance = normalizeLuminance(options.luminance || options.luminanceParams || DEFAULT_LUMINANCE);
  const orientation = resolveOrientation(mesh.orientation || options.orientation || "auto", table, layer);
  const forcedFactor =
    options.downsampleFactor === undefined || options.downsampleFactor === null
      ? null
      : Math.max(1, Number.parseInt(options.downsampleFactor, 10));
  const downsampleCache = options.downsampleCache || null;
  if (forcedFactor !== null) {
    return heightFieldAtFactor(image, layer, mesh, luminance, orientation, forcedFactor, downsampleCache);
  }
  const factorHint =
    options.downsampleFactorHint === undefined || options.downsampleFactorHint === null
      ? null
      : Math.max(1, Number.parseInt(options.downsampleFactorHint, 10));
  return buildRecoveredHeightField(image, layer, mesh, luminance, orientation, {
    downsampleCache,
    factorHint,
  });
}

function buildRecoveredHeightField(image, layer, mesh, luminance, orientation, options = {}) {
  const source = normalizeRgbaImage(image);
  const cap = mesh.max_triangles;
  const fieldCache = new Map();

  function isUnit(factor) {
    return Math.ceil(source.width / factor) <= 1 && Math.ceil(source.height / factor) <= 1;
  }

  function fieldAt(factor) {
    if (!fieldCache.has(factor)) {
      fieldCache.set(factor, heightFieldAtFactor(source, layer, mesh, luminance, orientation, factor, options.downsampleCache));
    }
    return fieldCache.get(factor);
  }

  let factor = Number.isInteger(options.factorHint) ? Math.max(1, options.factorHint) : 1;
  let field = fieldAt(factor);
  let bound = triangleCountUpperBound(field);
  if (bound > cap) {
    factor = Math.max(2, Math.ceil(Math.sqrt(bound / cap)));
    field = fieldAt(factor);
    while (triangleCountUpperBound(field) > cap) {
      if (isUnit(factor)) {
        throw new Error("image cannot be reduced under max_triangles");
      }
      factor += 1;
      field = fieldAt(factor);
    }
    while (factor > 1) {
      const finer = fieldAt(factor - 1);
      if (triangleCountUpperBound(finer) > cap) {
        break;
      }
      factor -= 1;
      field = finer;
    }
  } else {
    while (factor > 1) {
      const finer = fieldAt(factor - 1);
      if (triangleCountUpperBound(finer) > cap) {
        break;
      }
      factor -= 1;
      field = finer;
    }
  }

  const exactCache = new Map();
  function exactCandidate(candidateFactor) {
    if (!exactCache.has(candidateFactor)) {
      const candidateField = fieldAt(candidateFactor);
      const clips = saddleClips(candidateField);
      const exact = meshTriangleCountFromClips(candidateField, clips);
      exactCache.set(candidateFactor, { field: candidateField, clips, exact });
    }
    return exactCache.get(candidateFactor);
  }

  let exact = exactCandidate(factor);
  field = exact.field;
  while (exact.exact > cap) {
    if (isUnit(factor)) {
      throw new Error("image cannot be reduced under max_triangles");
    }
    factor += 1;
    exact = exactCandidate(factor);
    field = exact.field;
  }

  let safeFactor = factor;
  if (safeFactor > 1) {
    let low = 0;
    let probe = Math.max(1, Math.floor(safeFactor / 2));
    while (true) {
      const probeExact = exactCandidate(probe).exact;
      if (probeExact <= cap) {
        safeFactor = probe;
        if (probe === 1) {
          break;
        }
        probe = Math.max(1, Math.floor(probe / 2));
        continue;
      }
      low = probe;
      break;
    }

    if (low > 0) {
      let high = safeFactor;
      while (high - low > 1) {
        const mid = Math.floor((low + high) / 2);
        if (exactCandidate(mid).exact <= cap) {
          high = mid;
        } else {
          low = mid;
        }
      }
      safeFactor = high;
    }

    while (safeFactor > 1) {
      if (exactCandidate(safeFactor - 1).exact > cap) {
        break;
      }
      safeFactor -= 1;
    }

    factor = safeFactor;
    field = exactCandidate(factor).field;
  }

  return field;
}

function heightFieldAtFactor(image, layer, mesh, luminance, orientation, factor, downsampleCache = null) {
  const rgba = factor > 1 ? boxDownsampleRgba(image, factor, downsampleCache) : normalizeRgbaImage(image);
  const field = new Int32Array(rgba.width * rgba.height);
  const span = layer.total_layers - layer.base_layers;
  const identityLuminance =
    luminance.black_point === 0.0 && luminance.white_point === 1.0 && luminance.gamma === 1.0;
  for (let index = 0; index < rgba.width * rgba.height; index += 1) {
    const offset = index * 4;
    let value = relativeLuminance([rgba.data[offset], rgba.data[offset + 1], rgba.data[offset + 2]]);
    if (!identityLuminance) {
      value = Math.max(0, Math.min(1, (value - luminance.black_point) / (luminance.white_point - luminance.black_point)));
      value = value ** luminance.gamma;
    }
    const normalized = orientation === "dark" ? 1.0 - value : value;
    let layerValue = roundHalfEven(layer.base_layers + normalized * span);
    layerValue = Math.max(layer.base_layers, Math.min(layer.total_layers, layerValue));
    if (mesh.flatten_transparent && rgba.data[offset + 3] === 0) {
      layerValue = layer.base_layers;
    }
    field[index] = layerValue;
  }
  return {
    field,
    width: rgba.width,
    height: rgba.height,
    downsample_factor: factor,
    resolved_orientation: orientation,
  };
}

export function renderPreviewPixels(heightField, table) {
  const field = heightField.field || heightField;
  const width = heightField.width;
  const height = heightField.height;
  const pixels = new Uint8ClampedArray(width * height * 3);
  for (let index = 0; index < field.length; index += 1) {
    const rgb = table.layer_rgb[field[index]];
    if (!rgb) {
      throw new Error("height_field contains layer outside forward table");
    }
    const offset = index * 3;
    pixels[offset] = rgb[0];
    pixels[offset + 1] = rgb[1];
    pixels[offset + 2] = rgb[2];
  }
  return pixels;
}

export function renderPreviewRgba(heightField, table, view = {}) {
  const rgb = renderPreviewPixels(heightField, table);
  const field = heightField.field || heightField;
  const strength = Math.max(0, Math.min(1, Number(view.backlight_strength || 0)));
  const light = colorTemperatureToRgb(Number(view.backlight_color_k || 6500));
  const rgba = new Uint8ClampedArray(heightField.width * heightField.height * 4);
  for (let index = 0; index < heightField.width * heightField.height; index += 1) {
    const rgbOffset = index * 3;
    const rgbaOffset = index * 4;
    rgba[rgbaOffset] = clampChannel(rgb[rgbOffset] * (1 - strength) + light[0] * strength);
    rgba[rgbaOffset + 1] = clampChannel(rgb[rgbOffset + 1] * (1 - strength) + light[1] * strength);
    rgba[rgbaOffset + 2] = clampChannel(rgb[rgbOffset + 2] * (1 - strength) + light[2] * strength);
    rgba[rgbaOffset + 3] = field[index] === 0 ? 0 : 255;
  }
  return rgba;
}

export function renderToCanvas(canvas, heightField, table, view = {}) {
  const context = canvas.getContext("2d");
  const rgba = renderPreviewRgba(heightField, table, view);
  canvas.width = heightField.width;
  canvas.height = heightField.height;
  context.putImageData(new ImageData(rgba, heightField.width, heightField.height), 0, 0);
}

export function createPreview(state, options = {}) {
  const table =
    options.forwardTable ||
    buildForwardTable(state.palette.filaments, state.swap_schedule || [], state.layer || DEFAULT_LAYER, state.front_lit_scale);
  const heightField = buildHeightField(state.image, table, {
    layer: state.layer,
    mesh: state.mesh,
    luminance: state.luminance,
    downsampleFactor: options.downsampleFactor,
  });
  return {
    table,
    heightField,
    pixels: renderPreviewPixels(heightField, table),
  };
}

export function boxDownsampleRgba(image, factor, downsampleCache = null) {
  const source = normalizeRgbaImage(image);
  const outWidth = Math.max(1, Math.ceil(source.width / factor));
  const outHeight = Math.max(1, Math.ceil(source.height / factor));
  const cacheKey = `${source.width}x${source.height}:${factor}:${outWidth}x${outHeight}`;
  if (downsampleCache?.has(cacheKey)) {
    return downsampleCache.get(cacheKey);
  }
  const resized = boxResizeRgba(source, outWidth, outHeight);
  downsampleCache?.set(cacheKey, resized);
  return resized;
}

export function boxResizeRgba(image, outWidth, outHeight) {
  const source = normalizeRgbaImage(image);
  const xBuckets = boxBuckets(source.width, outWidth);
  const yBuckets = boxBuckets(source.height, outHeight);
  const temp = new Uint8ClampedArray(source.height * outWidth * 4);
  for (let sy = 0; sy < source.height; sy += 1) {
    for (let x = 0; x < outWidth; x += 1) {
      const xBucket = xBuckets[x];
      const premultiplied = [0, 0, 0];
      let alphaWeight = 0;
      let alphaSum = 0;
      let count = 0;
      for (let sx = xBucket.start; sx < xBucket.end; sx += 1) {
        const offset = (sy * source.width + sx) * 4;
        const alpha = source.data[offset + 3];
        premultiplied[0] += source.data[offset] * alpha;
        premultiplied[1] += source.data[offset + 1] * alpha;
        premultiplied[2] += source.data[offset + 2] * alpha;
        alphaWeight += alpha;
        alphaSum += alpha;
        count += 1;
      }
      const tempOffset = (sy * outWidth + x) * 4;
      temp[tempOffset] = alphaWeight > 0 ? clampBoxByte(premultiplied[0] / alphaWeight) : 0;
      temp[tempOffset + 1] = alphaWeight > 0 ? clampBoxByte(premultiplied[1] / alphaWeight) : 0;
      temp[tempOffset + 2] = alphaWeight > 0 ? clampBoxByte(premultiplied[2] / alphaWeight) : 0;
      temp[tempOffset + 3] = clampBoxByte(alphaSum / count);
    }
  }

  const data = new Uint8ClampedArray(outWidth * outHeight * 4);
  for (let y = 0; y < outHeight; y += 1) {
    const yBucket = yBuckets[y];
    for (let x = 0; x < outWidth; x += 1) {
      const premultiplied = [0, 0, 0];
      let alphaWeight = 0;
      let alphaSum = 0;
      let count = 0;
      for (let sy = yBucket.start; sy < yBucket.end; sy += 1) {
        const offset = (sy * outWidth + x) * 4;
        const alpha = temp[offset + 3];
        premultiplied[0] += temp[offset] * alpha;
        premultiplied[1] += temp[offset + 1] * alpha;
        premultiplied[2] += temp[offset + 2] * alpha;
        alphaWeight += alpha;
        alphaSum += alpha;
        count += 1;
      }
      const outOffset = (y * outWidth + x) * 4;
      data[outOffset] = alphaWeight > 0 ? clampBoxByte(premultiplied[0] / alphaWeight) : 0;
      data[outOffset + 1] = alphaWeight > 0 ? clampBoxByte(premultiplied[1] / alphaWeight) : 0;
      data[outOffset + 2] = alphaWeight > 0 ? clampBoxByte(premultiplied[2] / alphaWeight) : 0;
      data[outOffset + 3] = clampBoxByte(alphaSum / count);
    }
  }
  return { width: outWidth, height: outHeight, data };
}

function boxBuckets(inSize, outSize) {
  const scale = inSize / outSize;
  const support = 0.5 * Math.max(1, scale);
  return Array.from({ length: outSize }, (_, index) => {
    const center = (index + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(inSize, Math.floor(center + support + 0.5));
    return { start, end: Math.max(start + 1, end) };
  });
}

function clampBoxByte(value) {
  return Math.max(0, Math.min(255, Math.floor(value + 0.5)));
}

export function triangleCountUpperBound(heightField) {
  const field = heightField.field;
  const width = heightField.width;
  const height = heightField.height;
  if (width < 1 || height < 1) {
    return 0;
  }
  const corners = new Int32Array(width * height);
  for (let row = 1; row < height; row += 1) {
    for (let col = 1; col < width; col += 1) {
      const nw = field[(row - 1) * width + (col - 1)];
      const ne = field[(row - 1) * width + col];
      const sw = field[row * width + (col - 1)];
      const se = field[row * width + col];
      if (Math.min(nw, se) > Math.max(ne, sw) || Math.min(ne, sw) > Math.max(nw, se)) {
        corners[(row - 1) * width + (col - 1)] += 1;
        corners[(row - 1) * width + col] += 1;
        corners[row * width + (col - 1)] += 1;
        corners[row * width + col] += 1;
      }
    }
  }
  let topBase = 0;
  let walls = 0;
  for (let index = 0; index < field.length; index += 1) {
    topBase += 2 * (2 + corners[index]);
    walls += 2 * corners[index] * field[index];
  }
  let grid = 0;
  if (width > 1) {
    for (let row = 0; row < height; row += 1) {
      for (let col = 0; col < width - 1; col += 1) {
        const a = field[row * width + col];
        const b = field[row * width + col + 1];
        grid += 2 * Math.abs(a - b) + 4 * (a + b);
      }
    }
  }
  if (height > 1) {
    for (let row = 0; row < height - 1; row += 1) {
      for (let col = 0; col < width; col += 1) {
        const a = field[row * width + col];
        const b = field[(row + 1) * width + col];
        grid += 2 * Math.abs(a - b) + 4 * (a + b);
      }
    }
  }
  for (let row = 0; row < height; row += 1) {
    grid += 2 * (field[row * width] + field[row * width + width - 1]);
  }
  for (let col = 0; col < width; col += 1) {
    grid += 2 * (field[col] + field[(height - 1) * width + col]);
  }
  return topBase + walls + grid;
}

function saddleClips(heightField) {
  const field = heightField.field;
  const width = heightField.width;
  const height = heightField.height;
  const clips = new Uint8Array(width * height * 4);
  for (let row = 1; row < height; row += 1) {
    for (let col = 1; col < width; col += 1) {
      const nw = field[(row - 1) * width + (col - 1)];
      const ne = field[(row - 1) * width + col];
      const sw = field[row * width + (col - 1)];
      const se = field[row * width + col];
      if (Math.min(nw, se) > Math.max(ne, sw) || Math.min(ne, sw) > Math.max(nw, se)) {
        setClip(clips, width, row - 1, col - 1, SE);
        setClip(clips, width, row - 1, col, SW);
        setClip(clips, width, row, col - 1, NE);
        setClip(clips, width, row, col, NW);
      }
    }
  }
  return clips;
}

function meshTriangleCountFromClips(heightField, clips) {
  const field = heightField.field;
  const width = heightField.width;
  const height = heightField.height;
  let total = 0;
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const index = row * width + col;
      const corners = clipValue(clips, width, row, col, NW)
        + clipValue(clips, width, row, col, NE)
        + clipValue(clips, width, row, col, SE)
        + clipValue(clips, width, row, col, SW);
      const polygonVertices = 4 + corners;
      total += 2 * (polygonVertices - 2);
      total += 2 * corners * field[index];
    }
  }
  total += gridSideTriangleCount(heightField, clips);
  return total;
}

function gridSideTriangleCount(heightField, clips) {
  const field = heightField.field;
  const width = heightField.width;
  const height = heightField.height;
  let total = 0;
  for (let row = 0; row < height; row += 1) {
    for (let edgeCol = 0; edgeCol < width + 1; edgeCol += 1) {
      const left = edgeCol > 0 ? sideInterval(clips, width, row, edgeCol - 1, EAST) : null;
      const right = edgeCol < width ? sideInterval(clips, width, row, edgeCol, WEST) : null;
      const leftHeight = edgeCol > 0 ? field[row * width + edgeCol - 1] : 0;
      const rightHeight = edgeCol < width ? field[row * width + edgeCol] : 0;
      total += sharedEdgeTriangleCount(left, leftHeight, right, rightHeight);
    }
  }
  for (let edgeRow = 0; edgeRow < height + 1; edgeRow += 1) {
    for (let col = 0; col < width; col += 1) {
      const upper = edgeRow > 0 ? sideInterval(clips, width, edgeRow - 1, col, SOUTH) : null;
      const lower = edgeRow < height ? sideInterval(clips, width, edgeRow, col, NORTH) : null;
      const upperHeight = edgeRow > 0 ? field[(edgeRow - 1) * width + col] : 0;
      const lowerHeight = edgeRow < height ? field[edgeRow * width + col] : 0;
      total += sharedEdgeTriangleCount(upper, upperHeight, lower, lowerHeight);
    }
  }
  return total;
}

function sharedEdgeTriangleCount(first, firstHeight, second, secondHeight) {
  if (first !== null && second !== null) {
    let total = 0;
    const overlap = intersectInterval(first, second);
    if (overlap !== null && firstHeight !== secondHeight) {
      total += 2 * Math.abs(firstHeight - secondHeight);
    }
    total += 2 * firstHeight * subtractInterval(first, second).length;
    total += 2 * secondHeight * subtractInterval(second, first).length;
    return total;
  }
  if (first !== null) {
    return 2 * firstHeight;
  }
  if (second !== null) {
    return 2 * secondHeight;
  }
  return 0;
}

function sideInterval(clips, width, row, col, side) {
  let start;
  let end;
  if (side === NORTH) {
    start = reliefStart(clipValue(clips, width, row, col, NW));
    end = reliefEnd(clipValue(clips, width, row, col, NE));
  } else if (side === EAST) {
    start = reliefStart(clipValue(clips, width, row, col, NE));
    end = reliefEnd(clipValue(clips, width, row, col, SE));
  } else if (side === SOUTH) {
    start = reliefStart(clipValue(clips, width, row, col, SW));
    end = reliefEnd(clipValue(clips, width, row, col, SE));
  } else if (side === WEST) {
    start = reliefStart(clipValue(clips, width, row, col, NW));
    end = reliefEnd(clipValue(clips, width, row, col, SW));
  } else {
    throw new Error("unknown side");
  }
  if (start >= end) {
    return null;
  }
  return [start, end];
}

function reliefStart(clipped) {
  return clipped ? RELIEF_UNITS : 0;
}

function reliefEnd(clipped) {
  return clipped ? XY_UNITS - RELIEF_UNITS : XY_UNITS;
}

function intersectInterval(first, second) {
  const start = Math.max(first[0], second[0]);
  const end = Math.min(first[1], second[1]);
  if (start >= end) {
    return null;
  }
  return [start, end];
}

function subtractInterval(first, second) {
  const overlap = intersectInterval(first, second);
  if (overlap === null) {
    return [first];
  }
  const pieces = [];
  if (first[0] < overlap[0]) {
    pieces.push([first[0], overlap[0]]);
  }
  if (overlap[1] < first[1]) {
    pieces.push([overlap[1], first[1]]);
  }
  return pieces;
}

function setClip(clips, width, row, col, corner) {
  clips[(row * width + col) * 4 + corner] = 1;
}

function clipValue(clips, width, row, col, corner) {
  return clips[(row * width + col) * 4 + corner];
}

export function resolveOrientation(orientation, table, layerParams = DEFAULT_LAYER) {
  const layer = normalizeLayer(layerParams);
  if (!["auto", "dark", "bright"].includes(orientation)) {
    throw new Error("orientation must be auto, dark, or bright");
  }
  if (orientation !== "auto") {
    return orientation;
  }
  const low = relativeLuminance(table.layer_rgb[layer.base_layers]);
  const high = relativeLuminance(table.layer_rgb[layer.total_layers]);
  return high <= low ? "dark" : "bright";
}

export function normalizeRgbaImage(image) {
  if (!image || !Number.isInteger(image.width) || !Number.isInteger(image.height)) {
    throw new Error("image width and height are required");
  }
  const raw = image.rgba || image.data;
  if (!raw || raw.length !== image.width * image.height * 4) {
    throw new Error("image RGBA data length mismatch");
  }
  return {
    width: image.width,
    height: image.height,
    data: raw instanceof Uint8ClampedArray ? raw : new Uint8ClampedArray(raw),
  };
}

function normalizeLayer(layer) {
  return {
    layer_height_mm: Number(layer.layer_height_mm ?? layer.layerHeightMm ?? DEFAULT_LAYER.layer_height_mm),
    base_layers: Number.parseInt(layer.base_layers ?? layer.baseLayers ?? DEFAULT_LAYER.base_layers, 10),
    total_layers: Number.parseInt(layer.total_layers ?? layer.totalLayers ?? DEFAULT_LAYER.total_layers, 10),
  };
}

function normalizeMesh(mesh) {
  return {
    max_triangles: Number(mesh.max_triangles ?? mesh.maxTriangles ?? DEFAULT_MESH.max_triangles),
    flatten_transparent: Boolean(mesh.flatten_transparent ?? mesh.flattenTransparent ?? DEFAULT_MESH.flatten_transparent),
    orientation: mesh.orientation || DEFAULT_MESH.orientation,
  };
}

function normalizeLuminance(luminance) {
  const black = Number(luminance.black_point ?? luminance.black ?? DEFAULT_LUMINANCE.black_point);
  const white = Number(luminance.white_point ?? luminance.white ?? DEFAULT_LUMINANCE.white_point);
  const gamma = Number(luminance.gamma ?? DEFAULT_LUMINANCE.gamma);
  if (!Number.isFinite(black) || !Number.isFinite(white) || !Number.isFinite(gamma)) {
    throw new Error("luminance params must be finite");
  }
  if (!(0 <= black && black < white && white <= 1)) {
    throw new Error("lum_black must be >= 0, lum_white must be <= 1, and lum_black must be less than lum_white");
  }
  if (!(0.1 <= gamma && gamma <= 5.0)) {
    throw new Error("lum_gamma must be between 0.1 and 5.0");
  }
  return { black_point: black, white_point: white, gamma };
}

function colorTemperatureToRgb(kelvin) {
  const temp = Math.max(1000, Math.min(12000, kelvin)) / 100;
  let red;
  let green;
  let blue;
  if (temp <= 66) {
    red = 255;
    green = 99.4708025861 * Math.log(temp) - 161.1195681661;
    blue = temp <= 19 ? 0 : 138.5177312231 * Math.log(temp - 10) - 305.0447927307;
  } else {
    red = 329.698727446 * ((temp - 60) ** -0.1332047592);
    green = 288.1221695283 * ((temp - 60) ** -0.0755148492);
    blue = 255;
  }
  return [clampChannel(red), clampChannel(green), clampChannel(blue)];
}
