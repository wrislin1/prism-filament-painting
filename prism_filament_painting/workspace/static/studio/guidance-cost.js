// Compare printable heights on a small raster of the current stack while editing.
// The export supplies a second comparison after searching filament orders and swaps.
import { relativeLuminance } from "./engine.js";

function sample(image, width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, width, height);
  return ctx.getImageData(0, 0, width, height).data;
}

function targetRgb(rgb, settings) {
  const { black_point, white_point, gamma } = settings;
  if (black_point === 0 && white_point === 1 && gamma === 1) return rgb;
  const lum = relativeLuminance(rgb);
  const target = Math.max(0, Math.min(1, (lum - black_point) / (white_point - black_point))) ** gamma;
  return rgb.map(channel => {
    const x = channel / 255;
    const linear = x <= 0.04045 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
    const y = lum ? Math.min(1, linear * target / lum) : 0;
    return Math.round(255 * (y <= 0.0031308 ? 12.92 * y : 1.055 * y ** (1 / 2.4) - 0.055));
  });
}

export function previewCost(state, source, map, focus, table) {
  const scale = Math.min(1, 96 / Math.max(source.naturalWidth, source.naturalHeight));
  const w = Math.max(1, Math.round(source.naturalWidth * scale));
  const h = Math.max(1, Math.round(source.naturalHeight * scale));
  const pixels = sample(source, w, h);
  const depth = sample(map, w, h);
  const mask = sample(focus, w, h);
  const base = state.layer.base_layers;
  const span = state.layer.total_layers - base;
  let count = 0, sumWeight = 0, off = 0, guided = 0;
  for (let i = 0; i < pixels.length; i += 4) {
    if (!pixels[i + 3] && state.mesh.flatten_transparent) continue;
    sumWeight += 1 + 99 * mask[i] / 255;
    count++;
  }
  if (!count) return "The image has no visible colours to compare.";
  for (let i = 0; i < pixels.length; i += 4) {
    if (!pixels[i + 3] && state.mesh.flatten_transparent) continue;
    const rgb = targetRgb([pixels[i], pixels[i + 1], pixels[i + 2]], state.luminance);
    const near = Math.round(0.299 * depth[i] + 0.587 * depth[i + 1] + 0.114 * depth[i + 2]) / 255;
    const weight = (1 + 99 * mask[i] / 255) * count / sumWeight;
    let nearest = Infinity, best = Infinity, chosen = 0;
    for (let layer = base; layer <= state.layer.total_layers; layer++) {
      const c = table.layer_rgb[layer];
      const error = ((c[0] - rgb[0]) ** 2 + (c[1] - rgb[1]) ** 2 + (c[2] - rgb[2]) ** 2) / (3 * 255 ** 2);
      const cost = weight * error + state.height_strength * ((layer - base) / span - near)**2;
      nearest = Math.min(nearest, error);
      if (cost < best) { best = cost; chosen = error; }
    }
    off += nearest;
    guided += chosen;
  }
  return costText(off / count, guided / count, "Current stack estimate");
}

export function costText(off, guided, label) {
  const change = (Math.sqrt(guided) - Math.sqrt(off)) * 255;
  const amount = Math.abs(change);
  const words = amount < 0.05 ? "no measurable change in colour match" :
    `colours are about ${amount.toFixed(1)} shade steps ${change > 0 ? "farther from" : "closer to"} the image than with depth off`;
  return `${label}: ${words} (0–255 per colour channel). Colour error ${off.toFixed(5)} → ${guided.toFixed(5)}.`;
}
