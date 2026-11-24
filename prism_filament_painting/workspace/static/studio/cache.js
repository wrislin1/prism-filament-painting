const TABLE_CHANGES = new Set(["palette", "table", "layer", "swap"]);
const FIELD_CHANGES = new Set(["field", "luminance"]);

export function invalidatePreviewCaches(cache, changeType) {
  const next = { ...cache };
  if (TABLE_CHANGES.has(changeType)) {
    next.forwardTable = null;
    next.heightField = null;
    return next;
  }
  if (FIELD_CHANGES.has(changeType)) {
    next.heightField = null;
  }
  return next;
}
