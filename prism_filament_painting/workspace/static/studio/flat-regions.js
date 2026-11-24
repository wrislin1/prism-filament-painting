// Each saved mask spans the whole oriented image. Resizing the print samples
// the original mask at cell centres, including when editing an older project.
export function bindRegions(onChange, notice) {
  const el = id => document.getElementById(id);
  const editor = el("region-editor");
  const selector = el("edit-region");
  const canvas = el("region-canvas");
  const slots = Array.from(editor.querySelectorAll(".region-slot"));
  const ctx = canvas.getContext("2d");
  const undo = el("undo-region");
  let removed = null;
  let regions = [], background = [1, 2, 3, 4], source = {}, picture = null;
  let index = 0, lastPoint = null;


  function sample(mask, x, y) {
    const row = Math.floor((y + 0.5) * mask.length / canvas.height);
    const col = Math.floor((x + 0.5) * mask[0].length / canvas.width);
    return mask[row][col];
  }

  function repaint() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (picture?.naturalWidth) ctx.drawImage(picture, 0, 0, canvas.width, canvas.height);
    let overlaps = 0, count = 0;
    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const owners = regions.flatMap((r, i) => sample(r.mask, x, y) === "1" ? [i + 1] : []);
        const selected = (owners.at(-1) || 0) === index;
        count += Number(selected);
        if (owners.length > 1) overlaps++;
        if (owners.length > 1 || (index && owners.includes(index))) {
          ctx.fillStyle = owners.length > 1 ? "rgba(255,165,0,.8)" : "rgba(20,130,255,.7)";
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }
    el("region-info").textContent = count + " assigned cells · " + overlaps + " overlapping cells" +
      (index && !count ? ". This region is empty or covered by higher priority regions." : ".");
  }

  function controls() {
    selector.replaceChildren(new Option("Background", "0"),
      ...regions.map((r, i) => new Option(r.name, String(i + 1))));
    selector.value = String(index);
    const active = regions[index - 1];
    slots.forEach(box => { box.checked = (active?.slots ?? background).includes(Number(box.value)); });
    el("region-name-field").hidden = !active;
    el("region-name").value = active?.name || "";
    el("paint-controls").hidden = !active;
    if (active) {
      el("region-column").max = canvas.width;
      el("region-row").max = canvas.height;
      el("cell-grid-size").textContent = canvas.width + " columns × " + canvas.height + " rows.";
    }
    el("raise-region").disabled = !active || index === regions.length;
    el("remove-region").disabled = !active;
    undo.hidden = !removed;
    undo.disabled = regions.length >= 8;
    el("add-region").disabled = regions.length >= 8;
    repaint();
  }
  function changed() {

    onChange();
    controls();
  }
  selector.onchange = () => {  index = Number(selector.value); controls(); };
  el("add-region").onclick = () => {
    regions.push({ name: "Region " + (regions.length + 1), slots: [1, 2, 3, 4],
      mask: Array(canvas.height).fill("0".repeat(canvas.width)) });
    index = regions.length;
    changed();
  };
  el("remove-region").onclick = () => {
    removed = { position: index - 1, region: regions.splice(index - 1, 1)[0] };
    index--;
    changed();
    notice("Region removed.");
    undo.focus();
  };
  undo.onclick = () => {
    index = Math.min(removed.position, regions.length) + 1;
    regions.splice(index - 1, 0, removed.region);
    removed = null;
    changed();
    notice("Region restored.");
    selector.focus();
  };
  el("raise-region").onclick = () => {
    [regions[index - 1], regions[index]] = [regions[index], regions[index - 1]];
    index++;
    changed();
  };
  el("region-name").onchange = () => {
    regions[index - 1].name = el("region-name").value.trim() || "Region " + index;
    changed();
  };
  slots.forEach(box => { box.onchange = () => {
    const palette = slots.filter(s => s.checked).map(s => Number(s.value));
    if (!palette.length) { box.checked = true; notice("Keep at least one loaded slot in this region."); return; }
    if (index) regions[index - 1].slots = palette;
    else background = palette;
    changed();
  }; });

  function paintCell(mask, x, y) {
    mask[y] = mask[y].slice(0, x) + el("region-brush").value + mask[y].slice(x + 1);
  }

  el("apply-cell").onclick = () => {
    const column = el("region-column"), row = el("region-row");
    if (!column.reportValidity() || !row.reportValidity()) return;
    const mask = regions[index - 1].mask;
    // Older projects can retain masks at a different resolution from the preview.
    const x = Math.floor((column.valueAsNumber - 0.5) * mask[0].length / canvas.width);
    const y = Math.floor((row.valueAsNumber - 0.5) * mask.length / canvas.height);
    paintCell(mask, x, y);
    changed();
  };

  function paint(event) {
    const rect = canvas.getBoundingClientRect();
    const mask = regions[index - 1].mask;
    const x = Math.min(mask[0].length - 1, Math.max(0, Math.floor((event.clientX - rect.left) * mask[0].length / rect.width)));
    const y = Math.min(mask.length - 1, Math.max(0, Math.floor((event.clientY - rect.top) * mask.length / rect.height)));
    const [fromX, fromY] = lastPoint ?? [x, y];
    const steps = Math.max(1, Math.abs(x - fromX), Math.abs(y - fromY));
    for (let i = 1; i <= steps; i++) {
      const cx = Math.round(fromX + (x - fromX) * i / steps);
      const cy = Math.round(fromY + (y - fromY) * i / steps);
      paintCell(mask, cx, cy);
    }
    lastPoint = [x, y];
    changed();
  }
  canvas.onpointerdown = event => {
    if (!index || event.button !== 0) return;
    canvas.setPointerCapture(event.pointerId);
    lastPoint = null;
    paint(event);
  };
  canvas.onpointermove = event => { if (canvas.hasPointerCapture(event.pointerId) && lastPoint) paint(event); };
  canvas.onlostpointercapture = () => { lastPoint = null; };

  return {
    options() {
      return {...(regions.length ? {regions} : {}),
        ...(background.length !== 4 ? {background_slots: background} : {})};
    },
    clear() { removed = null;  editor.hidden = true; regions = []; background = [1, 2, 3, 4]; index = 0; },
    load(state) {
      removed = null;

      regions = structuredClone(state.options.regions || []);
      background = [...(state.options.background_slots || [1, 2, 3, 4])];
      index = Math.min(index, regions.length);
      source = state.image;
      canvas.width = state.flat.run.geometry.columns;
      canvas.height = state.flat.run.geometry.rows;
      picture = new Image();
      picture.onload = repaint;
      picture.src = source.project_id ? "/studio/projects/" + encodeURIComponent(source.project_id) + "/image" :
        "/studio/image/" + encodeURIComponent(source.upload_token);
      editor.hidden = false;
      controls();
    },
    busy(value) { editor.disabled = value;  },
  };
}
