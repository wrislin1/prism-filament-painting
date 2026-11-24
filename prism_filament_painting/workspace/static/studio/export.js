export function bindExport(getState, getPrintBaseFactor, onOptimized) {
  const button = document.getElementById("studio-export");
  const result = document.getElementById("export-result");
  if (!button || !result) {
    return;
  }
  let pending = null;
  button.addEventListener("click", async () => {
    const controller = new AbortController();
    pending = controller;
    button.disabled = true;
    result.textContent = "Preparing print files…";
    try {
      const options = {
        state: getState(),
        print_base_factor: getPrintBaseFactor?.(),
        mode: document.getElementById("cpu-optimize")?.checked ? "optimize" : "naive",
        include_band_pieces: document.getElementById("band-pieces").checked,
      };
      const file = document.getElementById("print-settings").files[0];
      if (file) {
        const form = new FormData();
        form.append("file", file);
        const parsed = await fetch("/api/print-settings", {method: "POST", body: form, signal: controller.signal});
        const settings = await parsed.json();
        if (!parsed.ok) throw new Error(settings.detail || "Invalid slicer settings");
        options.project_settings = settings;
      }
      const response = await fetch("/studio/generate", {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(options),
      });
      const payload = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) {
        throw new Error(payload.detail || "export failed");
      }
      renderArtifactLinks(result, payload);
      onOptimized?.(payload.metadata?.optimize);
    } catch (error) {
      if (!controller.signal.aborted) {
        result.textContent = "Could not prepare the print files. " + error.message + " Check the export options and try again.";
      }
    } finally {
      if (pending === controller) {
        pending = null;
        button.disabled = false;
      }
    }
  });
  return () => {
    pending?.abort();
    pending = null;
    result.replaceChildren();
  };
}

function renderArtifactLinks(container, payload) {
  const ready = document.createElement("p");
  ready.textContent = "Your print files are ready. Download the shape and filament change list for your slicer.";
  container.replaceChildren(ready);
  const links = [
    ["Print mesh (STL)", payload.stl_url, `prism-${payload.run_id}.stl`],
    ["Print preview (PNG)", payload.output_url, `prism-${payload.run_id}.png`],
    ["Filament change layers", payload.swap_url, `prism-${payload.run_id}.swaps.txt`],
  ];
  for (const piece of payload.bands?.pieces || []) {
    if (piece.stl_url) links.push([piece.filament, piece.stl_url, piece.stl_url.split("/").pop()]);
  }
  if (payload.project_url) {
    links.push(["3MF · filament changes", payload.project_url, `prism-${payload.run_id}.3mf`]);
  }
  if (payload.pause_project_url) {
    links.push(["3MF · manual pauses", payload.pause_project_url, `prism-${payload.run_id}.pause.3mf`]);
  }
  if (payload.describe_url) {
    links.push(["Print description", payload.describe_url, `prism-${payload.run_id}.describe.txt`]);
  }
  for (const [label, href, download] of links) {
    const anchor = document.createElement("a");
    anchor.className = "btn";
    anchor.href = href;
    anchor.download = download;
    anchor.textContent = label;
    container.append(anchor);
  }
  if (payload.metadata?.project_note) {
    const note = document.createElement("p");
    note.textContent = payload.metadata.project_note;
    container.append(note);
  }
  if (payload.project_url || payload.pause_project_url) {
    const note = document.createElement("p");
    note.textContent = "The 3MF uses your selected layer height for the first layer too. " +
      "At the 0.08 mm default, good first-layer adhesion can be difficult. Check your build plate and first layer before printing.";
    container.append(note);
  }
}
