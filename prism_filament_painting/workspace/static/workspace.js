"use strict";
window.PrismLibrary = { ask: async message => window.confirm(message) };
const byId = id => document.getElementById(id);

// The suggestions page switches between an automatic and an explicit material group.
function showInventoryMaterial(show) {
  const fields = byId("material-fields");
  if (show === fields.hasChildNodes()) return;
  byId("automatic-material").disabled = show;
  if (show) fields.append(byId("material-controls").content.cloneNode(true));
  else fields.replaceChildren();
  byId("material-availability").textContent = show ? "Material selection is available." : "Material is chosen automatically.";
}
document.addEventListener("change", event => {
  const select = event.target;
  if (select.id === "print-material") {
    byId("material-note").textContent = select.value === "mixed" ? select.dataset.mixed :
      select.value === "auto" ? select.dataset.auto : "You chose " + select.value + ".";
  }
});

async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new Error("Prism could not be reached. Check that the workspace is running and try again.");
  }
  if (!response.headers.get("content-type")?.includes("application/json")) {
    throw new Error("Prism could not finish this request. Try again.");
  }
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Check the supplied values.");
  return data;
}
function listItem(text, href) {
  const item = document.createElement("li");
  if (href) {
    const link = document.createElement("a");
    link.href = href;
    link.textContent = text;
    item.append(link);
  } else item.textContent = text;
  return item;
}
function submit(id, status, action) {
  const form = byId(id);
  if (!form) return;
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const button = form.querySelector("button");
    button.disabled = true;
    try { await action(new FormData(form)); }
    catch (error) { byId(status).textContent = error.message; }
    finally { button.disabled = false; }
  });
}
async function library() {
  const rows = await request("/api/filaments");
  const items = rows.map(row => {
    const item = listItem(row.brand + " · " + row.name);
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.backgroundColor = row.hex;
    swatch.setAttribute("aria-hidden", "true");
    item.prepend(swatch);
    const detail = document.createElement("small");
    detail.textContent = row.material + " · " + row.hex + " · TD " + row.td + " mm · Spool #" + row.id;
    item.append(detail);
    return item;
  });
  byId("spools").replaceChildren(...items);
  if (!rows.length) byId("spools").append(listItem("Your library is empty. Add your first spool above."));
}
if (byId("spools")) library().catch(error => { byId("library-status").textContent = error.message; });
submit("spool", "library-status", async data => {
  const fields = Object.fromEntries(data);
  fields.td = Number(fields.td);
  await request("/api/filaments", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(fields)});
  byId("library-status").textContent = fields.name + " added.";
  await library();
});
for (const [id, path] of [["import-spools", "import"], ["import-readings", "readings"]]) {
  submit(id, "library-status", async data => {
    const result = await request("/api/filaments/" + path, {method: "POST", body: data});
    if (path === "readings") {
      byId("library-status").textContent = "New meter readings saved: " + result.saved + ".";
    } else {
      const skipped = result.errors.map(([row, reason]) => "Row " + row + ": " + reason);
      byId("library-status").textContent = "Spools added: " + result.inserted + ". Updated: " + result.updated +
        ". Skipped: " + result.skipped + ". " + skipped.join(". ");
    }
    await library();
  });
}
if (byId("rank")) {
  const image = byId("rank").elements.image;
  const sample = byId("use-sample");
  sample.addEventListener("change", () => { image.required = !sample.checked; });
  image.addEventListener("change", () => {
    if (image.files.length) {
      sample.checked = false;
      image.required = true;
    }
  });
}
submit("rank", "rank-status", async data => {
  byId("matches").hidden = true;
  byId("ranked").replaceChildren();
  byId("rank-status").textContent = "Comparing layer colours with your picture…";
  if (byId("use-sample").checked) {
    const sample = await fetch("/sample/picture.png");
    if (!sample.ok) throw new Error("The sample picture could not be opened. Choose your own picture or try again.");
    data.set("image", await sample.blob(), "prism-sample.png");
  }
  const { candidates } = await request("/api/rank", {method: "POST", body: data});
  byId("rank-status").textContent = candidates.length ? "Comparison ready." : "No palettes found. Try fewer spools.";
  byId("ranked").replaceChildren(...candidates.map(candidate => {
    const palette = candidate.state.palette;
    const item = listItem(palette.filaments.map(spool => spool.name).join(" → "), candidate.studio_url);
    item.append(" · Colour error " + palette.score.toFixed(2));
    return item;
  }));
  byId("matches").hidden = !candidates.length;
  if (candidates.length) byId("matches-title").focus();
});
if (byId("saved")) request("/studio/projects").then(({ projects }) => {
  byId("saved").replaceChildren(...projects.map(p => listItem(p.filename, "/studio?project_id=" + encodeURIComponent(p.id))));
  if (!projects.length) byId("saved").append(listItem("Save a print in the editor to reopen it here."));
}).catch(error => { byId("saved").textContent = error.message; });
