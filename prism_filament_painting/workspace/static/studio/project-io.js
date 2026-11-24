export function bindProjectIo({ getState, setState, onNotice, getProjectId = state => state.image?.project_id }) {
  const saveButton = document.getElementById("studio-save");
  const list = document.getElementById("project-list");
  if (!saveButton || !list) {
    return;
  }

  saveButton.addEventListener("click", async () => {
    if (list.inert) return;
    const state = getState();
    const projectId = getProjectId(state);
    const path = projectId ? `/studio/projects/${encodeURIComponent(projectId)}` : "/studio/projects";
    if (projectId && !await window.PrismLibrary.ask(
      `Replace the saved version of “${state.image?.filename || projectId}” with these changes?`, "Replace project"
    )) return;
    // Refresh replaces the list buttons. Keep them out of the focus order
    // until the saved list is ready, including when a write fails.
    list.inert = true;
    try {
      const payload = await requestJson(path, {
        method: "POST",
        body: { state },
      });
      await setState(payload.state);
      onNotice("Project saved.");
      await refreshProjects(list, setState, onNotice);
    } catch (error) {
      onNotice(String(error.message || error), true);
    } finally {
      list.inert = false;
    }
  });

  refreshProjects(list, setState, onNotice).catch((error) => onNotice(String(error.message || error), true));
}

async function refreshProjects(list, setState, onNotice) {
  const payload = await requestJson("/studio/projects");
  renderProjects(list, payload.projects || [], setState, onNotice);
}

function renderProjects(list, projects, setState, onNotice) {
  list.replaceChildren();
  if (projects.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No projects saved.";
    list.append(empty);
    return;
  }
  for (const project of projects) {
    const row = document.createElement("div");
    row.className = "project-row";
    const meta = document.createElement("div");
    meta.className = "project-meta";
    const title = document.createElement("strong");
    title.textContent = project.filename || project.id;
    const palette = document.createElement("span");
    palette.textContent = project.palette || project.id;
    meta.append(title, palette);

    const actions = document.createElement("div");
    actions.className = "row-actions";
    const load = document.createElement("button");
    load.type = "button";
    load.className = "secondary";
    load.textContent = "Load";
    load.addEventListener("click", async () => {
      try {
        const payload = await requestJson(`/studio/projects/${encodeURIComponent(project.id)}?version=2`);
        if ((payload.state.project_type || "classic") !== (document.body.dataset.projectType || "classic")) {
          window.location.assign(`/studio?project_id=${encodeURIComponent(project.id)}`);
          return;
        }
        await setState(payload.state);
        // Surface the legacy-clamp migration notice (if any) so a silently-adjusted layer
        // count isn't overwritten on the next save without the user knowing.
        onNotice(payload.notice || "Project loaded.");
      } catch (error) {
        onNotice(String(error.message || error), true);
      }
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      if (!await window.PrismLibrary.ask(
        `Delete the saved project “${project.filename || project.id}” and its saved image?`, "Delete project"
      )) return;
      list.inert = true;
      try {
        await requestJson(`/studio/projects/${encodeURIComponent(project.id)}`, {
          method: "DELETE",
        });
        await refreshProjects(list, setState, onNotice);
        onNotice("Project deleted.");
      } catch (error) {
        onNotice(String(error.message || error), true);
      } finally {
        list.inert = false;
      }
    });
    actions.append(load, remove);
    row.append(meta, actions);
    list.append(row);
  }
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || "request failed");
  }
  return payload;
}
