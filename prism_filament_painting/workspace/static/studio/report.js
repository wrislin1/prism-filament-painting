// A report is computed on demand for the current neutral print preview.
export function bindAccuracyReport(getRequest) {
  const button = document.getElementById("studio-report");
  const status = document.getElementById("report-status");
  const result = document.getElementById("report-result");
  let pending = null;

  button.addEventListener("click", async () => {
    const request = getRequest();
    const controller = new AbortController();
    pending = controller;
    button.disabled = true;
    result.hidden = true;
    status.textContent = "Comparing source pixels with printable colours…";
    try {
      const response = await fetch(`/studio/report?downsample_factor=${request.factor}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request.state),
        signal: controller.signal,
      });
      const report = await response.json();
      if (controller.signal.aborted) return;
      if (!response.ok) throw new Error(report.detail || "Report unavailable");
      if (!report.visible_pixels) {
        status.textContent = "There are no visible source colours inside this plate to score.";
        return;
      }
      const sources = report.prediction_sources;
      document.getElementById("report-score").textContent =
        `Mean colour error ${report.mean_delta_e00.toFixed(2)} ΔE00 · ${sources.difference.join(", ")}`;
      const scale = report.peak_delta_e00 === 0
        ? "all visible pixels match"
        : `black at 0 to amber at ${report.peak_delta_e00.toFixed(2)} ΔE00`;
      document.getElementById("report-heat-legend").textContent =
        `Difference · ${scale} · ${sources.difference.join(", ")}`;
      const select = document.getElementById("reach-level");
      const selected = select.value || "5";
      const levels = report.reach_levels;
      select.replaceChildren(...levels.map(level => new Option(level.label + " · " + level.threshold + " ΔE00", level.threshold)));
      select.value = selected;
      document.getElementById("reach-shares").textContent = levels.map(level =>
        level.label + " " + (100 * level.pixels / report.visible_pixels).toFixed(1) + "%"
      ).join(" · ");
      const showLevel = () => {
        const level = levels.find(item => item.threshold === Number(select.value));
        const printable = request.state.project_type === "flat"
          ? "Every recipe the selected plan allows at that pixel" : "Every printable height in this order and swap schedule";
        document.getElementById("report-flag-legend").textContent =
          `Magenta marks ${level.pixels.toLocaleString()} of ${report.visible_pixels.toLocaleString()} visible pixels. ` +
          `${printable} differs by more than ${level.threshold} ΔE00 there. Prediction source: ${sources.reachability.join(", ")}.`;
        document.getElementById("report-flags").src = level.map;
      };
      select.onchange = showLevel;
      showLevel();
      document.getElementById("report-heat").src = report.heat_map;
      status.textContent = "";
      result.hidden = false;
    } catch (error) {
      if (!controller.signal.aborted) status.textContent = error.message || "Report unavailable";
    } finally {
      if (pending === controller) {
        pending = null;
        button.disabled = false;
      }
    }
  });

  return function invalidate() {
    pending?.abort();
    pending = null;
    status.textContent = "";
    result.hidden = true;
    button.disabled = true;
  };
}
