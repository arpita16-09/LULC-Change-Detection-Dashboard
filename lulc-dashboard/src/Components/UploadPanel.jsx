import { useMemo, useState } from "react";
import { apiUrl, fetchJSON } from "../api";
import { DEFAULT_AOI_LISTS } from "../lulcConstants";

function UploadPanel({
  selectedAoi,
  onAoiChange,
  onTransition,
  aois,
  layers = { t1: true, t2: true, transition: true },
  result,
}) {
  const lists = useMemo(() => {
    const train = aois?.train?.length ? aois.train : DEFAULT_AOI_LISTS.train;
    const test = aois?.test?.length ? aois.test : DEFAULT_AOI_LISTS.test;
    return {
      train: [...train].sort(),
      test: [...test].sort(),
    };
  }, [aois]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useOscdMask, setUseOscdMask] = useState(true);
  const [useTta, setUseTta] = useState(true);

  const runTransition = async () => {
    if (!selectedAoi) return;
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({
        use_oscd_mask: String(useOscdMask),
        tta: String(useTta),
      });
      const data = await fetchJSON(
        `/predict/transition/${encodeURIComponent(selectedAoi)}?${qs}`
      );
      onTransition?.(data);
    } catch (err) {
      let msg = err.message || "Could not reach backend / run transition.";
      try {
        const parsed = JSON.parse(msg);
        if (parsed.detail) msg = parsed.detail;
      } catch {
        /* keep raw */
      }
      setError(msg);
      onTransition?.(null);
    } finally {
      setLoading(false);
    }
  };

  const previews = [
    { key: "t1_lulc_color", layer: "t1", caption: "t1 LULC" },
    { key: "t2_lulc_color", layer: "t2", caption: "t2 LULC" },
    { key: "transition_color", layer: "transition", caption: "Transitions" },
  ].filter((p) => layers[p.layer] && result?.artifacts?.[p.key]);

  return (
    <div className="section-block">
      <div className="section-header">
        <h3 className="section-title">Run LULC Transition</h3>
        <span className="section-sub">
          OSCD-enhanced ResUNet on t1/t2 → from→to class changes
        </span>
      </div>

      <div className="upload-zone" style={{ cursor: "default" }}>
        <div className="upload-prompt" style={{ gap: "0.85rem", width: "100%" }}>
          <label className="field-label">
            Select AOI
            <select
              className="field-control"
              value={selectedAoi || ""}
              onChange={(e) => onAoiChange(e.target.value)}
            >
              <optgroup label={`Test AOIs (${lists.test.length})`}>
                {lists.test.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </optgroup>
              <optgroup label={`Train AOIs (${lists.train.length})`}>
                {lists.train.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </optgroup>
            </select>
          </label>
          <p className="upload-hint">
            {lists.train.length + lists.test.length} OSCD cities available
          </p>

          <div className="option-row">
            <label className="check-label">
              <input
                type="checkbox"
                checked={useOscdMask}
                onChange={(e) => setUseOscdMask(e.target.checked)}
              />
              Filter with OSCD change mask
            </label>
            <label className="check-label">
              <input
                type="checkbox"
                checked={useTta}
                onChange={(e) => setUseTta(e.target.checked)}
              />
              Test-time augmentation
            </label>
          </div>

          <button
            type="button"
            className="primary-btn"
            onClick={runTransition}
            disabled={loading || !selectedAoi}
          >
            {loading ? "Running model..." : "Run LULC Transition"}
          </button>

          <p className="upload-hint">
            Classes: Other, Water, Urban, Forest, Agriculture, Barren. Transition
            stats are from→to percentages of the AOI.
          </p>
        </div>
      </div>

      {error && <div className="upload-error">⚠️ {error}</div>}

      {result && (
        <div className="upload-result">
          <div className="result-header">
            ✅ {result.aoi}: {result.change_pct}% pixels changed
            {result.used_oscd_change_mask ? " (OSCD-masked)" : ""}
          </div>
          <div className="result-grid">
            <div className="result-item">
              <span className="result-label">Changed px</span>
              <span className="result-val">
                {Number(result.n_changed_pixels || 0).toLocaleString()}
              </span>
            </div>
            <div className="result-item">
              <span className="result-label">Model epoch</span>
              <span className="result-val">{result.model?.epoch ?? "—"}</span>
            </div>
          </div>
          {result.transitions?.length > 0 && (
            <div className="result-classes">
              {result.transitions.slice(0, 8).map((t) => (
                <span key={`${t.from}-${t.to}`} className="result-tag">
                  {t.from} → {t.to}: {t.pct_of_image}%
                </span>
              ))}
            </div>
          )}
          {previews.length > 0 && (
            <div className="result-previews">
              {previews.map((p) => (
                <figure key={p.key}>
                  <img src={apiUrl(result.artifacts[p.key])} alt={p.caption} />
                  <figcaption>{p.caption}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default UploadPanel;
