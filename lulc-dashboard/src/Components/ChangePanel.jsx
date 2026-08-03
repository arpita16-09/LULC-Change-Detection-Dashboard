const CHANGE_META = [
  { key: "forest_to_urban", label: "Forest → Urban", color: "#dc2626" },
  { key: "agriculture_to_urban", label: "Agriculture → Urban", color: "#ea580c" },
  { key: "water_to_urban", label: "Water → Urban", color: "#db2777" },
  { key: "barren_to_urban", label: "Barren → Urban", color: "#b91c1c" },
  { key: "forest_to_agriculture", label: "Forest → Agriculture", color: "#65a30d" },
  { key: "agriculture_to_forest", label: "Agriculture → Forest", color: "#15803d" },
  { key: "vegetation_loss", label: "Vegetation loss", color: "#a16207" },
];

function ChangePanel({ aoi, transition }) {
  const changes = transition?.changes || transition?.summary || null;
  const vals = CHANGE_META.map((m) => Number(changes?.[m.key] ?? 0));
  const maxVal = Math.max(0.01, ...vals);

  return (
    <div className="change-panel">
      <div className="section-header">
        <h3 className="section-title">LULC Transitions</h3>
        <span className="section-sub">{aoi ? `${aoi} · t1 → t2` : "from → to"}</span>
      </div>

      {!transition && (
        <p className="loading-text">
          Select an AOI and run the model to see Forest→Urban and other transitions.
        </p>
      )}

      {transition && (
        <p className="loading-text" style={{ marginBottom: "0.75rem" }}>
          Overall change: <strong>{transition.change_pct}%</strong>
          {transition.used_oscd_change_mask ? " · OSCD change mask on" : ""}
        </p>
      )}

      {changes && (
        <ul className="change-list">
          {CHANGE_META.map((meta) => {
            const val = Number(changes[meta.key] ?? 0);
            return (
              <li key={meta.key} className="change-item">
                <span className="change-label">{meta.label}</span>
                <span className="change-value" style={{ color: meta.color }}>
                  {val.toFixed(2)}%
                </span>
                <div className="change-bar-bg">
                  <div
                    className="change-bar-fill"
                    style={{
                      width: `${Math.min(100, (val / maxVal) * 100)}%`,
                      background: meta.color,
                    }}
                  />
                </div>
              </li>
            );
          })}
        </ul>
      )}

      {transition?.transitions?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <p className="legend-title">Top raw transitions</p>
          <ul className="change-list">
            {transition.transitions.slice(0, 5).map((t) => (
              <li key={`${t.from}-${t.to}`} className="change-item">
                <span className="change-label">
                  {t.from} → {t.to}
                </span>
                <span className="change-value">{t.pct_of_image}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ChangePanel;
