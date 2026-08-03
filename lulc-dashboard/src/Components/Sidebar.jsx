import { LULC_CLASSES, TRANSITION_LAYER_META } from "../lulcConstants";

const LAYER_IDS = ["t1", "t2", "transition"];

function Sidebar({ layers, onToggleLayer, modelInfo }) {
  return (
    <div className="sidebar">
      <div className="section-header">
        <h3 className="section-title">Result layers</h3>
        <span className="section-sub">Show / hide prediction maps</span>
      </div>

      <div className="layer-list">
        {LAYER_IDS.map((id) => {
          const meta = TRANSITION_LAYER_META[id];
          const on = Boolean(layers?.[id]);
          return (
            <div
              key={id}
              className={`layer-item ${on ? "layer-active" : "layer-inactive"}`}
              onClick={() => onToggleLayer?.(id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onToggleLayer?.(id);
              }}
            >
              <span className="layer-label">{meta.label}</span>
              <div
                className="layer-toggle"
                style={{ background: on ? meta.color : "#334155" }}
              >
                <div className={`toggle-knob ${on ? "knob-on" : "knob-off"}`} />
              </div>
            </div>
          );
        })}
      </div>

      <div className="legend">
        <p className="legend-title">LULC classes (model)</p>
        {LULC_CLASSES.map((item) => (
          <div key={item.key} className="legend-item">
            <span className="legend-dot" style={{ background: item.color }} />
            <span className="legend-label">
              {item.id}. {item.label}
            </span>
          </div>
        ))}
      </div>

      <div className="legend" style={{ marginTop: 12 }}>
        <p className="legend-title">Model</p>
        <p className="legend-label" style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.45 }}>
          {modelInfo?.checkpoint || "resunet_oscd_enhanced_best.pt"}
          {modelInfo?.val_miou != null && (
            <>
              <br />
              val mIoU {Number(modelInfo.val_miou).toFixed(3)}
            </>
          )}
          <br />
          Task: LULC transition (t1 → t2)
        </p>
      </div>
    </div>
  );
}

export default Sidebar;
