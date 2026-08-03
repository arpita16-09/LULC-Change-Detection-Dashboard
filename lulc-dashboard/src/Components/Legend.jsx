import { LULC_CLASSES } from "../lulcConstants";

function Legend() {
  return (
    <div className="legend">
      <h3>LULC Legend</h3>
      {LULC_CLASSES.map((c) => (
        <p key={c.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 12,
              height: 12,
              borderRadius: 3,
              background: c.color,
              display: "inline-block",
            }}
          />
          {c.label}
        </p>
      ))}
    </div>
  );
}

export default Legend;
