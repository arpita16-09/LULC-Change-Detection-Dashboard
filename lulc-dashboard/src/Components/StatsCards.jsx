import { LULC_CLASSES } from "../lulcConstants";

function normalizeStats(raw) {
  if (!raw) return null;
  const out = {};
  for (const [k, v] of Object.entries(raw)) {
    let key = k.toLowerCase().replace(/\s+/g, "_");
    if (key === "bare_soil") key = "barren";
    out[key] = Number(v);
  }
  return out;
}

function fmt(value) {
  return Number(value).toFixed(1);
}

function StatsCards({ t1Stats, t2Stats, stats: legacyStats }) {
  // Prefer explicit t1/t2; fall back to legacy single stats as t2-only
  const t1 = normalizeStats(t1Stats);
  const t2 = normalizeStats(t2Stats || (!t1Stats ? legacyStats : null));
  const empty = !t1 && !t2;
  const cards = LULC_CLASSES;

  const allVals = cards.flatMap((c) => [
    Number(t1?.[c.key] ?? 0),
    Number(t2?.[c.key] ?? 0),
  ]);
  const maxVal = Math.max(1, ...allVals);

  return (
    <div className="stats-container">
      {cards.map((meta) => {
        const v1 = t1 ? Number(t1[meta.key] ?? 0) : null;
        const v2 = t2 ? Number(t2[meta.key] ?? 0) : null;
        const delta =
          v1 != null && v2 != null ? Number((v2 - v1).toFixed(1)) : null;

        return (
          <div
            className="card"
            key={meta.key}
            style={{ "--card-accent": meta.color }}
          >
            <div className="card-head">
              <span
                className="card-icon"
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: 4,
                  background: meta.color,
                  display: "inline-block",
                }}
              />
              <h3 className="card-title">{meta.label}</h3>
            </div>

            {empty ? (
              <p className="card-value">
                — <span className="card-unit">run model</span>
              </p>
            ) : (
              <>
                <div className="card-dual">
                  <div className="card-time">
                    <span className="card-time-label">t1</span>
                    <span className="card-time-value">
                      {v1 == null ? "—" : `${fmt(v1)}%`}
                    </span>
                  </div>
                  <div className="card-time">
                    <span className="card-time-label">t2</span>
                    <span className="card-time-value">
                      {v2 == null ? "—" : `${fmt(v2)}%`}
                    </span>
                  </div>
                </div>

                {delta != null && (
                  <p
                    className={`card-delta ${
                      delta > 0 ? "up" : delta < 0 ? "down" : "flat"
                    }`}
                  >
                    {delta > 0 ? "+" : ""}
                    {fmt(delta)} pp (t2 − t1)
                  </p>
                )}

                <div className="card-bars">
                  <div className="card-bar-row">
                    <span className="card-bar-tag">t1</span>
                    <div className="card-bar">
                      <div
                        className="card-bar-fill"
                        style={{
                          width: `${Math.min(100, ((v1 ?? 0) / maxVal) * 100)}%`,
                          background: meta.color,
                          opacity: 0.55,
                        }}
                      />
                    </div>
                  </div>
                  <div className="card-bar-row">
                    <span className="card-bar-tag">t2</span>
                    <div className="card-bar">
                      <div
                        className="card-bar-fill"
                        style={{
                          width: `${Math.min(100, ((v2 ?? 0) / maxVal) * 100)}%`,
                          background: meta.color,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default StatsCards;
