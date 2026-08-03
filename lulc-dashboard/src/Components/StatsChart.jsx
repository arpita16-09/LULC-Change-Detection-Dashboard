import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";
import { Bar } from "react-chartjs-2";
import { LULC_CLASSES } from "../lulcConstants";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

function normalizeStats(raw) {
  if (!raw) return {};
  const out = {};
  for (const [k, v] of Object.entries(raw)) {
    let key = k.toLowerCase().replace(/\s+/g, "_");
    if (key === "bare_soil") key = "barren";
    out[key] = Number(v);
  }
  return out;
}

function StatsChart({ t1Stats, t2Stats, stats: legacyStats }) {
  const t1 = normalizeStats(t1Stats);
  const t2 = normalizeStats(t2Stats || (!t1Stats ? legacyStats : null));
  const keys = LULC_CLASSES.map((c) => c.key);
  const t1Values = keys.map((k) => t1[k] ?? 0);
  const t2Values = keys.map((k) => t2[k] ?? 0);
  const hasT1 = Object.keys(t1).length > 0;
  const hasT2 = Object.keys(t2).length > 0;
  const hasData = hasT1 || hasT2;

  const datasets = [];
  if (hasT1) {
    datasets.push({
      label: "t1 (%)",
      data: t1Values,
      backgroundColor: LULC_CLASSES.map((c) => `${c.color}99`),
      borderRadius: 6,
      borderWidth: 0,
    });
  }
  if (hasT2) {
    datasets.push({
      label: "t2 (%)",
      data: t2Values,
      backgroundColor: LULC_CLASSES.map((c) => c.color),
      borderRadius: 6,
      borderWidth: 0,
    });
  }
  if (!hasData) {
    datasets.push({
      label: "Share (%)",
      data: keys.map(() => 0),
      backgroundColor: LULC_CLASSES.map((c) => c.color),
      borderRadius: 6,
      borderWidth: 0,
    });
  }

  const data = {
    labels: LULC_CLASSES.map((c) => c.label),
    datasets,
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: "y",
    plugins: {
      legend: {
        display: hasT1 && hasT2,
        labels: { color: "#94a3b8", boxWidth: 12 },
      },
      tooltip: {
        callbacks: {
          label: (ctx) => ` ${ctx.dataset.label}: ${Number(ctx.parsed.x).toFixed(2)} %`,
        },
      },
    },
    scales: {
      x: {
        beginAtZero: true,
        suggestedMax: hasData ? undefined : 100,
        grid: { color: "rgba(255,255,255,0.06)" },
        ticks: { color: "#94a3b8", font: { size: 12 } },
        title: { display: true, text: "Share of AOI (%)", color: "#64748b" },
      },
      y: {
        grid: { display: false },
        ticks: { color: "#cbd5e1", font: { size: 13, weight: "500" } },
      },
    },
  };

  return (
    <div style={{ width: "100%", height: "300px", padding: "10px" }}>
      {!hasData && (
        <p className="loading-text" style={{ marginBottom: 8 }}>
          No prediction yet — run LULC Transition for an AOI.
        </p>
      )}
      <Bar data={data} options={options} />
    </div>
  );
}

export default StatsChart;
