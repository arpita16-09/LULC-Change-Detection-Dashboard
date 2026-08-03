import { useState, useEffect } from "react";
import StatsCards from "./StatsCards";
import StatsChart from "./StatsChart";
import MapView from "./MapView";
import ChangePanel from "./ChangePanel";
import Sidebar from "./Sidebar";
import UploadPanel from "./UploadPanel";
import { fetchJSON, API_BASE } from "../api";
import { DEFAULT_AOI_LISTS, ALL_OSCD_AOIS } from "../lulcConstants";

function Dashboard() {
  const [status, setStatus] = useState(null);
  const [aoiLists, setAoiLists] = useState(DEFAULT_AOI_LISTS);
  const [selectedAoi, setSelectedAoi] = useState("bordeaux");
  const [transition, setTransition] = useState(null);
  const [layers, setLayers] = useState({ t1: true, t2: true, transition: true });
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    fetchJSON("/status")
      .then((data) => {
        setStatus(data);
        setApiOnline(true);
      })
      .catch(() => {
        setApiOnline(false);
        setStatus({
          study_area: "OSCD multi-city (24 AOIs)",
          year1: "t1",
          year2: "t2",
          status: "offline",
        });
      });

    fetchJSON("/aois")
      .then((data) => {
        const train = data.train?.length ? data.train : DEFAULT_AOI_LISTS.train;
        const test = data.test?.length ? data.test : DEFAULT_AOI_LISTS.test;
        setAoiLists({ train, test });
      })
      .catch(() => setAoiLists(DEFAULT_AOI_LISTS));
  }, []);

  const handleAoiChange = (aoi) => {
    setSelectedAoi(aoi);
    setTransition(null);
  };

  const toggleLayer = (id) => {
    setLayers((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  return (
    <div className="dashboard">
      <div className="header">
        <div className="header-left">
          <span className="header-badge">OSCD · ResUNet</span>
          <h1 className="dashboard-title">LULC Transition Dashboard</h1>
          <p className="subtitle">
            Land-cover prediction on t1/t2 → from→to change (Forest→Urban, etc.)
          </p>
        </div>
        {status && (
          <div className="header-meta">
            <div className="meta-item">
              <span className="meta-label">Coverage</span>
              <span className="meta-value">{status.study_area}</span>
            </div>
            <div className="meta-divider" />
            <div className="meta-item">
              <span className="meta-label">Dates</span>
              <span className="meta-value">t1 → t2</span>
            </div>
            <div className="meta-divider" />
            <div className="meta-item">
              <span className="meta-label">API</span>
              <span
                className="meta-value status-dot"
                style={{ color: apiOnline ? "#22c55e" : "#f87171" }}
              >
                ● {apiOnline ? "online" : "offline"}
              </span>
            </div>
            <div className="meta-divider" />
            <div className="meta-item">
              <span className="meta-label">AOIs</span>
              <span className="meta-value">{ALL_OSCD_AOIS.length}</span>
            </div>
          </div>
        )}
      </div>

      {!apiOnline && (
        <div className="upload-error">
          ⚠️ Backend offline at <code>{API_BASE}</code>. Start it, then refresh:
          <br />
          <code>cd backend</code> →{" "}
          <code>python -m uvicorn main:app --host 127.0.0.1 --port 8000</code>
        </div>
      )}

      <StatsCards
        t1Stats={transition?.t1_fractions}
        t2Stats={transition?.t2_fractions || transition?.stats}
      />

      <UploadPanel
        selectedAoi={selectedAoi}
        onAoiChange={handleAoiChange}
        onTransition={setTransition}
        aois={aoiLists}
        layers={layers}
        result={transition}
      />

      <div className="section-block">
        <div className="section-header">
          <h3 className="section-title">AOI Map</h3>
          <span className="section-sub">
            {selectedAoi ? `Focused on ${selectedAoi}` : "OSCD cities"}
          </span>
        </div>
        <MapView
          aoi={selectedAoi}
          transition={transition}
          testAois={aoiLists.test?.length ? aoiLists.test : undefined}
        />
      </div>

      <div className="bottom-section">
        <Sidebar
          layers={layers}
          onToggleLayer={toggleLayer}
          modelInfo={transition?.model}
        />
        <div className="chart-container">
          <div className="section-header">
            <h3 className="section-title">Land Cover Share</h3>
            <span className="section-sub">
              {transition
                ? `${selectedAoi} · t1 vs t2 class fractions (%)`
                : "Run a transition to populate"}
            </span>
          </div>
          <StatsChart
            t1Stats={transition?.t1_fractions}
            t2Stats={transition?.t2_fractions || transition?.stats}
          />
        </div>
        <div className="change-container">
          <ChangePanel aoi={selectedAoi} transition={transition} />
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
