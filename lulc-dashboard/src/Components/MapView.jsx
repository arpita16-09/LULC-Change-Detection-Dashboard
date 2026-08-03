import { useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap, CircleMarker } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { AOI_COORDS, ALL_OSCD_AOIS, DEFAULT_CENTER, DEFAULT_ZOOM } from "../lulcConstants";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

function FlyToAoi({ center, zoom }) {
  const map = useMap();
  useEffect(() => {
    if (!center) return;
    map.flyTo(center, zoom ?? 11, { duration: 0.8 });
  }, [center, zoom, map]);
  return null;
}

function MapView({ aoi, transition, testAois = [] }) {
  const center = (aoi && AOI_COORDS[aoi]) || DEFAULT_CENTER;
  const zoom = aoi && AOI_COORDS[aoi] ? 11 : DEFAULT_ZOOM;
  // Show every OSCD city; highlight selected. testAois kept for API compat.
  const markers = ALL_OSCD_AOIS.length ? ALL_OSCD_AOIS : Object.keys(AOI_COORDS);

  return (
    <MapContainer
      center={center}
      zoom={zoom}
      style={{ height: "420px", width: "100%", borderRadius: "12px" }}
      scrollWheelZoom
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="© OpenStreetMap contributors"
      />
      <FlyToAoi center={center} zoom={zoom} />

      {markers.map((name) => {
        const pos = AOI_COORDS[name];
        if (!pos) return null;
        const selected = name === aoi;
        return (
          <CircleMarker
            key={name}
            center={pos}
            radius={selected ? 10 : 6}
            pathOptions={{
              color: selected ? "#14b8a6" : "#64748b",
              fillColor: selected ? "#2dd4bf" : "#94a3b8",
              fillOpacity: selected ? 0.9 : 0.45,
              weight: selected ? 2 : 1,
            }}
          >
            <Popup>
              <strong>{name}</strong>
              <br />
              OSCD AOI · LULC transition
              {selected && transition ? (
                <>
                  <br />
                  Change: {transition.change_pct}%
                  <br />
                  Top:{" "}
                  {transition.transitions?.[0]
                    ? `${transition.transitions[0].from} → ${transition.transitions[0].to}`
                    : "n/a"}
                </>
              ) : null}
            </Popup>
          </CircleMarker>
        );
      })}

      {aoi && AOI_COORDS[aoi] && (
        <Marker position={AOI_COORDS[aoi]}>
          <Popup>
            <strong>{aoi}</strong>
            <br />
            Selected study area
            {transition ? (
              <>
                <br />
                Model: OSCD-enhanced ResUNet
                <br />
                Changed pixels: {transition.change_pct}%
              </>
            ) : (
              <>
                <br />
                Run LULC Transition to see from→to stats
              </>
            )}
          </Popup>
        </Marker>
      )}
    </MapContainer>
  );
}

export default MapView;
