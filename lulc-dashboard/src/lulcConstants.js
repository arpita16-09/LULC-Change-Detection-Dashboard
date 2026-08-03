/** Matches backend / OSCD-enhanced ResUNet class IDs and viz colors. */
export const LULC_CLASSES = [
  { id: 0, key: "other", label: "Other", color: "#3c3c3c" },
  { id: 1, key: "water", label: "Water", color: "#1e90ff" },
  { id: 2, key: "urban", label: "Urban", color: "#dc143c" },
  { id: 3, key: "forest", label: "Forest", color: "#228b22" },
  { id: 4, key: "agriculture", label: "Agriculture", color: "#adff2f" },
  { id: 5, key: "barren", label: "Barren", color: "#d2b48c" },
];

export const LULC_BY_KEY = Object.fromEntries(LULC_CLASSES.map((c) => [c.key, c]));

/** Approximate AOI centers from OSCD geojson (lat, lon). */
export const AOI_COORDS = {
  abudhabi: [24.3214, 54.5698],
  aguasclaras: [-15.8423, -48.0305],
  beihai: [21.5654, 109.5127],
  beirut: [33.8481, 35.5042],
  bercy: [48.839, 2.3761],
  bordeaux: [44.8331, -0.5809],
  brasilia: [-15.7506, -47.8905],
  chongqing: [29.4074, 106.2907],
  cupertino: [37.3391, -122.0216],
  dubai: [25.0297, 55.2178],
  hongkong: [22.3024, 114.2483],
  lasvegas: [36.0158, -115.2436],
  milano: [45.497, 9.1696],
  montpellier: [43.5962, 3.8985],
  mumbai: [19.0575, 72.9223],
  nantes: [47.199, -1.5513],
  norcia: [42.7912, 13.0888],
  paris: [48.8166, 2.3283],
  pisa: [43.7082, 10.3939],
  rennes: [48.1093, -1.6595],
  rio: [-22.9714, -43.3873],
  saclay_e: [48.6962, 2.2491],
  saclay_w: [48.6962, 2.17],
  valencia: [39.5005, -0.4265],
};

export const DEFAULT_CENTER = [30, 10];
export const DEFAULT_ZOOM = 2;

/** Official OSCD splits (all 24 cities). */
export const OSCD_TEST_AOIS = [
  "brasilia",
  "montpellier",
  "norcia",
  "rio",
  "saclay_w",
  "valencia",
  "dubai",
  "lasvegas",
  "milano",
  "chongqing",
];

export const OSCD_TRAIN_AOIS = [
  "aguasclaras",
  "bercy",
  "bordeaux",
  "nantes",
  "paris",
  "rennes",
  "saclay_e",
  "abudhabi",
  "cupertino",
  "pisa",
  "beihai",
  "hongkong",
  "beirut",
  "mumbai",
];

export const ALL_OSCD_AOIS = [...OSCD_TRAIN_AOIS, ...OSCD_TEST_AOIS].sort();

export const DEFAULT_AOI_LISTS = {
  train: [...OSCD_TRAIN_AOIS].sort(),
  test: [...OSCD_TEST_AOIS].sort(),
};

export const TRANSITION_LAYER_META = {
  t1: { label: "t1 LULC map", color: "#38bdf8" },
  t2: { label: "t2 LULC map", color: "#4ade80" },
  transition: { label: "Transition map", color: "#f59e0b" },
};
