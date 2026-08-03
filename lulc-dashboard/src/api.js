const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export function apiUrl(path) {
  if (!path.startsWith("/")) path = `/${path}`;
  return `${API_BASE}${path}`;
}

export async function fetchJSON(path, options) {
  let res;
  try {
    res = await fetch(apiUrl(path), options);
  } catch (err) {
    throw new Error(
      `Failed to reach API at ${API_BASE}. Start the backend first:\n` +
        `cd backend\n` +
        `python -m pip install -r requirements.txt\n` +
        `python -m uvicorn main:app --host 127.0.0.1 --port 8000`
    );
  }
  if (!res.ok) {
    const text = await res.text();
    let detail = text || `HTTP ${res.status}`;
    try {
      const parsed = JSON.parse(text);
      if (parsed.detail) detail = parsed.detail;
    } catch {
      /* keep text */
    }
    throw new Error(detail);
  }
  return res.json();
}

export { API_BASE };
