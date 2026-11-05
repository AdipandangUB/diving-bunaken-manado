"""Monitoring Diving 3D + Misi Selam Immersive · Taman Nasional Bunaken (Manado)

Berkas yang harus berada di folder yang sama:
  app.py · bunaken_data.py · bunaken_geo.py · immersive.html · Batimetri_TN_Bunaken_Kontur.geojson
Cuaca & laut = Open-Meteo (live, tanpa API key). Data penyelam = simulasi demo atau upload CSV.
"""
import base64
import datetime as dt
import io
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components

import bunaken_geo as geo
from bunaken_data import DIVE_SPOTS, SPECIES, TURTLES

st.set_page_config(page_title="Monitoring Diving 3D · TN Bunaken", page_icon="🤿", layout="wide")
HERE = Path(__file__).parent
GEOJSON = HERE / "Batimetri_TN_Bunaken_Kontur.geojson"

# ------------------------------------------------------------------ KONFIGURASI
MAX_SAFE_DEPTH = 30  # batas kedalaman rekreasi (m)
LOW_AIR = 50  # bar, ambang udara menipis
MAX_DIVE_MIN = 60  # menit
BBOX = geo.BBOX

# Titik selam & penyu: georeferensi dari peta perjumpaan penyu (lihat bunaken_data.py)
SITE_DF = pd.DataFrame(DIVE_SPOTS, columns=["nama", "lon", "lat", "pulau", "kuota"])
TURTLE_DF = pd.DataFrame(TURTLES, columns=["jenis", "lon", "lat", "pulau"])
TURTLE_DF["nama"] = TURTLE_DF.jenis.map(lambda k: SPECIES[k]["nama"])
TURTLE_DF["color"] = TURTLE_DF.jenis.map(lambda k: [*SPECIES[k]["rgb"], 235])
TURTLE_DF["tip"] = ("<b>" + TURTLE_DF.nama + "</b><br><i>" + TURTLE_DF.jenis.map(lambda k: SPECIES[k]["latin"])
                    + "</i><br>Lokasi perjumpaan · " + TURTLE_DF.pulau)

SCENARIOS = {
    "1 · Normal": 1, "2 · Ramai": 2, "3 · Site penuh": 3,
    "4 · Ombak & hujan lebat": 4, "5 · SOS": 5, "6 · Krisis": 6,
}


# ------------------------------------------------------------------ CUACA & LAUT
@st.cache_data(ttl=900, show_spinner=False)
def fetch_weather(lats: tuple, lons: tuple):
    """Open-Meteo marine + forecast, satu panggilan multi-koordinat."""
    p = {"latitude": ",".join(map(str, lats)), "longitude": ",".join(map(str, lons))}
    try:
        m = requests.get("https://marine-api.open-meteo.com/v1/marine", timeout=10, params={
            **p, "current": "wave_height,sea_surface_temperature,ocean_current_velocity"}).json()
        f = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
            **p, "current": "precipitation,wind_speed_10m"}).json()
        m = m if isinstance(m, list) else [m]
        f = f if isinstance(f, list) else [f]
        out = []
        for a, b in zip(m, f):
            ca, cb = a["current"], b["current"]
            out.append(dict(wave=ca.get("wave_height") or 0, sst=ca.get("sea_surface_temperature"),
                            arus=ca.get("ocean_current_velocity") or 0,
                            rain=cb.get("precipitation") or 0, wind=cb.get("wind_speed_10m") or 0))
        return out if len(out) == len(lats) else None
    except Exception:
        return None


def get_weather(scn, live, rng):
    wx = fetch_weather(tuple(SITE_DF.lat.round(3)), tuple(SITE_DF.lon.round(3))) if live else None
    src = "Open-Meteo (live)"
    if wx is None:
        src = "simulasi"
        wx = [dict(wave=float(rng.uniform(0.3, 1.0)), sst=29.0, arus=float(rng.uniform(0.3, 2.0)),
                   rain=0.0, wind=float(rng.uniform(5, 15))) for _ in range(len(SITE_DF))]
    if scn in (4, 6):  # override skenario cuaca buruk
        wx = [dict(w, wave=2.6, rain=22.0, arus=4.5, wind=38.0) for w in wx]
        src += " + skenario"
    return wx, src


def rain_icon(mm):
    return "☀️" if mm == 0 else "🌦️" if mm <= 5 else "🌧️" if mm <= 10 else "⛈️" if mm <= 20 else "🌪️"


def halted(w):
    return w["wave"] >= 2.0 or w["rain"] > 20 or w["arus"] > 4 or w["wind"] > 35


# ------------------------------------------------------------------ BATIMETRI
DEPTH_STOPS = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]
DEPTH_COLORS = [(165, 232, 232), (125, 218, 226), (75, 192, 216), (32, 152, 202), (22, 122, 186), (20, 92, 166),
                (26, 62, 142), (26, 42, 112), (20, 27, 82), (12, 16, 56), (5, 8, 35)]
LAND_STOPS = [0, 50, 300, 800, 1500, 2500]
LAND_COLORS = [(196, 204, 146), (124, 172, 92), (72, 132, 72), (112, 112, 72), (152, 132, 102), (205, 195, 185)]
LEGEND_STOPS = [0, 10, 20, 30, 50, 100, 200, 500, 1000]
CONTOUR_RGB = {10: (124, 249, 255), 20: (87, 216, 255), 50: (82, 162, 255), 100: (139, 123, 255),
               200: (210, 107, 255), 500: (255, 107, 200), 1000: (255, 90, 90)}


@st.cache_data(show_spinner="Memuat kontur batimetri…")
def load_contours_cached(raw: bytes | None):
    """Kontur dari upload (raw) atau berkas GeoJSON di folder aplikasi. None bila tidak ada."""
    try:
        if raw:
            return geo.load_contours(json.loads(raw.decode("utf-8")))
        if GEOJSON.exists():
            return geo.load_contours(GEOJSON)
    except Exception:
        pass
    return None


@st.cache_data(ttl=86400, show_spinner="Memuat data batimetri…")
def terrain_grid(raw: bytes | None, size=512):
    """Grid elevasi 512x512 (m, negatif = laut) di BBOX. Utama: AWS Terrarium (darat+laut);
    cadangan (offline): interpolasi kontur GeoJSON. Mengembalikan (grid, sumber)."""
    H = geo.fetch_terrarium(BBOX, 11, size)
    if H is not None:
        return H, "AWS Terrarium (GEBCO + SRTM)"
    cont = load_contours_cached(raw)
    if cont:
        return geo.resize_grid(geo.grid_from_contours(cont, BBOX, 160), size), "interpolasi kontur GeoJSON (offline)"
    return None, "tidak tersedia"


def _uri(rgb):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def elevation_uri(h, clip):
    """Encode Terrarium (R*256+G-32768); kedalaman dipotong di -clip agar zona selam tetap terbaca."""
    v = np.round(np.maximum(h, -clip)) + 32768
    return _uri(np.stack([v // 256, v % 256, np.zeros_like(v)], -1))


def texture_uri(h, night):
    """Warna kedalaman + hillshade (garis kontur digambar sebagai PathLayer dari GeoJSON)."""
    d, up = np.clip(-h, 0, None), np.clip(h, 0, None)
    rgb = np.stack([np.where(h < 0, np.interp(d, DEPTH_STOPS, [c[i] for c in DEPTH_COLORS]),
                             np.interp(up, LAND_STOPS, [c[i] for c in LAND_COLORS])) for i in range(3)], -1)
    cell = (BBOX[2] - BBOX[0]) * 111320 * math.cos(math.radians((BBOX[1] + BBOX[3]) / 2)) / h.shape[1]
    gy, gx = np.gradient(h, cell)
    gx, gy = gx * 3, gy * 3  # eksagerasi relief untuk bayangan
    shade = (0.5 * gx + 0.5 * gy + 0.707) / np.sqrt(gx ** 2 + gy ** 2 + 1) / 0.707
    rgb = rgb * (0.6 + 0.4 * np.clip(shade, 0, 1.4))[..., None]
    return _uri(rgb * (0.6 if night else 1))


@st.cache_data(show_spinner=False)
def bathy_images(raw, night, clip):
    h, _ = terrain_grid(raw)
    return None if h is None else (elevation_uri(h, clip), texture_uri(h, night))


def legend_html():
    cols = [DEPTH_COLORS[DEPTH_STOPS.index(t)] for t in LEGEND_STOPS]
    grad = ",".join(f"rgb{c}" for c in cols)
    labs = "".join(f"<span>{t}</span>" for t in LEGEND_STOPS[:-1]) + "<span>1000+</span>"
    return (f"<div style='height:14px;border-radius:4px;background:linear-gradient(90deg,{grad})'></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:11px'>{labs}</div>"
            "<div style='font-size:11px;opacity:.7'>Kedalaman (m) · garis kontur GeoJSON 10/20/50/100/200/500/1000 m</div>")


# ------------------------------------------------------------------ DATA PENYELAM
def simulate(scn, rng):
    mult = {1: 1, 2: 2, 3: 1.3, 4: 0.6, 5: 1, 6: 2}[scn]
    rows = []
    for i, r in enumerate(SITE_DF.itertuples()):
        n = int(rng.poisson(r.kuota * 0.45 * mult * 0.5))  # 22 site: skala dikecilkan agar total wajar
        if scn == 3 and i == 0:
            n = r.kuota + 4
        for j in range(n):
            depth = float(np.clip(rng.gamma(4, 5), 3, 42 if scn == 6 else 36))
            dive = int(rng.integers(5, 68 if scn == 6 else 55))
            air = float(np.clip(200 - dive * (2 + depth / 10) + rng.normal(0, 12), 10, 200))
            arah = "turun" if dive < 10 else "naik" if (air < 80 or dive > 45) else rng.choice(["turun", "naik"])
            rows.append(dict(nama=f"Diver {i+1}{j+1:02d}", site=r.nama, depth_m=round(depth, 1),
                             air_bar=int(air), dive_min=dive, arah=arah, sos=0))
    df = pd.DataFrame(rows, columns=["nama", "site", "depth_m", "air_bar", "dive_min", "arah", "sos"])
    n_sos = {5: 1, 6: 3}.get(scn, 0)
    if n_sos and len(df):
        df.loc[rng.choice(len(df), min(n_sos, len(df)), replace=False), "sos"] = 1
    return df


def from_csv(file):
    """CSV kolom: nama, site, depth_m, air_bar, dive_min, arah(turun/naik), sos(0/1)"""
    df = pd.read_csv(file)
    for c, d in [("arah", "turun"), ("sos", 0), ("air_bar", 150), ("dive_min", 20)]:
        if c not in df:
            df[c] = d
    ok = df.site.isin(SITE_DF.nama)
    if (~ok).any():
        st.sidebar.warning(f"{int((~ok).sum())} baris CSV dilewati: nama site tidak dikenal "
                           f"({', '.join(sorted(set(df.site[~ok].astype(str)))[:4])}…)")
    return df[ok].reset_index(drop=True)


def place(df, rng, zs):
    """zs = skala vertikal (eksagerasi; 0 pada mode lite)."""
    if df.empty:
        return df.assign(lon=[], lat=[], z=[])
    s = SITE_DF.set_index("nama")
    df = df.copy()
    df["lon"] = df.site.map(s.lon) + rng.normal(0, 0.0025, len(df))
    df["lat"] = df.site.map(s.lat) + rng.normal(0, 0.0025, len(df))
    df["z"] = -df.depth_m * zs
    df["color"] = df.arah.map(lambda a: [52, 152, 219, 230] if a == "turun" else [255, 140, 0, 230])
    df.loc[df.sos == 1, "color"] = pd.Series([[255, 0, 0, 255]] * int((df.sos == 1).sum()), index=df.index[df.sos == 1])
    df["tip"] = ("<b>" + df.nama + "</b> · " + df.site + "<br>Kedalaman " + df.depth_m.astype(str) + " m<br>Udara "
                 + df.air_bar.astype(str) + " bar · " + df.dive_min.astype(str) + " menit<br>Arah: " + df.arah
                 + np.where(df.sos == 1, "<br>🆘 SOS", ""))
    return df


def build_sites(divers, wx):
    s = SITE_DF.copy()
    s["n"] = s.nama.map(divers.site.value_counts()).fillna(0).astype(int)
    s["rasio"] = s.n / s.kuota
    s["wave"], s["arus"] = [w["wave"] for w in wx], [w["arus"] for w in wx]
    s["rain"], s["wind"], s["sst"] = [w["rain"] for w in wx], [w["wind"] for w in wx], [w["sst"] for w in wx]
    s["ditahan"] = [halted(w) for w in wx]
    s["color"] = s.rasio.map(lambda r: [46, 204, 113, 200] if r < .5 else [241, 196, 15, 200] if r < .8 else [231, 76, 60, 200])
    s["tip"] = ("<b>" + s.nama + "</b> · " + s.pulau + "<br>" + s.n.astype(str) + "/" + s.kuota.astype(str)
                + " penyelam<br>Gelombang " + s.wave.round(1).astype(str) + " m · Arus " + s.arus.round(1).astype(str) + " km/j"
                + np.where(s.ditahan, "<br>🔴 DIVING DITAHAN", ""))
    return s


def build_alerts(divers, sites):
    a = []
    for _, r in divers[divers.sos == 1].iterrows():
        a.append(("🆘", f"SOS {r.nama} di {r.site} ({r.depth_m} m)"))
    for _, r in divers[(divers.air_bar < LOW_AIR) & (divers.sos == 0)].iterrows():
        a.append(("🟠", f"Udara menipis {r.nama} ({r.air_bar} bar) di {r.site}"))
    for _, r in divers[divers.depth_m > MAX_SAFE_DEPTH].iterrows():
        a.append(("🟡", f"{r.nama} melewati {MAX_SAFE_DEPTH} m ({r.depth_m} m) di {r.site}"))
    for _, r in divers[divers.dive_min > MAX_DIVE_MIN].iterrows():
        a.append(("🟡", f"{r.nama} menyelam {r.dive_min} menit di {r.site}"))
    for _, r in sites.iterrows():
        if r.n > r.kuota:
            a.append(("🔴", f"{r.nama} melebihi kuota ({r.n}/{r.kuota})"))
        if r.ditahan and r.n > 0:
            a.append(("🔴", f"{r.nama}: kondisi laut buruk, {r.n} penyelam masih di air"))
    return a


# ------------------------------------------------------------------ PETA 3D (pydeck)
def build_deck(divers, sites, contours, H, opt):
    zs = 0 if opt["lite"] else opt["exag"]
    dec = ({"rScaler": 0, "gScaler": 0, "bScaler": 0, "offset": 0} if opt["lite"] else
           {"rScaler": 256 * zs, "gScaler": zs, "bScaler": zs / 256, "offset": -32768 * zs})
    img = bathy_images(opt["raw"], opt["night"], opt["clip"]) if opt["mode"] == "Batimetri berwarna" else None
    if img:  # single-image TerrainLayer: relief & warna batimetri dari grid sendiri
        L = [pdk.Layer("TerrainLayer", elevation_decoder=dec, elevation_data=pdk.types.String(img[0]), texture=pdk.types.String(img[1]),
                       bounds=list(BBOX), mesh_max_error=3)]
    else:  # mode satelit / grid gagal dimuat
        tex = ("https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png" if opt["night"] else
               "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")
        elev = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        L = [pdk.Layer("TerrainLayer", elevation_decoder=dec, texture=pdk.types.String(tex), elevation_data=pdk.types.String(elev), max_zoom=15)]
    if opt["water"]:
        w, s_, e, n = BBOX
        L.append(pdk.Layer("PolygonLayer", [{"p": [[w, s_, 0], [e, s_, 0], [e, n, 0], [w, n, 0]]}], get_polygon="p",
                           get_fill_color=[0, 140, 255, 35], stroked=False))
    if opt["contour"] and contours:  # kontur batimetri GeoJSON di kedalaman sebenarnya (3D)
        rows = geo.contour_paths(contours, zs, opt["clip"])
        for r in rows:
            r["color"] = [*CONTOUR_RGB.get(r["depth"], (200, 255, 255)), 230]
        L.append(pdk.Layer("PathLayer", rows, get_path="path", get_color="color", get_width=1.5,
                           width_min_pixels=2, pickable=True, auto_highlight=True))
        if opt["label"]:
            lab = []  # satu label per level (garis terpanjang) agar peta tidak penuh teks
            for d, lines in contours.items():
                ln = max(lines, key=len)
                x, y = ln[len(ln) // 2]
                lab.append(dict(lon=x, lat=y, z=-min(d, opt["clip"]) * zs, t=f"-{d} m"))
            L.append(pdk.Layer("TextLayer", lab, get_position="[lon, lat, z]", get_text="t", get_size=11,
                               get_color=[200, 255, 255, 255], background=True, get_background_color=[0, 30, 50, 150]))
    L.append(pdk.Layer("ColumnLayer", sites, get_position="[lon, lat]", get_elevation="n", elevation_scale=40,
                       radius=130, get_fill_color="color", extruded=True, pickable=True))
    ring = sites[sites.ditahan]
    if len(ring):
        L.append(pdk.Layer("ScatterplotLayer", ring, get_position="[lon, lat, 5]", get_radius=260, stroked=True,
                           filled=False, get_line_color=[255, 0, 0], line_width_min_pixels=3))
    if opt["turtles"]:  # titik perjumpaan penyu (hijau / sisik / lekang) melayang di kedalaman perkiraan
        t = TURTLE_DF.copy()
        if H is not None:
            d = np.array([max(0.0, -geo.sample(H, lo, la)) for lo, la in zip(t.lon, t.lat)])
            t["z"] = -np.clip(d - 2, 2, 12) * zs
        else:
            t["z"] = -6 * zs
        L.append(pdk.Layer("ScatterplotLayer", t, get_position="[lon, lat, z]", get_radius=55, get_fill_color="color",
                           stroked=True, get_line_color=[0, 0, 0, 200], line_width_min_pixels=1, pickable=True))
        L.append(pdk.Layer("LineLayer", t, get_source_position="[lon, lat, 0]", get_target_position="[lon, lat, z]",
                           get_color=[255, 255, 255, 60], get_width=1))
    if opt["divers"] and len(divers):
        L.append(pdk.Layer("LineLayer", divers, get_source_position="[lon, lat, 0]", get_target_position="[lon, lat, z]",
                           get_color=[200, 200, 200, 70], get_width=1))
        L.append(pdk.Layer("ScatterplotLayer", divers, get_position="[lon, lat, z]", get_radius=22,
                           get_fill_color="color", pickable=True))
        sos = divers[divers.sos == 1]
        if len(sos):
            L.append(pdk.Layer("ScatterplotLayer", sos, get_position="[lon, lat, z]", get_radius=70, stroked=True,
                               filled=False, get_line_color=[255, 0, 0], line_width_min_pixels=3))
    if opt["label"]:
        L.append(pdk.Layer("TextLayer", sites, get_position="[lon, lat, 200]", get_text="nama", get_size=11,
                           get_color=[255, 255, 255, 255], background=True, get_background_color=[0, 0, 0, 150]))
    view = pdk.ViewState(latitude=1.62, longitude=124.765, zoom=opt["zoom"], pitch=opt["pitch"],
                         bearing=opt["bearing"], max_pitch=85)
    return pdk.Deck(layers=L, initial_view_state=view, map_style=None, tooltip={"html": "{tip}"})


# ------------------------------------------------------------------ SCENE IMMERSIVE
@st.cache_data(show_spinner="Menyiapkan scene 3D…")
def game_html(raw, start, briefing_json):
    H, src = terrain_grid(raw)
    cont = load_contours_cached(raw)
    if H is None or not cont:
        return None
    briefing = json.loads(briefing_json)
    payload = geo.build_payload(cont, H, DIVE_SPOTS, TURTLES, SPECIES, BBOX,
                                extra=dict(start=start, briefing=briefing, maxSafe=MAX_SAFE_DEPTH, lowAir=LOW_AIR,
                                           arus=briefing.get(start, {}).get("arus", 0), terrainSrc=src))
    return geo.inject((HERE / "immersive.html").read_text(encoding="utf-8"), payload)


def embed_html(html: str, height: int):
    """st.iframe (Streamlit terbaru); st.components.v1.html sudah deprecated & akan dihapus."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=False)


# ------------------------------------------------------------------ UI
DEF = dict(zoom=10.6, pitch=55, bearing=20)
for k, v in DEF.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.header("⚙️ Kontrol")
    scn_name = st.selectbox("Skenario demo", list(SCENARIOS), index=0)
    live = st.toggle("Cuaca & laut live (Open-Meteo)", True)
    st.subheader("Peta 3D (monitoring)")
    basemap = st.radio("Peta dasar", ["Batimetri berwarna", "Citra satelit"])
    EXAG = st.slider("Eksagerasi vertikal", 1, 8, 3)
    CLIP = st.select_slider("Potong kedalaman visual (m)", [100, 200, 300, 600, 1500], 300,
                            help="Hanya memengaruhi relief 3D; warna & profil tetap kedalaman asli.")
    show_cont = st.toggle("Kontur batimetri (GeoJSON)", True)
    show_turtle = st.toggle("Sebaran penyu", True)
    water = st.toggle("Permukaan air", False)
    show_div = st.toggle("Penyelam", True)
    label = st.toggle("Label", True)
    night = st.toggle("Mode malam", False)
    lite = st.toggle("Mode lite (tanpa relief 3D)", False)
    st.slider("Zoom", 8.0, 14.0, key="zoom", step=0.1)
    st.slider("Kemiringan", 0, 85, key="pitch")
    st.slider("Putar", -180, 180, key="bearing")
    if st.button("Reset kamera"):
        for k, v in DEF.items():
            st.session_state[k] = v
        st.rerun()
    up = st.file_uploader("Data penyelam (CSV)", type="csv",
                          help="Kolom: nama, site, depth_m, air_bar, dive_min, arah (turun/naik), sos (0/1)")
    up_gj = st.file_uploader("Kontur batimetri (GeoJSON, opsional)", type=["geojson", "json"],
                             help="Bila kosong dipakai Batimetri_TN_Bunaken_Kontur.geojson di folder aplikasi. "
                                  "Properti 'kedalaman' berisi angka negatif (m).")
    st.button("🔄 Muat ulang data")

raw_gj = up_gj.getvalue() if up_gj else None
contours = load_contours_cached(raw_gj)
H, terrain_src = terrain_grid(raw_gj)

scn = SCENARIOS[scn_name]
wita = dt.datetime.now(ZoneInfo("Asia/Makassar"))
rng = np.random.default_rng(scn * 1000 + wita.hour * 60 + wita.minute // 5)

wx, wx_src = get_weather(scn, live, rng)
divers = place(from_csv(up) if up else simulate(scn, rng), rng, 0 if lite else EXAG)
sites = build_sites(divers, wx)
alerts = build_alerts(divers, sites)

st.title("🤿 Taman Nasional Bunaken · Monitoring & Misi Selam 3D")
st.caption(f"{wita:%d %b %Y %H:%M} WITA · Data penyelam: {'CSV upload' if up else 'simulasi demo'} · Cuaca: {wx_src} · "
           f"Relief: {terrain_src}")
if contours is None:
    st.warning("Berkas Batimetri_TN_Bunaken_Kontur.geojson tidak ditemukan — letakkan di folder yang sama dengan app.py "
               "atau upload lewat sidebar. Kontur 3D & scene immersive dinonaktifkan.")

tab_game, tab_mon = st.tabs(["🎮 Misi Selam 3D (Immersive)", "📊 Monitoring 3D"])

# ---------------------------------------------------------------- TAB GAME
with tab_game:
    c1, c2 = st.columns([1, 2])
    start = c1.selectbox("Dive spot awal", SITE_DF.nama, index=int(SITE_DF.nama.tolist().index("Likuan II")))
    # briefing dibekukan per sesi supaya scene tidak dimuat ulang saat cuaca ter-refresh di tengah permainan
    key = f"brief_{scn}_{live}"
    if key not in st.session_state or c2.button("🔄 Perbarui briefing cuaca"):
        st.session_state[key] = json.dumps({
            r.nama: dict(wave=round(float(w["wave"]), 2), arus=round(float(w["arus"]), 2),
                         sst=None if w["sst"] is None else round(float(w["sst"]), 1), ditahan=bool(halted(w)))
            for r, w in zip(SITE_DF.itertuples(), wx)})
    html = game_html(raw_gj, start, st.session_state[key])
    if html is None:
        st.error("Scene tidak dapat dibuat (data batimetri/kontur tidak tersedia).")
    else:
        b = json.loads(st.session_state[key])[start]
        c2.caption(f"Briefing {start}: gelombang {b['wave']:.1f} m · arus {b['arus']:.1f} km/j · "
                   f"{'🔴 diving ditahan' if b['ditahan'] else '🟢 layak selam'}")
        embed_html(html, 740)
        st.caption("Diver dipandu menyusuri lereng terumbu → cincin emas di titik perjumpaan penyu → safety stop 5 m → "
                   "permukaan. Tiga mode: 🌊 Jelajah Santai (first-person, bebas berenang & mengisi jurnal spesies), otomatis (🧭), atau manual terpandu. Kontrol: ↑ maju, A/D geser/belok, W turun, Spasi naik, seret mouse = arah pandang (klik area game dulu agar keyboard aktif). Relief mengikuti "
                   "batimetri GEBCO; garis berwarna = kontur GeoJSON di kedalaman sebenarnya. Simulasi edukasi.")

# ---------------------------------------------------------------- TAB MONITORING
with tab_mon:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Penyelam di air", len(divers))
    c2.metric("Site aktif", int((sites.n > 0).sum()))
    c3.metric("Site ditahan", int(sites.ditahan.sum()))
    c4.metric("Alert", len(alerts))

    left, right = st.columns([2.2, 1])
    with left:
        deck = build_deck(divers, sites, contours, H, dict(
            mode=basemap, clip=CLIP, exag=EXAG, water=water, night=night, lite=lite, divers=show_div, label=label,
            contour=show_cont, turtles=show_turtle, raw=raw_gj, **{k: st.session_state[k] for k in DEF}))
        st.pydeck_chart(deck, height=640)
        st.caption("🔵 menyelam turun · 🟠 naik · 🔴 cincin merah = diving ditahan / SOS · Kolom = jumlah penyelam "
                   "(hijau lega → merah padat) · Titik: 🟢 penyu hijau · 🔴 penyu sisik · 🟡 penyu lekang. "
                   "Kiri-drag: putar · Kanan-drag: geser · Scroll: zoom.")
        st.caption("☀️ 0 · 🌦️ 0,1–5 · 🌧️ 5,1–10 · ⛈️ 10,1–20 · 🌪️ >20 mm/jam. Batimetri: GEBCO via AWS Terrain Tiles "
                   "(resolusi kasar ±450 m, bukan survei lokal) · Citra: Esri World Imagery · Posisi titik selam & penyu "
                   "dari peta perjumpaan penyu (akurasi ±15–40 m).")

    with right:
        st.subheader("Batimetri")
        st.markdown(legend_html(), unsafe_allow_html=True)
        if H is None:
            st.warning("Data batimetri gagal dimuat; peta memakai tile AWS langsung.")
        else:
            sel = st.selectbox("Profil kedalaman di site", SITE_DF.nama)
            r_ = SITE_DF.set_index("nama").loc[sel]
            st.caption(f"Kedalaman model di titik site: {max(0, -geo.sample(H, r_.lon, r_.lat)):.0f} m (±), "
                       "tegak lurus garis pantai tidak dijamin.")
            span = int(0.04 / (BBOX[2] - BBOX[0]) * H.shape[1])
            cc = int(round((r_.lon - BBOX[0]) / (BBOX[2] - BBOX[0]) * (H.shape[1] - 1)))
            rr = int(round((BBOX[3] - r_.lat) / (BBOX[3] - BBOX[1]) * (H.shape[0] - 1)))
            lo, hi = max(0, cc - span), min(H.shape[1], cc + span)
            lon = BBOX[0] + np.arange(lo, hi) / (H.shape[1] - 1) * (BBOX[2] - BBOX[0])
            st.area_chart(pd.DataFrame({"Elevasi (m)": H[rr, lo:hi]}, index=pd.Index(lon.round(4), name="Bujur")), height=170)

        st.subheader("Sebaran penyu")
        cnt = TURTLE_DF.groupby(["pulau", "nama"]).size().unstack(fill_value=0)
        st.dataframe(cnt, width="stretch")

        st.subheader("Status site & cuaca")
        t = pd.DataFrame({
            "Site": sites.nama, "Diver": sites.n.astype(str) + "/" + sites.kuota.astype(str),
            "Ombak (m)": sites.wave.round(1), "Arus (km/j)": sites.arus.round(1),
            "Hujan": [rain_icon(r) for r in sites.rain],
            "Status": np.where(sites.ditahan, "🔴 Ditahan", np.where(sites.rasio >= .8, "🟠 Padat", "🟢 Aman")),
        })
        st.dataframe(t, hide_index=True, width="stretch")

        st.subheader("Site terpadat")
        st.bar_chart(sites.set_index("nama").rasio.mul(100).sort_values(ascending=False).head(5), horizontal=True)

        st.subheader("Sebaran kedalaman")
        if len(divers):
            bins = pd.cut(divers.depth_m, [0, 10, 20, 30, 100], labels=["0–10 m", "10–20 m", "20–30 m", ">30 m"])
            st.bar_chart(bins.value_counts().sort_index())

        st.subheader("Perlu perhatian")
        if not alerts:
            st.success("Tidak ada alert.")
        for icon, msg in alerts[:12]:
            st.write(f"{icon} {msg}")
        if len(alerts) > 12:
            st.caption(f"+{len(alerts) - 12} alert lainnya")
