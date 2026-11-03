"""Dashboard 3D Monitoring Diving · Taman Nasional Bunaken (Manado)
Streamlit Cloud + pydeck. Data penyelam = simulasi demo (atau upload CSV);
cuaca & laut = Open-Meteo (live, gratis tanpa API key).
"""
import base64
import datetime as dt
import io
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

st.set_page_config(page_title="Monitoring Diving 3D · TN Bunaken", page_icon="🤿", layout="wide")

# ------------------------------------------------------------------ KONFIGURASI
EXAG = 4  # eksagerasi vertikal (kedalaman & bathimetri)
MAX_SAFE_DEPTH = 30  # batas kedalaman rekreasi (m)
LOW_AIR = 50  # bar, ambang udara menipis
MAX_DIVE_MIN = 60  # menit

# nama, lat, lon, kuota penyelam bersamaan  (KOORDINAT PERKIRAAN - verifikasi dgn BTNB/GPS)
SITES = [
    ("Lekuan 1-3", 1.6140, 124.7560, 20),
    ("Fukui", 1.6080, 124.7610, 15),
    ("Alung Banua", 1.6290, 124.7520, 15),
    ("Muka Kampung", 1.6370, 124.7580, 15),
    ("Mandolin", 1.6425, 124.7630, 20),
    ("Sachiko's Point", 1.6460, 124.7700, 12),
    ("Siladen", 1.6280, 124.8000, 18),
    ("Tanjung Pisok", 1.6540, 124.7150, 12),
    ("Montehage", 1.7000, 124.7500, 12),
]
SITE_DF = pd.DataFrame(SITES, columns=["nama", "lat", "lon", "kuota"])

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
        return out
    except Exception:
        return None


def get_weather(scn, live, rng):
    wx = fetch_weather(tuple(SITE_DF.lat), tuple(SITE_DF.lon)) if live else None
    src = "Open-Meteo (live)"
    if wx is None:
        src = "simulasi"
        wx = [dict(wave=float(rng.uniform(0.3, 1.0)), sst=29.0, arus=float(rng.uniform(0.3, 2.0)),
                   rain=0.0, wind=float(rng.uniform(5, 15))) for _ in SITES]
    if scn in (4, 6):  # override skenario cuaca buruk
        wx = [dict(w, wave=2.6, rain=22.0, arus=4.5, wind=38.0) for w in wx]
        src += " + skenario"
    return wx, src


def rain_icon(mm):
    return "☀️" if mm == 0 else "🌦️" if mm <= 5 else "🌧️" if mm <= 10 else "⛈️" if mm <= 20 else "🌪️"


def halted(w):
    return w["wave"] >= 2.0 or w["rain"] > 20 or w["arus"] > 4 or w["wind"] > 35


# ------------------------------------------------------------------ BATIMETRI
BBOX = (124.66, 1.55, 124.86, 1.75)  # barat, selatan, timur, utara
DEPTH_STOPS = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]
DEPTH_COLORS = [(165, 232, 232), (125, 218, 226), (75, 192, 216), (32, 152, 202), (22, 122, 186), (20, 92, 166),
                (26, 62, 142), (26, 42, 112), (20, 27, 82), (12, 16, 56), (5, 8, 35)]
LAND_STOPS = [0, 50, 300, 800, 1500, 2500]
LAND_COLORS = [(196, 204, 146), (124, 172, 92), (72, 132, 72), (112, 112, 72), (152, 132, 102), (205, 195, 185)]
LEGEND_STOPS = [0, 10, 20, 30, 50, 100, 200, 500, 1000]


def _px(lon, lat, z):
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


@st.cache_data(ttl=86400, show_spinner="Memuat data batimetri…")
def load_bathymetry(bbox, z=11, size=512):
    """Mosaik AWS Terrarium (termasuk bathimetri GEBCO) -> grid elevasi (m, negatif = laut)."""
    try:
        from PIL import Image
        w, s_, e, n = bbox
        x0, y0 = _px(w, n, z)
        x1, y1 = _px(e, s_, z)
        tx0, tx1, ty0, ty1 = int(x0 // 256), int(x1 // 256), int(y0 // 256), int(y1 // 256)
        mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.float32)
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                r = requests.get(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png", timeout=15)
                r.raise_for_status()
                a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), np.float32)
                mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = (
                    a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
        crop = mosaic[int(y0 - ty0 * 256):int(math.ceil(y1 - ty0 * 256)), int(x0 - tx0 * 256):int(math.ceil(x1 - tx0 * 256))]
        return np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR), np.float32)
    except Exception:
        return None


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
    """Warna kedalaman + hillshade + garis kontur kedalaman."""
    d, up = np.clip(-h, 0, None), np.clip(h, 0, None)
    rgb = np.stack([np.where(h < 0, np.interp(d, DEPTH_STOPS, [c[i] for c in DEPTH_COLORS]),
                             np.interp(up, LAND_STOPS, [c[i] for c in LAND_COLORS])) for i in range(3)], -1)
    cell = (BBOX[2] - BBOX[0]) * 111320 * math.cos(math.radians((BBOX[1] + BBOX[3]) / 2)) / h.shape[1]
    gy, gx = np.gradient(h, cell)
    gx, gy = gx * 3, gy * 3  # eksagerasi relief untuk bayangan
    shade = (0.5 * gx + 0.5 * gy + 0.707) / np.sqrt(gx ** 2 + gy ** 2 + 1) / 0.707
    rgb = rgb * (0.6 + 0.4 * np.clip(shade, 0, 1.4))[..., None]
    for t in (10, 20, 30, 50, 100, 200, 500, 1000):  # kontur kedalaman
        m = d >= t
        edge = np.zeros_like(m)
        edge[:, :-1] |= m[:, :-1] != m[:, 1:]
        edge[:-1] |= m[:-1] != m[1:]
        rgb[edge] = rgb[edge] * 0.55 + 255 * 0.45
    return _uri(rgb * (0.6 if night else 1))


@st.cache_data(show_spinner=False)
def bathy_images(night, clip):
    h = load_bathymetry(BBOX)
    return None if h is None else (elevation_uri(h, clip), texture_uri(h, night))


def depth_at(h, lon, lat):
    r = int(round((BBOX[3] - lat) / (BBOX[3] - BBOX[1]) * (h.shape[0] - 1)))
    c = int(round((lon - BBOX[0]) / (BBOX[2] - BBOX[0]) * (h.shape[1] - 1)))
    return h, r, c


def legend_html():
    cols = [DEPTH_COLORS[DEPTH_STOPS.index(t)] for t in LEGEND_STOPS]
    grad = ",".join(f"rgb{c}" for c in cols)
    labs = "".join(f"<span>{t}</span>" for t in LEGEND_STOPS[:-1]) + "<span>1000+</span>"
    return (f"<div style='height:14px;border-radius:4px;background:linear-gradient(90deg,{grad})'></div>"
            f"<div style='display:flex;justify-content:space-between;font-size:11px'>{labs}</div>"
            "<div style='font-size:11px;opacity:.7'>Kedalaman (m) · garis putih = kontur 10/20/30/50/100/200/500/1000 m</div>")


# ------------------------------------------------------------------ DATA PENYELAM
def simulate(scn, rng):
    mult = {1: 1, 2: 2, 3: 1.3, 4: 0.6, 5: 1, 6: 2}[scn]
    rows = []
    for i, (nama, lat, lon, kuota) in enumerate(SITES):
        n = int(rng.poisson(kuota * 0.45 * mult))
        if scn == 3 and i == 0:
            n = kuota + 4
        for j in range(n):
            depth = float(np.clip(rng.gamma(4, 5), 3, 42 if scn == 6 else 36))
            dive = int(rng.integers(5, 68 if scn == 6 else 55))
            air = float(np.clip(200 - dive * (2 + depth / 10) + rng.normal(0, 12), 10, 200))
            arah = "turun" if dive < 10 else "naik" if (air < 80 or dive > 45) else rng.choice(["turun", "naik"])
            rows.append(dict(nama=f"Diver {i+1}{j+1:02d}", site=nama, depth_m=round(depth, 1),
                             air_bar=int(air), dive_min=dive, arah=arah, sos=0))
    df = pd.DataFrame(rows)
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
    return df[df.site.isin(SITE_DF.nama)].reset_index(drop=True)


def place(df, rng):
    if df.empty:
        return df.assign(lon=[], lat=[], z=[])
    s = SITE_DF.set_index("nama")
    df = df.copy()
    df["lon"] = df.site.map(s.lon) + rng.normal(0, 0.0025, len(df))
    df["lat"] = df.site.map(s.lat) + rng.normal(0, 0.0025, len(df))
    df["z"] = -df.depth_m * EXAG
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
    s["tip"] = ("<b>" + s.nama + "</b><br>" + s.n.astype(str) + "/" + s.kuota.astype(str) + " penyelam<br>Gelombang "
                + s.wave.round(1).astype(str) + " m · Arus " + s.arus.round(1).astype(str) + " km/j"
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


# ------------------------------------------------------------------ PETA 3D
def build_deck(divers, sites, opt):
    flat = opt["lite"]
    dec = ({"rScaler": 0, "gScaler": 0, "bScaler": 0, "offset": 0} if flat else
           {"rScaler": 256 * EXAG, "gScaler": EXAG, "bScaler": EXAG / 256, "offset": -32768 * EXAG})
    img = bathy_images(opt["night"], opt["clip"]) if opt["mode"] == "Batimetri berwarna" else None
    if img:  # single-image TerrainLayer: relief & warna batimetri dari grid sendiri
        L = [pdk.Layer("TerrainLayer", elevation_decoder=dec, elevation_data=img[0], texture=img[1],
                       bounds=list(BBOX), mesh_max_error=3)]
    else:  # fallback / mode satelit
        tex = ("https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png" if opt["night"] else
               "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")
        elev = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        L = [pdk.Layer("TerrainLayer", elevation_decoder=dec, texture=tex, elevation_data=elev, max_zoom=15)]
    if opt["water"]:
        w, s_, e, n = BBOX
        L.append(pdk.Layer("PolygonLayer", [{"p": [[w, s_, 0], [e, s_, 0], [e, n, 0], [w, n, 0]]}], get_polygon="p",
                           get_fill_color=[0, 140, 255, 35], stroked=False))
    L.append(pdk.Layer("ColumnLayer", sites, get_position="[lon, lat]", get_elevation="n", elevation_scale=40,
                       radius=130, get_fill_color="color", extruded=True, pickable=True))
    ring = sites[sites.ditahan]
    if len(ring):
        L.append(pdk.Layer("ScatterplotLayer", ring, get_position="[lon, lat, 5]", get_radius=260, stroked=True,
                           filled=False, get_line_color=[255, 0, 0], line_width_min_pixels=3))
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
        L.append(pdk.Layer("TextLayer", sites, get_position="[lon, lat, 200]", get_text="nama", get_size=14,
                           get_color=[255, 255, 255, 255], background=True, get_background_color=[0, 0, 0, 150]))
    view = pdk.ViewState(latitude=1.64, longitude=124.76, zoom=opt["zoom"], pitch=opt["pitch"],
                         bearing=opt["bearing"], max_pitch=85)
    return pdk.Deck(layers=L, initial_view_state=view, map_style=None, tooltip={"html": "{tip}"})


# ------------------------------------------------------------------ UI
DEF = dict(zoom=10.6, pitch=55, bearing=20)
for k, v in DEF.items():
    st.session_state.setdefault(k, v)

with st.sidebar:
    st.header("⚙️ Kontrol")
    scn_name = st.selectbox("Skenario demo", list(SCENARIOS), index=0)
    live = st.toggle("Cuaca & laut live (Open-Meteo)", True)
    basemap = st.radio("Peta dasar", ["Batimetri berwarna", "Citra satelit"])
    EXAG = st.slider("Eksagerasi vertikal", 1, 8, 3)
    CLIP = st.select_slider("Potong kedalaman visual (m)", [100, 200, 300, 600, 1500], 300,
                            help="Hanya memengaruhi relief 3D; warna & profil tetap kedalaman asli.")
    water = st.toggle("Permukaan air", False)
    show_div = st.toggle("Penyelam", True)
    label = st.toggle("Label site", True)
    night = st.toggle("Mode malam", False)
    lite = st.toggle("Mode lite (tanpa bathimetri 3D)", False)
    st.slider("Zoom", 8.0, 14.0, key="zoom", step=0.1)
    st.slider("Kemiringan", 0, 85, key="pitch")
    st.slider("Putar", -180, 180, key="bearing")
    if st.button("Reset kamera"):
        for k, v in DEF.items():
            st.session_state[k] = v
        st.rerun()
    up = st.file_uploader("Data real (CSV)", type="csv",
                          help="Kolom: nama, site, depth_m, air_bar, dive_min, arah (turun/naik), sos (0/1)")
    st.button("🔄 Muat ulang data")

scn = SCENARIOS[scn_name]
wita = dt.datetime.now(ZoneInfo("Asia/Makassar"))
rng = np.random.default_rng(scn * 1000 + wita.hour * 60 + wita.minute // 5)

wx, wx_src = get_weather(scn, live, rng)
divers = place(from_csv(up) if up else simulate(scn, rng), rng)
sites = build_sites(divers, wx)
alerts = build_alerts(divers, sites)

st.title("🤿 Taman Nasional Bunaken · Monitoring Diving 3D")
st.caption(f"{wita:%d %b %Y %H:%M} WITA · Data penyelam: {'CSV upload' if up else 'simulasi demo'} · Cuaca: {wx_src}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Penyelam di air", len(divers))
c2.metric("Site aktif", int((sites.n > 0).sum()))
c3.metric("Site ditahan", int(sites.ditahan.sum()))
c4.metric("Alert", len(alerts))

left, right = st.columns([2.2, 1])
with left:
    deck = build_deck(divers, sites, dict(mode=basemap, clip=CLIP, water=water, night=night, lite=lite, divers=show_div, label=label, **{
        k: st.session_state[k] for k in DEF}))
    st.pydeck_chart(deck, height=640)
    st.caption("🔵 menyelam turun · 🟠 naik/ascent · 🔴 cincin merah = diving ditahan / SOS · Kolom = jumlah penyelam "
               "(hijau lega → merah padat). Kiri-drag: putar · Kanan-drag: geser · Scroll: zoom.")
    st.caption("☀️ 0 · 🌦️ 0,1–5 · 🌧️ 5,1–10 · ⛈️ 10,1–20 · 🌪️ >20 mm/jam. "
               "Bathimetri: AWS Terrain Tiles/GEBCO (resolusi kasar ±450 m, bukan survei lokal) · Citra: Esri World Imagery.")

with right:
    st.subheader("Batimetri")
    st.markdown(legend_html(), unsafe_allow_html=True)
    H = load_bathymetry(BBOX)
    if H is None:
        st.warning("Data batimetri gagal dimuat; peta memakai tile AWS langsung.")
    else:
        sel = st.selectbox("Profil kedalaman di site", SITE_DF.nama)
        r_ = SITE_DF.set_index("nama").loc[sel]
        _, rr, cc = depth_at(H, r_.lon, r_.lat)
        st.caption(f"Kedalaman model di titik site: {max(0, -float(H[rr, cc])):.0f} m (±), tegak lurus garis pantai tidak dijamin.")
        span = int(0.04 / (BBOX[2] - BBOX[0]) * H.shape[1])
        lo, hi = max(0, cc - span), min(H.shape[1], cc + span)
        lon = BBOX[0] + np.arange(lo, hi) / (H.shape[1] - 1) * (BBOX[2] - BBOX[0])
        st.area_chart(pd.DataFrame({"Elevasi (m)": H[rr, lo:hi]}, index=pd.Index(lon.round(4), name="Bujur")),
                      height=170)

    st.subheader("Status site & cuaca")
    t = pd.DataFrame({
        "Site": sites.nama, "Diver": sites.n.astype(str) + "/" + sites.kuota.astype(str),
        "Ombak (m)": sites.wave.round(1), "Arus (km/j)": sites.arus.round(1),
        "Hujan": [rain_icon(r) for r in sites.rain],
        "Status": np.where(sites.ditahan, "🔴 Ditahan", np.where(sites.rasio >= .8, "🟠 Padat", "🟢 Aman")),
    })
    st.dataframe(t, hide_index=True, use_container_width=True)

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
