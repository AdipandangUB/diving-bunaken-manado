"""Fungsi geospasial TN Bunaken (tanpa Streamlit, supaya mudah diuji).

- load_contours      : baca kontur batimetri GeoJSON -> {kedalaman_m: [garis [[lon, lat], ...]]}
- fetch_terrarium    : grid elevasi dari AWS Terrarium (darat + laut), butuh internet
- grid_from_contours : grid batimetri hasil interpolasi kontur GeoJSON (cadangan saat offline)
- pack_grid          : kemas grid jadi int16 base64 untuk dikirim ke scene Three.js
- build_payload      : data lengkap untuk scene immersive (immersive.html)
"""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

import numpy as np

BBOX = (124.66, 1.52, 124.88, 1.80)  # barat, selatan, timur, utara (mencakup seluruh GeoJSON)


# ------------------------------------------------------------------ KONTUR GEOJSON
def load_contours(src) -> dict[int, list[list[list[float]]]]:
    """src: path, file-like (upload Streamlit), str JSON, atau dict FeatureCollection.
    Kedalaman dibaca dari properti 'kedalaman' (negatif) dan dikembalikan sebagai angka positif (m)."""
    if isinstance(src, dict):
        gj = src
    elif isinstance(src, (str, Path)) and Path(str(src)).exists():
        gj = json.loads(Path(src).read_text(encoding="utf-8"))
    elif hasattr(src, "read"):
        raw = src.read()
        gj = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    else:
        gj = json.loads(src)
    out: dict[int, list] = {}
    for f in gj.get("features", []):
        p, g = f.get("properties", {}), f.get("geometry") or {}
        d = p.get("kedalaman", p.get("depth", p.get("DEPTH")))
        if d is None or not g:
            continue
        lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]] if g["type"] == "LineString" else []
        out.setdefault(int(round(abs(float(d)))), []).extend([[[c[0], c[1]] for c in ln] for ln in lines if len(ln) > 1])
    return dict(sorted(out.items()))


def contour_paths(cont, z_scale: float, clip: float = 1e9):
    """Baris data untuk pydeck PathLayer: z = -kedalaman * z_scale (dipotong di -clip)."""
    rows = []
    for d, lines in cont.items():
        for ln in lines:
            z = -min(d, clip) * z_scale
            rows.append(dict(path=[[x, y, z] for x, y in ln], depth=d, label=f"-{d} m"))
    return rows


# ------------------------------------------------------------------ GRID ELEVASI
def _px(lon, lat, z):
    n = 256 * 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def fetch_terrarium(bbox=BBOX, z=11, size=512):
    """Mosaik AWS Terrarium (SRTM + GEBCO) -> grid elevasi (m; negatif = laut); None bila gagal."""
    try:
        import requests
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
        return resize_grid(crop, size)
    except Exception:
        return None


def resize_grid(H, n):
    from PIL import Image
    return np.asarray(Image.fromarray(np.asarray(H, np.float32), mode="F").resize((n, n), Image.BILINEAR), np.float32)


def grid_from_contours(cont, bbox=BBOX, n=192, iters=400):
    """Interpolasi permukaan dasar laut dari kontur: tiap sel diberi kedalaman kontur terdekat, lalu
    direlaksasi (Laplace) dengan sel yang dilewati kontur dikunci. Tidak ada daratan (kontur hanya di laut);
    di antara kontur 10 m dan pantai kedalaman dianggap datar -10 m. Baris 0 = utara."""
    w, s, e, nth = bbox
    P, V = [], []
    for d, lines in cont.items():
        for ln in lines:
            a = np.asarray(ln, float)
            for p, q in zip(a[:-1], a[1:]):
                k = max(1, int(math.hypot(*(q - p)) / 0.0006))  # ~65 m
                P.append(p + (q - p) * np.linspace(0, 1, k, endpoint=False)[:, None])
                V.append(np.full(k, -float(d)))
            P.append(a[-1:])
            V.append(np.full(1, -float(d)))
    if not P:
        return np.full((n, n), -50, np.float32)
    P, V = np.vstack(P).astype(np.float32), np.concatenate(V).astype(np.float32)
    xs = w + (np.arange(n) + .5) / n * (e - w)
    ys = nth - (np.arange(n) + .5) / n * (nth - s)
    X, Y = np.meshgrid(xs, ys)
    qx, qy = X.ravel().astype(np.float32), Y.ravel().astype(np.float32)
    H = np.empty(n * n, np.float32)
    for i in range(0, n * n, 384):  # terdekat, per potongan agar hemat memori
        dx = qx[i:i + 384, None] - P[None, :, 0]
        dy = qy[i:i + 384, None] - P[None, :, 1]
        H[i:i + 384] = V[np.argmin(dx * dx + dy * dy, axis=1)]
    H = H.reshape(n, n)
    ix = np.clip(((P[:, 0] - w) / (e - w) * n).astype(int), 0, n - 1)
    iy = np.clip(((nth - P[:, 1]) / (nth - s) * n).astype(int), 0, n - 1)
    tot, cnt = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
    np.add.at(tot, (iy, ix), V)
    np.add.at(cnt, (iy, ix), 1)
    fixed = cnt > 0
    fix_val = np.where(fixed, tot / np.maximum(cnt, 1), 0)
    H = np.where(fixed, fix_val, H)
    for _ in range(iters):
        p = np.pad(H, 1, mode="edge")
        H = np.where(fixed, fix_val, 0.25 * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]))
    return H.astype(np.float32)


def pack_grid(H, n=256):
    """Grid -> int16 (meter) base64, baris 0 = utara."""
    G = resize_grid(H, n) if H.shape != (n, n) else np.asarray(H, np.float32)
    return dict(w=n, h=n, data=base64.b64encode(np.round(np.clip(G, -32000, 32000)).astype("<i2").tobytes()).decode())


def sample(H, lon, lat, bbox=BBOX):
    r = int(round((bbox[3] - lat) / (bbox[3] - bbox[1]) * (H.shape[0] - 1)))
    c = int(round((lon - bbox[0]) / (bbox[2] - bbox[0]) * (H.shape[1] - 1)))
    return float(H[min(max(r, 0), H.shape[0] - 1), min(max(c, 0), H.shape[1] - 1)])


# ------------------------------------------------------------------ PAYLOAD SCENE IMMERSIVE
def build_payload(cont, H, spots, turtles, species, bbox=BBOX, extra=None):
    """spots: [(nama, lon, lat, pulau, kuota)], turtles: [(jenis, lon, lat, pulau)]."""
    pl = dict(
        bbox=list(bbox),
        grid=pack_grid(H, 256),
        contours=[dict(d=int(d), lines=[[[round(x, 5), round(y, 5)] for x, y in ln] for ln in lines])
                  for d, lines in cont.items()],
        spots=[dict(n=n, lon=lo, lat=la, isl=p, quota=q) for n, lo, la, p, q in spots],
        turtles=[dict(sp=k, lon=lo, lat=la, isl=p) for k, lo, la, p in turtles],
        species={k: dict(nama=v["nama"], latin=v["latin"], rgb=list(v["rgb"]), poin=v["poin"], fakta=v["fakta"])
                 for k, v in species.items()},
    )
    pl.update(extra or {})
    return pl


def inject(template: str, payload: dict) -> str:
    """Sisipkan payload JSON ke immersive.html (menggantikan token /*__DATA__*/null)."""
    js = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return template.replace("/*__DATA__*/null", js, 1)
