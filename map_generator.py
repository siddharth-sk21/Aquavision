"""
AquaVision - Enhanced Map Generator
=====================================
Generates a WhatsApp-ready PNG with:
  - NDMI satellite overlay      (GEE)
  - 📍 Farmer location marker   (center pin)
  - 🔵 Buffer radius circle     (analyzed area boundary)
  - Verdict title bar           (HIGH / MEDIUM / LOW)
  - Legend                      (color → moisture level)
  - Scale bar                   (distance reference)
  - Coordinate footnote

Flow:
  GEE getThumbURL → PIL Image → Matplotlib overlay → PNG saved
"""

import io
import math
import urllib.request
import matplotlib
matplotlib.use("Agg")                   # non-interactive, safe for Flask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from PIL import Image
import numpy as np
from pathlib import Path
from datetime import date as date_cls

from gee_engine import initialize_ee
from rules import THRESHOLDS
import ee

# ─── Output folder ──────────────────────────────────────────────────────────────
MAPS_DIR = Path("static/maps")
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ──────────────────────────────────────────────────────────────────
MAP_WIDTH  = 512
MAP_HEIGHT = 512
DPI        = 100   # → 5.12 × 5.12 inch figure at 100 dpi = 512 × 512 px

# Hex colors WITHOUT '#' for GEE palette
NDMI_GEE_PALETTE = ["d73027", "fdae61", "2c7bb6"]

# Same colors WITH '#' for matplotlib
NDMI_MPL_COLORS  = {"LOW": "#d73027", "MEDIUM": "#fdae61", "HIGH": "#2c7bb6"}
VERDICT_COLORS   = {"HIGH": "#2c7bb6", "MEDIUM": "#fdae61", "LOW": "#d73027", "UNKNOWN": "#888888"}


# ─── Helpers ────────────────────────────────────────────────────────────────────

def _is_valid_coord(lat: float, lon: float) -> bool:
    return (
        isinstance(lat, (int, float)) and math.isfinite(lat) and -90  <= lat <= 90  and
        isinstance(lon, (int, float)) and math.isfinite(lon) and -180 <= lon <= 180
    )


def _resolve_dates(start_date, end_date):
    """Return a valid (start, end) date string pair using dry-season defaults."""
    if start_date and end_date:
        return start_date, end_date
    today = date_cls.today()
    if 2 <= today.month <= 5:
        return f"{today.year}-02-01", today.isoformat()
    if today.month == 1:
        return f"{today.year - 1}-02-01", f"{today.year - 1}-05-31"
    return f"{today.year}-02-01", f"{today.year}-05-31"


# ─── Step 1: Fetch NDMI image from GEE ──────────────────────────────────────────

def _fetch_ndmi_pil(lat, lon, start_date, end_date, cloud_threshold, buffer_meters):
    """
    Pulls NDMI-classified thumbnail from GEE and returns a PIL Image.
    Returns (PIL.Image, None) on success or (None, error_string) on failure.
    """
    point  = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_meters).bounds()

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
    )

    def mask_clouds(img):
        scl     = img.select("SCL")
        allowed = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return img.updateMask(allowed)

    if collection.size().getInfo() == 0:
        return None, f"No satellite images found ({start_date} → {end_date})"

    image = collection.map(mask_clouds).median()
    ndmi  = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

    # Classify into 0 (LOW) / 1 (MEDIUM) / 2 (HIGH)
    ndmi_score = (
        ee.Image(0)
        .where(ndmi.gte(THRESHOLDS["ndmi"]["medium"]), 1)
        .where(ndmi.gte(THRESHOLDS["ndmi"]["high"]),   2)
        .rename("NDMI_CLASS")
    )

    thumb_url = ndmi_score.getThumbURL({
        "min":        0,
        "max":        2,
        "palette":    NDMI_GEE_PALETTE,
        "region":     region,
        "dimensions": f"{MAP_WIDTH}x{MAP_HEIGHT}",
        "format":     "png",
    })

    with urllib.request.urlopen(thumb_url) as resp:
        raw = resp.read()

    return Image.open(io.BytesIO(raw)).convert("RGB"), None


# ─── Step 2: Overlay everything with Matplotlib ─────────────────────────────────

def _draw_and_save(
    pil_img:       Image.Image,
    lat:           float,
    lon:           float,
    location_name: str,
    buffer_meters: int,
    verdict:       str,
    map_path:      Path,
) -> None:
    """
    Composites NDMI image + overlays onto a matplotlib figure and saves PNG.

    Coordinate system used: image pixels (0,0) = bottom-left, (512,512) = top-right.
    Farmer point is always at the center (256, 256).
    """
    fig, ax = plt.subplots(
        figsize=(MAP_WIDTH / DPI, MAP_HEIGHT / DPI),
        dpi=DPI,
    )
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    # ── NDMI background ───────────────────────────────────────────────────────
    # imshow with origin="upper" maps row 0 → top of image
    # We flip via extent so our coordinate system has (0,0) at bottom-left.
    img_arr = np.array(pil_img)
    ax.imshow(
        img_arr,
        extent=[0, MAP_WIDTH, 0, MAP_HEIGHT],
        origin="upper",
        zorder=1,
    )

    cx = MAP_WIDTH  / 2   # 256
    cy = MAP_HEIGHT / 2   # 256

    # ── Buffer radius circle ───────────────────────────────────────────────────
    # The GEE thumbnail spans the bounding box of point.buffer(buffer_meters)
    # so the circle fills ~92% of each dimension.
    radius_px = (MAP_WIDTH / 2) * 0.92

    buf_circle = Circle(
        (cx, cy),
        radius    = radius_px,
        fill      = False,
        edgecolor = "#00e5ff",
        linewidth = 2.5,
        linestyle = "--",
        alpha     = 0.90,
        zorder    = 3,
    )
    ax.add_patch(buf_circle)

    # Radius distance label on the right side of circle
    ax.text(
        cx + radius_px + 6, cy,
        f"{buffer_meters}m",
        color="white", fontsize=6.5, va="center", ha="left",
        fontweight="bold", zorder=5,
    )

    # ── Farmer location marker ─────────────────────────────────────────────────
    # Outer glow ring
    ax.plot(cx, cy, "o",
            markersize=20, color="#ff000055", zorder=4)
    # Main red dot
    ax.plot(cx, cy, "o",
            markersize=13, color="#ff3333",
            markeredgecolor="white", markeredgewidth=2.5, zorder=5)
    # Inner white dot
    ax.plot(cx, cy, "o",
            markersize=5, color="white", zorder=6)

    # ── Location label above marker ────────────────────────────────────────────
    ax.text(
        cx, cy + 22,
        f"{location_name}",
        color="white", fontsize=8, ha="center", va="bottom",
        fontweight="bold",
        bbox=dict(
               boxstyle="round,pad=0.3",
               facecolor="black",
               alpha=0.35,
               edgecolor="none",
        ),
        zorder=7,
    )
    ax.text(
    14, 14,
    "AquaVision",
    color="white", fontsize=6.5, ha="left", va="bottom",
    alpha=0.40, zorder=8,
    )

    # ── Legend (bottom-right) ──────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(facecolor=NDMI_MPL_COLORS["HIGH"],   label="High moisture"),
        mpatches.Patch(facecolor=NDMI_MPL_COLORS["MEDIUM"], label="Medium moisture"),
        mpatches.Patch(facecolor=NDMI_MPL_COLORS["LOW"],    label="Low moisture"),
    ]
    legend = ax.legend(
        handles     = legend_patches,
        loc         = "lower right",
        fontsize    = 6.5,
           framealpha  = 0.35,
           facecolor   = "black",
           edgecolor   = "none",
        labelcolor  = "white",
        handlelength= 1.1,
        handleheight= 0.85,
        borderpad   = 0.6,
    )
    legend.set_zorder(8)

    # ── Verdict title bar (top) ────────────────────────────────────────────────
    ax.text(
        18, MAP_HEIGHT - 18,
        f" Verdict: {verdict}",
        color      = "#0d0b02",
        fontsize   = 7.5,
        ha         = "left",
        va         = "top",
        fontweight = "bold",
        bbox       = dict(
            boxstyle  = "round,pad=0.35",
            facecolor = "none",
            alpha     = 0.0,
               edgecolor = "none",
        ),
        zorder=8,
    )

    # ── North indicator (top-right) ───────────────────────────────────────────
    north_label = "N ^" if lat >= 0 else "N v"
    ax.text(
        MAP_WIDTH - 28, MAP_HEIGHT - 28,
        north_label,
        color="#0d0b02", fontsize=10, ha="right", va="center",
        fontweight="bold", alpha=0.85, zorder=8,
    )

    # ── Coordinate footnote (very bottom) ─────────────────────────────────────
    ax.text(
        cx, 11,
        f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}   {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}",
        color="white", fontsize=6.5, ha="center", va="center",
        alpha=0.75, zorder=8,
    )

    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.axis("off")

    plt.tight_layout(pad=0)
    fig.savefig(
        map_path,
        dpi        = DPI,
        bbox_inches= "tight",
        pad_inches = 0,
        facecolor  = fig.get_facecolor(),
    )
    plt.close(fig)


# ─── Public API ─────────────────────────────────────────────────────────────────

def generate_ndmi_map(
    lat:             float,
    lon:             float,
    location_name:   str  = "location",
    start_date:      str  = None,
    end_date:        str  = None,
    cloud_threshold: int  = 20,
    buffer_meters:   int  = 50,
    verdict:         str  = "UNKNOWN",
) -> dict:
    """
    Generate an enhanced NDMI map PNG ready for WhatsApp delivery.

    Returns dict:
        success    → bool
        map_path   → str path to saved PNG  (None on failure)
        error      → str error message      (None on success)
    """
    if not _is_valid_coord(lat, lon):
        return {
            "success":  False,
            "map_path": None,
            "error":    f"Invalid coordinates: lat={lat}, lon={lon}",
        }

    try:
        initialize_ee()
    except RuntimeError as e:
        return {"success": False, "map_path": None, "error": str(e)}

    try:
        start_date, end_date = _resolve_dates(start_date, end_date)

        # ── Fetch NDMI image from GEE ──────────────────────────────────────────
        pil_img, error = _fetch_ndmi_pil(
            lat, lon, start_date, end_date, cloud_threshold, buffer_meters
        )
        if pil_img is None:
            return {"success": False, "map_path": None, "error": error}

        # ── Build output path ──────────────────────────────────────────────────
        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in location_name.lower().replace(" ", "_")
        )
        map_path = MAPS_DIR / f"ndmi_{safe_name}_{lat:.4f}_{lon:.4f}.png"

        # ── Draw overlays and save ─────────────────────────────────────────────
        _draw_and_save(pil_img, lat, lon, location_name, buffer_meters, verdict, map_path)

        return {"success": True, "map_path": str(map_path), "error": None}

    except ee.EEException as e:
        return {"success": False, "map_path": None, "error": f"GEE error: {e}"}
    except Exception as e:
        return {"success": False, "map_path": None, "error": f"Unexpected error: {e}"}


# ─── WhatsApp Caption (unchanged) ───────────────────────────────────────────────

def format_map_caption(verdict: str, location_name: str, reasons=None) -> str:
    emoji_map = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔴", "UNKNOWN": "⚪"}
    emoji = emoji_map.get(verdict, "⚪")

    reason_text = ""
    if reasons:
        reason_lines = "\n".join(f"- {r}" for r in reasons[:3])
        reason_text  = f"\n💡 *Why:*\n{reason_lines}\n"

    return (
        f"🛰️ *AquaVision NDMI Map*\n"
        f"📍 {location_name}\n"
        f"\n"
        f"{emoji} Verdict: *{verdict}*\n"
        f"{reason_text}"
        f"\n"
        f"🔵 Blue   = High soil moisture\n"
        f"🟡 Yellow = Medium soil moisture\n"
        f"🔴 Red    = Low soil moisture\n"
        f"⭕ Circle = Analyzed area\n"
        f"📍 Pin    = Your location\n"
        f"\n"
        f"_Final verdict uses multi-factor AI classifier._"
    )