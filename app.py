"""AquaVision application orchestration layer."""

import time
from datetime import date
from typing import Dict, Any

from classifier import classify_groundwater_potential
from gee_engine import analyze_location
from map_generator import generate_ndmi_map


BUFFER_BY_SIZE_OPTION = {
    "1": 50,   # small farm: 1-2 acres
    "2": 80,   # medium farm: 2-5 acres
    "3": 120,  # large farm: 5+ acres
}

# Simple in-memory cache for demo performance.
_PIPELINE_CACHE = {}


def get_buffer(size_option: str) -> int:
    """Map user size option to a buffer radius in meters."""
    return BUFFER_BY_SIZE_OPTION.get(str(size_option).strip(), BUFFER_BY_SIZE_OPTION["1"])


def _resolve_date_range(start_date: str, end_date: str) -> tuple[str, str]:
    """Resolve default dry-season dates when inputs are missing."""
    if start_date is None or end_date is None:
        today = date.today()
        if 2 <= today.month <= 5:
            start_date = f"{today.year}-02-01"
            end_date = today.isoformat()
        elif today.month == 1:
            start_date = f"{today.year - 1}-02-01"
            end_date = f"{today.year - 1}-05-31"
        else:
            start_date = f"{today.year}-02-01"
            end_date = f"{today.year}-05-31"
    return start_date, end_date


def run_pipeline(
    lat: float,
    lon: float,
    location_name: str,
    size_option: str = "1",
    start_date: str = None,
    end_date: str = None,
    cloud_threshold: int = 20,
) -> Dict[str, Any]:
    """Run full AquaVision flow with a single shared buffer setting."""
    buffer_meters = get_buffer(size_option)
    resolved_start_date, resolved_end_date = _resolve_date_range(start_date, end_date)
    cache_key = (lat, lon, buffer_meters, resolved_start_date, resolved_end_date, cloud_threshold)

    cached = None
    try:
        cached = _PIPELINE_CACHE.get(cache_key)
    except Exception as exc:
        print(f"Cache lookup failed, proceeding without cache: {exc}")

    if cached:
        try:
            features = cached["features"]
            map_result = cached["map"]
        except Exception as exc:
            print(f"Cache read failed, recomputing: {exc}")
            cached = None

    if cached:
        classification = classify_groundwater_potential(
            ndvi=features.get("NDVI"),
            ndwi=features.get("NDWI"),
            ndmi=features.get("NDMI"),
            elevation=features.get("elevation"),
            slope=features.get("slope"),
        )

    if not cached:
        last_error = None
        for attempt in range(1, 4):
            try:
                features = analyze_location(
                    lat=lat,
                    lon=lon,
                    start_date=resolved_start_date,
                    end_date=resolved_end_date,
                    cloud_threshold=cloud_threshold,
                    buffer_meters=buffer_meters,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    print(f"Retrying GEE request (attempt {attempt + 1}/3)...")
                    time.sleep(2)

        if last_error is not None:
            raise last_error

        classification = classify_groundwater_potential(
            ndvi=features.get("NDVI"),
            ndwi=features.get("NDWI"),
            ndmi=features.get("NDMI"),
            elevation=features.get("elevation"),
            slope=features.get("slope"),
        )

        map_result = generate_ndmi_map(
            lat=lat,
            lon=lon,
            location_name=location_name,
            start_date=resolved_start_date,
            end_date=resolved_end_date,
            cloud_threshold=cloud_threshold,
            buffer_meters=buffer_meters,
            verdict=classification["verdict"],
        )

        _PIPELINE_CACHE[cache_key] = {
            "features": features,
            "map": map_result,
        }
    return {
        "buffer_meters": buffer_meters,
        "features": features,
        "classification": classification,
        "map": map_result,
    }
# ─── LOCAL TEST CASE ──────────────────────────────────────────────────────────
# This block runs ONLY when you execute 'python app.py' directly.
# It will NOT run when we eventually hook this up to the WhatsApp server.
if __name__ == "__main__":
    import csv

    print("\n🚀 Firing up AquaVision Pipeline Test...")
    print("Connecting to Google Earth Engine... (This may take 10-15 seconds)")

    def _format_coord(value: float, pos: str, neg: str) -> str:
        suffix = pos if value >= 0 else neg
        return f"{abs(value):.4f}°{suffix}"

    def _fmt(value: float | None, fmt: str, suffix: str = "") -> str:
        if value is None:
            return "N/A"
        return f"{value:{fmt}}{suffix}"

    try:
        with open("demo_cases.csv", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                TEST_LAT = float(row["Latitude"])
                TEST_LON = float(row["Longitude"])
                LOCATION = row["Location"]
                SIZE     = row["Size"]

                # 1. Run the entire engine
                results = run_pipeline(
                    lat=TEST_LAT,
                    lon=TEST_LON,
                    location_name=LOCATION,
                    size_option=SIZE
                )

                # 2. Extract nested data
                brain_result = results["classification"]
                map_info     = results["map"]
                features     = results["features"]

                lat_label = _format_coord(TEST_LAT, "N", "S")
                lon_label = _format_coord(TEST_LON, "E", "W")

                verdict = brain_result.get("verdict")
                rule_verdict = brain_result.get("rule_verdict")
                if rule_verdict and verdict and rule_verdict == verdict:
                    agreement = "✅ Agreed"
                elif rule_verdict and verdict:
                    agreement = f"⚡ ML refinement applied ({rule_verdict} → {verdict})"
                else:
                    agreement = "N/A"

                rf_status = "Yes" if brain_result.get("ml_used") else "Rules only"
                ai_layer = "Gemini" if brain_result.get("ai_powered") else "Fallback"
                final_recommendation = {
                    "HIGH": "High groundwater potential detected. Borewell planning is reasonable after local verification.",
                    "MEDIUM": "Moderate groundwater potential detected. Local verification is recommended before drilling.",
                    "LOW": "Low groundwater potential detected. Alternative drilling locations should be considered.",
                    "UNKNOWN": "Groundwater potential could not be determined. Re-check inputs or seasonal window.",
                }.get(verdict, "Groundwater potential could not be determined. Re-check inputs or seasonal window.")

                # 3. Print report
                print("\n" + "=" * 50)
                print("🌊 AQUAVISION ANALYSIS COMPLETE")
                print("=" * 50)
                print(f"📍 Location: {LOCATION} ({lat_label}, {lon_label})")
                print(f"📏 Buffer:   {results['buffer_meters']}m radius")
                print(
                    "🛰️ Readings: "
                    f"NDMI: {_fmt(features.get('NDMI'), '.3f')}  "
                    f"NDWI: {_fmt(features.get('NDWI'), '.3f')}  "
                    f"NDVI: {_fmt(features.get('NDVI'), '.3f')}  "
                    f"Elev: {_fmt(features.get('elevation'), '.0f', 'm')}  "
                    f"Slope: {_fmt(features.get('slope'), '.1f', '°')}"
                )
                print("-" * 50)
                print(f"{brain_result['emoji']} Verdict:   {verdict}")
                print(f"📈 Confidence: {brain_result.get('ml_confidence', 'N/A')}%")
                print(f"🔀 Agreement: {agreement}")
                print(f"🤖 RF Model:  {rf_status}")
                print(f"✨ AI Layer:  {ai_layer}")
                print("-" * 50)
                print("📋 Summary:")
                print(brain_result.get("summary", ""))
                print("\n🛰️ Satellite Interpretation:")

                for reason in brain_result.get("reasons", []):
                    print(f"   • {reason}")

                print("\n💡 Advice:")
                print(brain_result.get("advice", ""))
                print("\n⚠️ Risk:")
                print(brain_result.get("risk", ""))
                print("\n📌 Final Recommendation:")
                print(final_recommendation)
                print(f"\n📂 Map: {map_info['map_path']}")
                print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n❌ PIPELINE CRASHED: {e}")
        print("Check your GEE project ID or Internet connection.")
