"""
AquaVision – AI Explanation Layer
===================================
Gemini's ONLY job here is to generate a concise, technical explanation.

It does NOT:
  ✗ Decide the verdict
  ✗ Override the classifier
  ✗ Re-run hydro logic

It ONLY:
    ✓ Explains WHY the satellite numbers led to this verdict
    ✓ Gives one practical decision-support recommendation
    ✓ Keeps the tone professional and hydrogeology-focused

Usage:
    from ai_advisor import generate_explanation
    result = generate_explanation(
        verdict="HIGH",
        ndvi=0.45, ndwi=0.12, ndmi=0.15,
        elevation=180.0, slope=3.2,
        location_name="Bijapur",
        ml_confidence=82.5,
        rule_verdict="HIGH",
    )
    print(result["summary"])
    print(result["reasons"])
    print(result["action"])
    print(result["risk"])
"""

import os
import logging
import textwrap
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Gemini setup ──────────────────────────────────────────────────────────────

try:
    from google import genai
    _API_KEY = os.getenv("GEMINI_API_KEY")
    if _API_KEY:
        _GEMINI_CLIENT = genai.Client(api_key=_API_KEY)
        logger.info("Gemini AI explanation layer loaded ✅")
    else:
        _GEMINI_CLIENT = None
        logger.warning("GEMINI_API_KEY not set → AI explanations disabled")
except ImportError:
    _GEMINI_CLIENT = None
    logger.warning("google-genai not installed → AI explanations disabled")


# ── Satellite reading interpretation helpers ──────────────────────────────────

def _ndvi_label(v: float) -> str:
    if v >= 0.4:  return f"{v:.3f} (dense vegetation)"
    if v >= 0.2:  return f"{v:.3f} (moderate vegetation)"
    return             f"{v:.3f} (sparse / bare soil)"

def _ndwi_label(v: float) -> str:
    if v >= 0.1:   return f"{v:.3f} (surface water present)"
    if v >= -0.1:  return f"{v:.3f} (limited surface water)"
    return              f"{v:.3f} (dry surface conditions)"

def _ndmi_label(v: float) -> str:
    if v >= 0.1:   return f"{v:.3f} (high soil moisture)"
    if v >= -0.05: return f"{v:.3f} (moderate soil moisture)"
    return              f"{v:.3f} (low soil moisture)"

def _elev_label(v: float) -> str:
    if v <= 200:  return f"{v:.0f}m (low elevation — recharge-favorable)"
    if v <= 500:  return f"{v:.0f}m (moderate elevation)"
    return             f"{v:.0f}m (high elevation — reduced recharge)"

def _slope_label(v: float) -> str:
    if v <= 5:   return f"{v:.1f}° (flat — higher infiltration potential)"
    if v <= 15:  return f"{v:.1f}° (gentle slope)"
    return            f"{v:.1f}° (steep — runoff-dominant)"


# ── Fallback explanations (no Gemini) ────────────────────────────────────────

_FALLBACK = {
    "HIGH": {
        "summary": "Satellite data suggests strong groundwater potential in this area.",
        "reasons": [
            "NDMI values indicate elevated subsurface moisture retention.",
            "NDVI shows dense vegetation consistent with reliable moisture access.",
            "Low elevation and gentle slope support groundwater infiltration.",
        ],
        "action":  "Drilling is reasonable after quick local checks on soil and depth.",
        "risk":    "Yields may drop if the borewell is placed on rocky or sloped patches.",
    },
    "MEDIUM": {
        "summary": "Satellite data suggests moderate groundwater potential in this area.",
        "reasons": [
            "NDMI values indicate moderate subsurface moisture retention.",
            "NDVI shows mixed vegetation, suggesting uneven moisture availability.",
            "Terrain is acceptable but not strongly recharge-favorable.",
        ],
        "action":  "Local verification is recommended before drilling.",
        "risk":    "Water availability may reduce during long dry seasons.",
    },
    "LOW": {
        "summary": "Satellite data suggests low groundwater potential in this area.",
        "reasons": [
            "NDMI is low, indicating limited subsurface moisture retention.",
            "NDVI is sparse, suggesting weak vegetation-supported moisture signals.",
            "Slope/elevation conditions reduce infiltration and recharge likelihood.",
        ],
        "action":  "Consider alternative locations or other water sources before drilling.",
        "risk":    "High chance of a low-yield or dry borewell.",
    },
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    verdict:       str,
    ndvi:          float,
    ndwi:          float,
    ndmi:          float,
    elevation:     float,
    slope:         float,
    location_name: str,
    ml_confidence: float,
    rule_verdict:  str,
) -> str:

    agreed = "✓ Models agree" if ml_confidence and rule_verdict == verdict else "⚠ Models disagreed — rules prioritized"

    return textwrap.dedent(f"""
        You are a groundwater analyst writing a concise, decision-support note.
        Your explanation must be based ONLY on the satellite readings below.

        ════════════════════════════════════════════
        SCIENTIFIC VERDICT (already decided — DO NOT change it): {verdict}
        Location: {location_name}
        Model agreement: {agreed}
        ML Confidence: {f"{ml_confidence}%" if ml_confidence else "N/A"}
        ════════════════════════════════════════════

        SATELLITE READINGS (use these numbers in your explanation):
          🌿 Vegetation  (NDVI):  {_ndvi_label(ndvi)}
          💧 Water Index (NDWI):  {_ndwi_label(ndwi)}
          🌊 Moisture    (NDMI):  {_ndmi_label(ndmi)}
          ⛰  Elevation:           {_elev_label(elevation)}
          📐 Slope:               {_slope_label(slope)}

        ════════════════════════════════════════════
        YOUR TASK — respond in EXACTLY this format, nothing else:

        SUMMARY: [1 sentence. Simple, farmer-friendly statement of groundwater potential.]

        REASONS:
        - [Moisture: cite NDMI and subsurface moisture retention]
        - [Vegetation: cite NDVI and moisture availability signal]
        - [Terrain: cite elevation/slope and infiltration or runoff tendency]

        ACTION: [1 specific, farmer-friendly recommendation]

        RISK: [1 sentence in simple language about likely downside]
        ════════════════════════════════════════════

        Rules you must follow:
        - DO NOT question, soften, or second-guess the verdict: {verdict}
        - DO NOT say "however" or "but" about the verdict
        - DO NOT use greetings or informal phrases
        - Use technical terms only in REASONS, keep SUMMARY/ACTION/RISK simple
        - Reference actual numbers from the satellite readings above
        - Keep total response under 90 words
    """).strip()


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_response(text: str) -> dict:
    """
    Extracts SUMMARY / REASONS / ACTION / RISK from Gemini's structured output.
    Falls back to returning the raw text if parsing fails.
    """
    result = {
        "summary": "",
        "reasons": [],
        "action":  "",
        "risk":    "",
        "raw":     text,
    }

    try:
        lines = text.strip().splitlines()
        current = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("SUMMARY:"):
                current = "summary"
                result["summary"] = line.split(":", 1)[-1].strip()

            elif line.upper().startswith("REASONS:"):
                current = "reasons"

            elif line.upper().startswith("ACTION:"):
                current = "action"
                result["action"] = line.split(":", 1)[-1].strip()

            elif line.upper().startswith("RISK:"):
                current = "risk"
                result["risk"] = line.split(":", 1)[-1].strip()

            elif current == "reasons" and line.startswith("-"):
                reason = line.lstrip("- ").strip()
                if reason:
                    result["reasons"].append(reason)

            elif current == "summary" and not result["summary"]:
                result["summary"] = line

            elif current == "action" and not result["action"]:
                result["action"] = line.split(":", 1)[-1].strip()

            elif current == "risk" and not result["risk"]:
                result["risk"] = line.split(":", 1)[-1].strip()

        result["reasons"] = result["reasons"][:3]

    except Exception as e:
        logger.warning(f"Response parsing failed: {e} — using raw output")

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def generate_explanation(
    verdict:       str,
    ndvi:          float,
    ndwi:          float,
    ndmi:          float,
    elevation:     float,
    slope:         float,
    location_name: str   = "this location",
    ml_confidence: float = None,
    rule_verdict:  str   = None,
) -> dict:
    """
    Generate a Gemini-powered human explanation for the classifier's verdict.

    Parameters
    ----------
    verdict       : Final verdict from classifier ("HIGH" / "MEDIUM" / "LOW")
    ndvi          : Vegetation index from GEE
    ndwi          : Water index from GEE
    ndmi          : Moisture index from GEE
    elevation     : Elevation in meters
    slope         : Slope in degrees
    location_name : Farmer's location label
    ml_confidence : RandomForest confidence % (optional)
    rule_verdict  : What rule engine decided (for agreement check)

    Returns
    -------
    dict with keys:
        summary     → str   (1 sentence plain language summary)
        reasons     → list  (up to 3 bullet reasons)
        action      → str   (1 next step for farmer)
        risk        → str   (consequence of ignoring)
        raw         → str   (full Gemini response)
        ai_powered  → bool  (True if Gemini responded, False if fallback)
        error       → str | None
    """

    if verdict not in ("HIGH", "MEDIUM", "LOW"):
        return {
            **_FALLBACK.get("MEDIUM", {}),
            "ai_powered": False,
            "error": f"Unexpected verdict value: {verdict}",
            "raw": "",
        }

    # ── No Gemini available → use fallback ────────────────────────────────────
    if _GEMINI_CLIENT is None:
        logger.info("Gemini unavailable — using static fallback explanation")
        return {
            **_FALLBACK[verdict],
            "ai_powered": False,
            "error": "Gemini not configured (check GEMINI_API_KEY)",
            "raw": "",
        }

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        prompt   = _build_prompt(
            verdict, ndvi, ndwi, ndmi, elevation, slope,
            location_name, ml_confidence, rule_verdict or verdict
        )
        response = _GEMINI_CLIENT.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
        )
        raw_text = response.text.strip()

        parsed = _parse_response(raw_text)

        # Safety net — if parsing returned empty fields, fill from fallback
        fallback = _FALLBACK[verdict]
        return {
            "summary":    parsed["summary"]  or fallback["summary"],
            "reasons":    parsed["reasons"]  or fallback["reasons"],
            "action":     parsed["action"]   or fallback["action"],
            "risk":       parsed["risk"]     or fallback["risk"],
            "raw":        raw_text,
            "ai_powered": True,
            "error":      None,
        }

    except Exception as e:
        logger.warning(f"Gemini call failed: {e} — falling back to static advice")
        return {
            **_FALLBACK[verdict],
            "ai_powered": False,
            "error": str(e),
            "raw": "",
        }


# ── WhatsApp message formatter ────────────────────────────────────────────────

def format_whatsapp_message(
    verdict:       str,
    location_name: str,
    explanation:   dict,
    features:      dict = None,
) -> str:
    """
    Formats the full WhatsApp message combining classifier verdict
    and Gemini explanation.

    Parameters
    ----------
    verdict       : "HIGH" / "MEDIUM" / "LOW"
    location_name : Farmer's location
    explanation   : Output from generate_explanation()
    features      : Raw GEE features dict (optional, for index display)
    """
    emoji_map = {"HIGH": "🔵", "MEDIUM": "🟡", "LOW": "🔴"}
    emoji = emoji_map.get(verdict, "⚪")

    reasons_text = "\n".join(
        f"  • {r}" for r in explanation.get("reasons", [])
    )

    features_text = ""
    if features:
        features_text = (
            f"\n📡 *Satellite Readings:*\n"
            f"  NDMI (Moisture): {features.get('NDMI', 0):.3f}\n"
            f"  NDWI (Water):    {features.get('NDWI', 0):.3f}\n"
            f"  NDVI (Veg):      {features.get('NDVI', 0):.3f}\n"
        )

    powered_by = (
        "🤖 _Explained by Gemini AI_"
        if explanation.get("ai_powered")
        else "📋 _Rule-based explanation_"
    )

    return (
        f"🛰️ *AquaVision Report*\n"
        f"📍 {location_name}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{emoji} *Verdict: {verdict}*\n\n"
        f"📝 *Summary:*\n{explanation.get('summary', '')}\n\n"
        f"🔬 *Why this verdict:*\n{reasons_text}\n\n"
        f"✅ *What to do next:*\n{explanation.get('action', '')}\n\n"
        f"⚠️ *Risk if ignored:*\n{explanation.get('risk', '')}\n"
        f"{features_text}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{powered_by}"
    )
