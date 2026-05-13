# 🌊 AquaVision – Final Hybrid Classifier 

import math
import logging
import joblib
import os
import pandas as pd
from rules import LABEL_MAP, score_inputs
from ai_advisor import generate_explanation

logging.basicConfig(level=logging.INFO, format="%(levelname)s → %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = "aqua_model.pkl"

def _load_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("ML model loaded ✅")
            return model
        except Exception as e:
            logger.warning(f"ML load failed: {e}")
    else:
        logger.warning("No ML model found → using rules")
    return None

_MODEL = _load_model()

# ─── Validation ─────────────────────────────────────────

def is_valid(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def sanitize(ndvi, ndwi, ndmi, elevation, slope):
    if is_valid(ndvi) and not (-1 <= ndvi <= 1): ndvi = None
    if is_valid(ndwi) and not (-1 <= ndwi <= 1): ndwi = None
    if is_valid(ndmi) and not (-1 <= ndmi <= 1): ndmi = None
    if is_valid(elevation) and elevation < -500: elevation = None
    if is_valid(slope) and slope < 0: slope = None
    return ndvi, ndwi, ndmi, elevation, slope

# ─── Fallback advice (Gemini failure safety net) ─────────────────────────────

advice_map_fallback = {
    "HIGH": "Good groundwater potential. Drilling likely to succeed.",
    "MEDIUM": "Moderate potential. Consider survey before drilling.",
    "LOW": "Low potential. High risk of failure.",
}

# ─── Classifier ─────────────────────────────────────────

def classify_groundwater_potential(ndvi, ndwi, ndmi, elevation, slope):

    ndvi, ndwi, ndmi, elevation, slope = sanitize(ndvi, ndwi, ndmi, elevation, slope)

    valid_count = sum(1 for v in [ndvi, ndwi, ndmi, elevation, slope] if is_valid(v))
    if valid_count < 3:
        return {
            "verdict": "UNKNOWN",
            "emoji": "⚪",
            "advice": "Insufficient satellite data.",
            "summary": "",
            "reasons": [],
            "risk": "",
            "ml_used": False,
            "ml_confidence": None,
            "rule_verdict": None,
            "ai_powered": False,
        }

    # Fill missing values with neutral values (avoid bias toward MEDIUM)
    if not is_valid(ndvi): ndvi = 0.0
    if not is_valid(ndwi): ndwi = 0.0
    if not is_valid(ndmi): ndmi = 0.0
    if not is_valid(elevation): elevation = 300   # mid-range terrain
    if not is_valid(slope): slope = 10            # moderate slope
    
    # Rule scoring
    _, _, _, rule_verdict = score_inputs(ndvi, ndwi, ndmi, elevation, slope)

    verdict = rule_verdict
    ml_used = False
    ml_conf = None

    # ML layer (safe)
    if _MODEL:
        try:
            features = pd.DataFrame(
                [[ndvi, ndwi, ndmi, elevation, slope]],
                columns=["ndvi", "ndwi", "ndmi", "elevation", "slope"]
                )
            pred = _MODEL.predict(features)[0]
            probs = _MODEL.predict_proba(features)[0]
            ml_conf = round(max(probs) * 100, 1)
            ml_pred = LABEL_MAP[pred]

            # Only trust ML if confident
            if ml_conf >= 60:
                verdict = ml_pred
                ml_used = True
            else:
                verdict = rule_verdict

        except Exception as e:
            logger.warning(f"ML error: {e}") 

    # ─── Gemini explanation layer ─────────────────────────

    try:
        explanation = generate_explanation(
            verdict=verdict,
            ndvi=ndvi,
            ndwi=ndwi,
            ndmi=ndmi,
            elevation=elevation,
            slope=slope,
            location_name="this location",
            ml_confidence=ml_conf,
            rule_verdict=rule_verdict,
        )
    except Exception as e:
        logger.warning(f"Gemini explanation failed: {e}")
        explanation = {
            "summary": "",
            "reasons": [],
            "action": advice_map_fallback.get(verdict, ""),
            "risk": "",
            "ai_powered": False,
        }

    return {
        "verdict": verdict,
        "emoji": {"HIGH":"🔵","MEDIUM":"🟡","LOW":"🔴"}[verdict],
        "advice": explanation.get("action") or advice_map_fallback.get(verdict, ""),
        "summary": explanation.get("summary", ""),
        "reasons": explanation.get("reasons", []),
        "risk": explanation.get("risk", ""),
        "ml_used": ml_used,
        "ml_confidence": ml_conf,
        "rule_verdict": rule_verdict,
        "ai_powered": explanation.get("ai_powered", False),
    }
