"""Shared scoring rules for AquaVision."""

WEIGHTS = {
    "ndvi": 1.0,
    "ndwi": 1.5,
    "ndmi": 1.5,
    "elevation": 1.0,
    "slope": 1.0,
}

MAX_SCORE = sum(2 * weight for weight in WEIGHTS.values())
VERDICT_THRESHOLDS = {"high": 0.65, "medium": 0.35}
LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

THRESHOLDS = {
    "ndvi": {"medium": 0.2, "high": 0.4},
    "ndwi": {"medium": -0.1, "high": 0.1},
    "ndmi": {"medium": -0.05, "high": 0.1},
    "elevation": {"medium": 500, "high": 200},
    "slope": {"medium": 15, "high": 5},
}


def score_ndvi(value):
    return 2 if value >= 0.4 else (1 if value >= 0.2 else 0)


def score_ndwi(value):
    return 2 if value >= 0.1 else (1 if value >= -0.1 else 0)


def score_ndmi(value):
    return 2 if value >= 0.1 else (1 if value >= -0.05 else 0)


def score_elev(value):
    return 2 if value <= 200 else (1 if value <= 500 else 0)


def score_slope(value):
    return 2 if value <= 5 else (1 if value <= 15 else 0)


def score_inputs(ndvi, ndwi, ndmi, elevation, slope):
    raw_scores = {
        "ndvi": score_ndvi(ndvi),
        "ndwi": score_ndwi(ndwi),
        "ndmi": score_ndmi(ndmi),
        "elevation": score_elev(elevation),
        "slope": score_slope(slope),
    }

    weighted_total = sum(raw_scores[name] * WEIGHTS[name] for name in raw_scores)
    percentage = weighted_total / MAX_SCORE
    verdict = (
        "HIGH" if percentage >= VERDICT_THRESHOLDS["high"]
        else "MEDIUM" if percentage >= VERDICT_THRESHOLDS["medium"]
        else "LOW"
    )

    return raw_scores, weighted_total, percentage, verdict