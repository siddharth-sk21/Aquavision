"""Flask frontend for AquaVision using Leaflet and the existing pipeline."""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, render_template, request, url_for

from app import run_pipeline


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "aquavision-dev-key")


def _safe_float(value: str) -> float:
	try:
		return float(value)
	except (TypeError, ValueError) as exc:
		raise ValueError("Invalid coordinate value.") from exc


def _build_map_url(map_payload: dict[str, Any]) -> str | None:
	if not isinstance(map_payload, dict):
		return None

	if not map_payload.get("success"):
		return None

	map_path = map_payload.get("map_path")
	if not map_path or not isinstance(map_path, str):
		return None

	normalized = map_path.replace("\\", "/")
	if normalized.startswith("static/"):
		static_rel = normalized[len("static/") :]
		return url_for("static", filename=static_rel)
	return None


@app.get("/")
def home():
	return render_template("index.html")


@app.post("/analyze")
def analyze():
	location_name = (request.form.get("location_name") or "Selected Farm").strip()
	size_option = (request.form.get("size_option") or "1").strip()
	selected_lat = request.form.get("selected_lat")
	selected_lon = request.form.get("selected_lon")

	if not selected_lat or not selected_lon:
		return render_template(
			"index.html",
			error_message="Please click the exact farm location on the map before analysis.",
			previous_location_name=location_name,
			previous_size_option=size_option,
		)

	try:
		lat = _safe_float(selected_lat)
		lon = _safe_float(selected_lon)
	except ValueError:
		return render_template(
			"index.html",
			error_message="Invalid coordinates captured. Please select the point again on the map.",
			previous_location_name=location_name,
			previous_size_option=size_option,
		)

	try:
		result = run_pipeline(
			lat=lat,
			lon=lon,
			location_name=location_name,
			size_option=size_option,
		)
	except Exception as exc:
		return render_template(
			"index.html",
			error_message=f"Analysis failed: {exc}",
			previous_location_name=location_name,
			previous_size_option=size_option,
		)

	classification = result.get("classification", {}) or {}
	features = result.get("features", {}) or {}
	map_payload = result.get("map", {}) or {}
	map_url = _build_map_url(map_payload)

	return render_template(
		"result.html",
		location_name=location_name,
		lat=lat,
		lon=lon,
		buffer_meters=result.get("buffer_meters"),
		classification=classification,
		features=features,
		map_payload=map_payload,
		map_url=map_url,
	)


if __name__ == "__main__":
	port = int(os.getenv("PORT", "5000"))
	app.run(host="0.0.0.0", port=port, debug=True)
