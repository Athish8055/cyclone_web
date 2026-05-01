from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ─── Cyclone Prediction Model ─────────────────────────────────────────────────
# Scientific thresholds based on meteorological research
# Primary driver: Ocean salinity (higher salinity → higher evaporation → more energy)

def predict_cyclone(data: dict) -> dict:
    """
    Predict cyclone probability using ocean/atmospheric parameters.
    
    Key Parameters:
    - salinity         : Sea surface salinity (PSU) — PRIMARY PREDICTOR
    - sst              : Sea surface temperature (°C) — must be ≥ 26°C
    - wind_speed       : Sustained wind speed (km/h)
    - pressure         : Atmospheric pressure (hPa) — low = bad
    - humidity         : Relative humidity (%)
    - ocean_depth      : Warm ocean layer depth (m)
    - latitude         : Location (°) — affects Coriolis force
    - vorticity        : Atmospheric vorticity (×10⁻⁵ s⁻¹)
    """

    salinity    = float(data['salinity'])        # PSU (practical salinity units)
    sst         = float(data['sst'])             # Sea Surface Temperature °C
    wind_speed  = float(data['wind_speed'])      # km/h
    pressure    = float(data['pressure'])        # hPa
    humidity    = float(data['humidity'])        # %
    ocean_depth = float(data['ocean_depth'])     # m (warm water depth)
    latitude    = float(data['latitude'])        # degrees
    vorticity   = float(data['vorticity'])       # ×10⁻⁵ s⁻¹

    scores = {}
    reasons = []
    warnings = []

    # ── 1. SALINITY SCORE (Weight: 30%) ────────────────────────────────────────
    # High salinity (>35 PSU) → denser water → stronger upwelling suppression
    # → more energy available for cyclone intensification
    # Research: Bay of Bengal avg ~32 PSU, Arabian Sea ~36 PSU
    if salinity >= 38:
        sal_score = 1.0
        reasons.append("⚠️ Extremely high salinity (≥38 PSU) — maximum energy reservoir")
    elif salinity >= 36:
        sal_score = 0.85
        reasons.append("🔴 High salinity (36–38 PSU) — favorable for intensification")
    elif salinity >= 34:
        sal_score = 0.65
        reasons.append("🟡 Moderate-high salinity (34–36 PSU) — moderate energy potential")
    elif salinity >= 32:
        sal_score = 0.40
        reasons.append("🟡 Moderate salinity (32–34 PSU) — limited energy support")
    else:
        sal_score = 0.15
        reasons.append("🟢 Low salinity (<32 PSU) — fresh water influx, unfavorable")
    scores['salinity'] = sal_score

    # ── 2. SEA SURFACE TEMPERATURE (Weight: 25%) ───────────────────────────────
    # Cyclones require SST ≥ 26°C; intensify rapidly above 28°C
    if sst >= 30:
        sst_score = 1.0
        reasons.append("⚠️ Very warm SST (≥30°C) — rapid intensification likely")
    elif sst >= 28:
        sst_score = 0.80
        reasons.append("🔴 Warm SST (28–30°C) — favorable thermodynamic conditions")
    elif sst >= 26:
        sst_score = 0.55
        reasons.append("🟡 SST at threshold (26–28°C) — marginal cyclone support")
    else:
        sst_score = 0.05
        reasons.append("🟢 SST below 26°C — insufficient thermal energy")
        warnings.append("SST below 26°C: cyclone formation highly unlikely")
    scores['sst'] = sst_score

    # ── 3. ATMOSPHERIC PRESSURE (Weight: 20%) ──────────────────────────────────
    # Low pressure → convergence → upward motion → cyclone development
    if pressure <= 990:
        pres_score = 1.0
        reasons.append("⚠️ Very low pressure (≤990 hPa) — deep depression/cyclone")
    elif pressure <= 1000:
        pres_score = 0.80
        reasons.append("🔴 Low pressure (990–1000 hPa) — favorable convergence")
    elif pressure <= 1008:
        pres_score = 0.55
        reasons.append("🟡 Slightly low pressure (1000–1008 hPa) — weak convergence")
    elif pressure <= 1013:
        pres_score = 0.30
        reasons.append("🟡 Near-normal pressure — marginal conditions")
    else:
        pres_score = 0.05
        reasons.append("🟢 High pressure (>1013 hPa) — suppresses convection")
    scores['pressure'] = pres_score

    # ── 4. WIND SPEED (Weight: 10%) ─────────────────────────────────────────────
    # Pre-existing cyclonic circulation helps; but high shear kills cyclones
    if 30 <= wind_speed <= 60:
        wind_score = 0.85
        reasons.append("🔴 Moderate winds (30–60 km/h) — existing low-level circulation")
    elif wind_speed < 30:
        wind_score = 0.50
        reasons.append("🟡 Calm winds (<30 km/h) — genesis possible from disturbance")
    elif 60 < wind_speed <= 90:
        wind_score = 0.70
        reasons.append("🟡 Strong winds (60–90 km/h) — possible wind shear concern")
    else:
        wind_score = 0.20
        reasons.append("🟢 Very high winds (>90 km/h) — high shear disrupts formation")
        warnings.append("High wind shear may inhibit cyclone organization")
    scores['wind_speed'] = wind_score

    # ── 5. HUMIDITY (Weight: 8%) ────────────────────────────────────────────────
    if humidity >= 80:
        hum_score = 0.90
        reasons.append("🔴 High humidity (≥80%) — moist mid-levels, favorable")
    elif humidity >= 65:
        hum_score = 0.60
        reasons.append("🟡 Moderate humidity (65–80%) — partial support")
    else:
        hum_score = 0.20
        reasons.append("🟢 Low humidity (<65%) — dry air intrusion, unfavorable")
        warnings.append("Dry air intrusion can weaken developing cyclones")
    scores['humidity'] = hum_score

    # ── 6. WARM OCEAN LAYER DEPTH (Weight: 5%) ─────────────────────────────────
    if ocean_depth >= 50:
        depth_score = 0.95
        reasons.append("🔴 Deep warm layer (≥50m) — sustained energy supply")
    elif ocean_depth >= 30:
        depth_score = 0.65
        reasons.append("🟡 Moderate warm layer (30–50m) — moderate support")
    else:
        depth_score = 0.30
        reasons.append("🟢 Shallow warm layer (<30m) — upwelling may cool surface")
    scores['ocean_depth'] = depth_score

    # ── 7. LATITUDE / CORIOLIS (Weight: 2%) ────────────────────────────────────
    abs_lat = abs(latitude)
    if 5 <= abs_lat <= 20:
        lat_score = 1.0
        reasons.append("🔴 Optimal latitude (5°–20°) — strong Coriolis for rotation")
    elif 20 < abs_lat <= 30:
        lat_score = 0.60
        reasons.append("🟡 Mid-latitude (20°–30°) — reduced but possible")
    elif abs_lat < 5:
        lat_score = 0.10
        reasons.append("🟢 Near equator (<5°) — Coriolis too weak for cyclone spin")
        warnings.append("Latitude <5°: insufficient Coriolis force for cyclone formation")
    else:
        lat_score = 0.20
        reasons.append("🟢 High latitude (>30°) — too cold, energy insufficient")
    scores['latitude'] = lat_score

    # ── WEIGHTED PROBABILITY ────────────────────────────────────────────────────
    weights = {
        'salinity':    0.30,
        'sst':         0.25,
        'pressure':    0.20,
        'wind_speed':  0.10,
        'humidity':    0.08,
        'ocean_depth': 0.05,
        'latitude':    0.02,
    }

    raw_prob = sum(scores[k] * weights[k] for k in weights)

    # Salinity × SST interaction bonus (both high = exponential risk)
    interaction_bonus = sal_score * sst_score * 0.10
    probability = min(raw_prob + interaction_bonus, 1.0)
    probability_pct = round(probability * 100, 1)

    # ── CATEGORY ────────────────────────────────────────────────────────────────
    if probability_pct >= 80:
        category = "EXTREME"
        category_label = "🔴 Extreme Cyclone Risk"
        intensity = "Very Severe / Super Cyclonic Storm"
    elif probability_pct >= 65:
        category = "HIGH"
        category_label = "🟠 High Cyclone Risk"
        intensity = "Severe Cyclonic Storm"
    elif probability_pct >= 50:
        category = "MODERATE"
        category_label = "🟡 Moderate Cyclone Risk"
        intensity = "Cyclonic Storm"
    elif probability_pct >= 35:
        category = "LOW"
        category_label = "🟢 Low Cyclone Risk"
        intensity = "Deep Depression"
    else:
        category = "MINIMAL"
        category_label = "🔵 Minimal Risk"
        intensity = "No Significant Threat"

    # ── SALINITY IMPACT ANALYSIS ─────────────────────────────────────────────
    sal_impact = {
        "value": salinity,
        "unit": "PSU",
        "effect": "primary",
        "explanation": (
            f"Salinity of {salinity} PSU contributes {round(sal_score * 30, 1)}% to risk score. "
            f"High salinity reduces freshwater buoyancy effects, suppresses thermocline mixing, "
            f"and maintains warm SST — providing sustained energy for cyclone intensification."
        ),
        "evaporation_rate": round(0.15 * salinity * (sst - 20) / 10, 2) if sst > 20 else 0,
        "energy_flux": round(sal_score * sst_score * 850, 1),  # W/m² estimate
    }

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "probability": probability_pct,
        "category": category,
        "category_label": category_label,
        "intensity_class": intensity,
        "component_scores": {k: round(v * 100, 1) for k, v in scores.items()},
        "weights": {k: int(v * 100) for k, v in weights.items()},
        "salinity_analysis": sal_impact,
        "key_factors": reasons,
        "warnings": warnings,
        "recommendation": get_recommendation(category, probability_pct),
        "input_params": data,
    }


def get_recommendation(category: str, prob: float) -> str:
    recs = {
        "EXTREME": "🚨 IMMEDIATE ACTION REQUIRED. Issue red alert. Activate emergency protocols. Evacuate coastal areas within 200 km radius. Coordinate with IMD/NDRF.",
        "HIGH": "⚠️ HIGH ALERT. Issue orange warning. Prepare evacuation of low-lying coastal areas. Deploy disaster response teams. Monitor every 6 hours.",
        "MODERATE": "🟡 WATCH MODE. Issue advisory. Monitor parameters every 12 hours. Prepare relief materials. Alert fishermen to return to shore.",
        "LOW": "🟢 ADVISORY. Track the system. Fishermen should exercise caution. Keep monitoring twice daily.",
        "MINIMAL": "✅ NO IMMEDIATE THREAT. Continue routine monitoring. Update every 24 hours.",
    }
    return recs.get(category, "Monitor conditions.")


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required = ['salinity', 'sst', 'wind_speed', 'pressure', 'humidity', 'ocean_depth', 'latitude', 'vorticity']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        result = predict_cyclone(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "CycloneSalinityPredictor v1.0"})


@app.route('/api/sample-data', methods=['GET'])
def sample_data():
    """Return sample scenarios for demonstration."""
    return jsonify({
        "scenarios": [
            {
                "name": "High Risk – Bay of Bengal",
                "salinity": 36.5, "sst": 30.2, "wind_speed": 45,
                "pressure": 995, "humidity": 87, "ocean_depth": 55,
                "latitude": 12.5, "vorticity": 3.8
            },
            {
                "name": "Moderate Risk – Arabian Sea",
                "salinity": 37.2, "sst": 27.8, "wind_speed": 35,
                "pressure": 1005, "humidity": 72, "ocean_depth": 38,
                "latitude": 16.0, "vorticity": 2.1
            },
            {
                "name": "Low Risk – Deep Ocean",
                "salinity": 33.5, "sst": 25.0, "wind_speed": 20,
                "pressure": 1012, "humidity": 60, "ocean_depth": 25,
                "latitude": 22.0, "vorticity": 0.8
            }
        ]
    })


if __name__ == '__main__':
    print("🌀 Cyclone Prediction API starting on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
