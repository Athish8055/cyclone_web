# 🌀 CycloneAI — Ocean Salinity-Based Cyclone Prediction System

## Overview

A scientific cyclone prediction system that uses **ocean salinity as the primary driver**,
combined with other oceanographic and atmospheric parameters.

## Scientific Basis

| Parameter        | Weight | Role                                                      |
|-----------------|--------|-----------------------------------------------------------|
| **Salinity**    | **30%**| PRIMARY — Higher salinity → less freshwater dilution → maintains warm SST → more energy |
| SST             | 25%    | Must be ≥26°C; ≥28°C = rapid intensification              |
| Pressure        | 20%    | Low pressure = convergence = upward motion                |
| Wind Speed      | 10%    | Existing circulation; high shear kills cyclones           |
| Humidity        | 8%     | Moist mid-levels favor convection                         |
| Ocean Depth     | 5%     | Warm layer depth prevents upwelling cooling               |
| Latitude        | 2%     | Coriolis force (optimal: 5°–20°)                          |

## How Salinity Predicts Cyclones

1. **High salinity (>35 PSU)** → Water is denser → resists vertical mixing
2. **Less mixing** → Warm surface water stays warm → SST remains high
3. **High SST + High Salinity** → Maximum evaporation → More water vapor = cyclone fuel
4. **Arabian Sea (36–37 PSU)** has historically produced intense cyclones
5. **Bay of Bengal (30–32 PSU)** has lower salinity but compensates with higher SST

## Setup & Installation

### Backend (Python Flask)

```bash
cd cyclone_predictor

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
# → Runs on http://localhost:5000
```

### Frontend (Static HTML)

Just open `index.html` in any browser. The frontend:
- Tries to connect to `http://localhost:5000` (Flask backend)
- Falls back to built-in JavaScript prediction model if backend is offline

## API Endpoints

### POST /api/predict
```json
{
  "salinity": 36.5,
  "sst": 30.2,
  "wind_speed": 45,
  "pressure": 995,
  "humidity": 87,
  "ocean_depth": 55,
  "latitude": 12.5,
  "vorticity": 2.5
}
```

**Response:**
```json
{
  "probability": 78.4,
  "category": "HIGH",
  "category_label": "🟠 High Cyclone Risk",
  "intensity_class": "Severe Cyclonic Storm",
  "salinity_analysis": {
    "value": 36.5,
    "effect": "primary",
    "evaporation_rate": 1.26,
    "energy_flux": 612.0
  },
  "component_scores": {...},
  "key_factors": [...],
  "recommendation": "..."
}
```

### GET /api/health
### GET /api/sample-data

## Risk Categories

| Probability | Category | Intensity               |
|-------------|----------|------------------------|
| ≥ 80%       | EXTREME  | Super Cyclonic Storm   |
| 65–80%      | HIGH     | Severe Cyclonic Storm  |
| 50–65%      | MODERATE | Cyclonic Storm         |
| 35–50%      | LOW      | Deep Depression        |
| < 35%       | MINIMAL  | No Threat              |

## Future Enhancements

- [ ] Real-time data integration (INCOIS/NOAA APIs)
- [ ] Machine learning model trained on historical cyclone track data
- [ ] Satellite SST and salinity data ingestion (SMOS, Aquarius)
- [ ] Track prediction (path forecasting)
- [ ] Time-series analysis and 72-hour forecast
- [ ] SMS/email alert system integration

## Data Sources for Real-World Use

- **INCOIS**: Indian National Centre for Ocean Information Services
- **IMD**: India Meteorological Department
- **NOAA**: National Oceanic and Atmospheric Administration
- **SMOS/Aquarius**: ESA/NASA satellite salinity datasets
- **ERA5**: ECMWF Reanalysis v5 (pressure, humidity, wind)
