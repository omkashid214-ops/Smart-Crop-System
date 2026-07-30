"""
generate_datasets.py
---------------------
Generates realistic SYNTHETIC datasets (5000+ rows each, balanced classes,
no missing values) for:
    1. Disease Risk Prediction   -> dataset/disease_risk.csv
    2. Fertilizer Recommendation -> dataset/fertilizer_recommendation.csv
    3. Irrigation Recommendation -> dataset/irrigation.csv
    4. Farmer Records            -> dataset/farmer_records.csv

The real Crop_recommendation.csv (Kaggle, 2200 rows, 22 crops) is used as-is
and is NOT touched by this script.

Run once:  python utils/generate_datasets.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

CROPS = [
    "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
    "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
    "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
    "coconut", "cotton", "jute", "coffee",
]

STATES = [
    "Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka", "Tamil Nadu",
    "Gujarat", "Madhya Pradesh", "Rajasthan", "West Bengal", "Bihar",
    "Andhra Pradesh", "Haryana",
]

SEASONS = ["Kharif", "Rabi", "Zaid"]

# Disease catalogue: crop-family -> plausible diseases
DISEASE_MAP = {
    "rice": ["Rice Blast", "Bacterial Leaf Blight", "Sheath Blight", "Healthy"],
    "maize": ["Maize Leaf Blight", "Common Rust", "Gray Leaf Spot", "Healthy"],
    "chickpea": ["Ascochyta Blight", "Fusarium Wilt", "Healthy"],
    "kidneybeans": ["Bean Rust", "Angular Leaf Spot", "Healthy"],
    "pigeonpeas": ["Fusarium Wilt", "Sterility Mosaic", "Healthy"],
    "mothbeans": ["Yellow Mosaic Virus", "Powdery Mildew", "Healthy"],
    "mungbean": ["Yellow Mosaic Virus", "Cercospora Leaf Spot", "Healthy"],
    "blackgram": ["Yellow Mosaic Virus", "Leaf Crinkle", "Healthy"],
    "lentil": ["Rust", "Wilt", "Healthy"],
    "pomegranate": ["Bacterial Blight", "Fruit Rot", "Healthy"],
    "banana": ["Panama Wilt", "Sigatoka Leaf Spot", "Healthy"],
    "mango": ["Anthracnose", "Powdery Mildew", "Healthy"],
    "grapes": ["Downy Mildew", "Powdery Mildew", "Healthy"],
    "watermelon": ["Fusarium Wilt", "Anthracnose", "Healthy"],
    "muskmelon": ["Powdery Mildew", "Downy Mildew", "Healthy"],
    "apple": ["Apple Scab", "Fire Blight", "Healthy"],
    "orange": ["Citrus Canker", "Greening (HLB)", "Healthy"],
    "papaya": ["Papaya Ring Spot Virus", "Powdery Mildew", "Healthy"],
    "coconut": ["Bud Rot", "Leaf Blight", "Healthy"],
    "cotton": ["Bacterial Blight", "Leaf Curl Virus", "Healthy"],
    "jute": ["Stem Rot", "Anthracnose", "Healthy"],
    "coffee": ["Coffee Leaf Rust", "Berry Disease", "Healthy"],
}

FERTILIZERS = [
    ("Urea", 46, 0, 0),
    ("DAP", 18, 46, 0),
    ("MOP", 0, 0, 60),
    ("NPK 19-19-19", 19, 19, 19),
    ("NPK 10-26-26", 10, 26, 26),
    ("SSP", 0, 16, 0),
    ("Ammonium Sulphate", 21, 0, 0),
]

ORGANIC_ALTERNATIVES = [
    "Vermicompost", "Farmyard Manure (FYM)", "Neem Cake", "Green Manure",
    "Compost + Biofertilizer", "Poultry Manure",
]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


# ----------------------------------------------------------------------
# 1. DISEASE RISK DATASET
# ----------------------------------------------------------------------
def generate_disease_risk(n_per_crop: int = 250) -> pd.DataFrame:
    rows = []
    for crop in CROPS:
        diseases = DISEASE_MAP[crop]
        for _ in range(n_per_crop):
            temperature = rng.normal(27, 5)
            humidity = rng.normal(75, 12)
            rainfall = rng.gamma(2, 60)
            leaf_wetness_hours = rng.normal(8, 4)

            temperature = _clip(temperature, 5, 45)
            humidity = _clip(humidity, 20, 100)
            rainfall = _clip(rainfall, 0, 400)
            leaf_wetness_hours = _clip(leaf_wetness_hours, 0, 24)

            # Risk score: high humidity + high leaf wetness + moderate temp -> higher fungal/bacterial risk
            risk_score = (
                0.35 * (humidity / 100)
                + 0.30 * (leaf_wetness_hours / 24)
                + 0.20 * (rainfall / 400)
                + 0.15 * (1 - abs(temperature - 27) / 27)
            )
            risk_score = float(_clip(risk_score + rng.normal(0, 0.05), 0, 1))

            if risk_score < 0.35:
                risk_level = "Low"
                disease = "Healthy"
            elif risk_score < 0.65:
                risk_level = "Medium"
                disease = rng.choice([d for d in diseases if d != "Healthy"])
            else:
                risk_level = "High"
                disease = rng.choice([d for d in diseases if d != "Healthy"])

            rows.append({
                "crop": crop,
                "temperature": round(temperature, 2),
                "humidity": round(humidity, 2),
                "rainfall": round(rainfall, 2),
                "leaf_wetness_hours": round(leaf_wetness_hours, 2),
                "risk_score": round(risk_score, 3),
                "risk_level": risk_level,
                "disease_name": disease,
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. FERTILIZER RECOMMENDATION DATASET
# ----------------------------------------------------------------------
def generate_fertilizer(n_per_crop: int = 230) -> pd.DataFrame:
    rows = []
    for crop in CROPS:
        for _ in range(n_per_crop):
            N = _clip(rng.normal(60, 25), 0, 140)
            P = _clip(rng.normal(45, 20), 0, 145)
            K = _clip(rng.normal(48, 22), 0, 205)
            ph = _clip(rng.normal(6.5, 0.7), 3.5, 9.5)

            n_deficit = max(0, 80 - N)
            p_deficit = max(0, 60 - P)
            k_deficit = max(0, 60 - K)

            if n_deficit >= p_deficit and n_deficit >= k_deficit and n_deficit > 5:
                fert_name, fN, fP, fK = FERTILIZERS[0]
            elif p_deficit >= k_deficit and p_deficit > 5:
                fert_name, fN, fP, fK = FERTILIZERS[1]
            elif k_deficit > 5:
                fert_name, fN, fP, fK = FERTILIZERS[2]
            else:
                fert_name, fN, fP, fK = rng.choice(
                    [FERTILIZERS[3], FERTILIZERS[4]]
                )

            deficit_total = n_deficit + p_deficit + k_deficit
            quantity_kg_per_acre = round(_clip(20 + deficit_total * 0.9 + rng.normal(0, 5), 15, 220), 1)
            cost_per_kg = {
                "Urea": 6.5, "DAP": 27, "MOP": 17, "NPK 19-19-19": 24,
                "NPK 10-26-26": 23, "SSP": 9, "Ammonium Sulphate": 12,
            }[fert_name]
            estimated_cost_inr = round(quantity_kg_per_acre * cost_per_kg, 2)
            organic_alt = rng.choice(ORGANIC_ALTERNATIVES)

            rows.append({
                "crop": crop,
                "N": round(N, 1),
                "P": round(P, 1),
                "K": round(K, 1),
                "ph": round(ph, 2),
                "recommended_fertilizer": fert_name,
                "quantity_kg_per_acre": quantity_kg_per_acre,
                "estimated_cost_inr": estimated_cost_inr,
                "organic_alternative": organic_alt,
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. IRRIGATION DATASET
# ----------------------------------------------------------------------
def generate_irrigation(n_per_crop: int = 230) -> pd.DataFrame:
    water_loving = {"rice", "banana", "watermelon", "muskmelon", "sugarcane", "jute"}
    rows = []
    for crop in CROPS:
        base_need = 6.5 if crop in water_loving else 4.0
        for _ in range(n_per_crop):
            soil_moisture = _clip(rng.normal(45, 15), 5, 95)
            temperature = _clip(rng.normal(28, 5), 8, 45)
            rainfall_forecast = _clip(rng.gamma(2, 25), 0, 250)
            season = rng.choice(SEASONS)

            need_score = (
                base_need
                + (100 - soil_moisture) / 100 * 5
                + (temperature - 25) / 10
                - rainfall_forecast / 60
            )
            need_score = float(_clip(need_score, 0, 15))

            if rainfall_forecast > 40 or need_score < 3:
                irrigation_needed = "No"
                water_liters_per_sqm = 0.0
            else:
                irrigation_needed = "Yes"
                water_liters_per_sqm = round(need_score * 1.8 + rng.normal(0, 1), 2)
                water_liters_per_sqm = float(_clip(water_liters_per_sqm, 1, 40))

            if need_score < 4:
                schedule = "Every 7-10 days"
            elif need_score < 8:
                schedule = "Every 4-6 days"
            else:
                schedule = "Every 1-3 days"

            rows.append({
                "crop": crop,
                "season": season,
                "soil_moisture_pct": round(soil_moisture, 1),
                "temperature": round(temperature, 2),
                "rainfall_forecast_mm": round(rainfall_forecast, 1),
                "irrigation_needed": irrigation_needed,
                "water_liters_per_sqm": water_liters_per_sqm,
                "irrigation_schedule": schedule,
            })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4. FARMER RECORDS DATASET
# ----------------------------------------------------------------------
def generate_farmer_records(n: int = 5200) -> pd.DataFrame:
    rows = []
    for i in range(n):
        crop = rng.choice(CROPS)
        state = rng.choice(STATES)
        season = rng.choice(SEASONS)
        land_size_acres = round(float(_clip(rng.gamma(2, 1.6), 0.25, 25)), 2)
        experience_years = int(_clip(rng.normal(14, 9), 0, 50))
        irrigation_type = rng.choice(["Canal", "Borewell", "Drip", "Rain-fed", "Sprinkler"])
        yield_quintal_per_acre = round(float(_clip(rng.normal(18, 6), 2, 45)), 2)
        market_price_per_quintal = round(float(_clip(rng.normal(2200, 700), 500, 9000)), 2)
        revenue_inr = round(yield_quintal_per_acre * land_size_acres * market_price_per_quintal, 2)

        rows.append({
            "farmer_id": f"F{i+1:05d}",
            "state": state,
            "crop": crop,
            "season": season,
            "land_size_acres": land_size_acres,
            "experience_years": experience_years,
            "irrigation_type": irrigation_type,
            "yield_quintal_per_acre": yield_quintal_per_acre,
            "market_price_per_quintal": market_price_per_quintal,
            "estimated_revenue_inr": revenue_inr,
        })
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    disease_df = generate_disease_risk()
    fert_df = generate_fertilizer()
    irrigation_df = generate_irrigation()
    farmer_df = generate_farmer_records()

    disease_df.to_csv(os.path.join(DATASET_DIR, "disease_risk.csv"), index=False)
    fert_df.to_csv(os.path.join(DATASET_DIR, "fertilizer_recommendation.csv"), index=False)
    irrigation_df.to_csv(os.path.join(DATASET_DIR, "irrigation.csv"), index=False)
    farmer_df.to_csv(os.path.join(DATASET_DIR, "farmer_records.csv"), index=False)

    for name, df in [
        ("disease_risk.csv", disease_df),
        ("fertilizer_recommendation.csv", fert_df),
        ("irrigation.csv", irrigation_df),
        ("farmer_records.csv", farmer_df),
    ]:
        assert df.isnull().sum().sum() == 0, f"{name} has missing values!"
        print(f"{name}: {df.shape[0]} rows, {df.shape[1]} cols, nulls=0")


if __name__ == "__main__":
    main()
