"""
crop_info.py
------------
Static reference data used to enrich model predictions with human-readable
context: crop descriptions, typical yield ranges, and disease
prevention/treatment guidance. This is domain reference content, not
learned from data, and is clearly presented as general guidance in the UI.
"""

CROP_DETAILS = {
    "rice": {"category": "Cereal", "expected_yield_range": "25-40 quintal/acre", "growing_season": "Kharif", "water_need": "High"},
    "maize": {"category": "Cereal", "expected_yield_range": "20-30 quintal/acre", "growing_season": "Kharif/Rabi", "water_need": "Medium"},
    "chickpea": {"category": "Pulse", "expected_yield_range": "8-12 quintal/acre", "growing_season": "Rabi", "water_need": "Low"},
    "kidneybeans": {"category": "Pulse", "expected_yield_range": "6-10 quintal/acre", "growing_season": "Kharif", "water_need": "Medium"},
    "pigeonpeas": {"category": "Pulse", "expected_yield_range": "6-9 quintal/acre", "growing_season": "Kharif", "water_need": "Low"},
    "mothbeans": {"category": "Pulse", "expected_yield_range": "3-6 quintal/acre", "growing_season": "Kharif", "water_need": "Very Low"},
    "mungbean": {"category": "Pulse", "expected_yield_range": "4-7 quintal/acre", "growing_season": "Kharif/Zaid", "water_need": "Low"},
    "blackgram": {"category": "Pulse", "expected_yield_range": "4-7 quintal/acre", "growing_season": "Kharif", "water_need": "Low"},
    "lentil": {"category": "Pulse", "expected_yield_range": "6-9 quintal/acre", "growing_season": "Rabi", "water_need": "Low"},
    "pomegranate": {"category": "Fruit", "expected_yield_range": "60-100 quintal/acre", "growing_season": "Year-round", "water_need": "Medium"},
    "banana": {"category": "Fruit", "expected_yield_range": "300-450 quintal/acre", "growing_season": "Year-round", "water_need": "High"},
    "mango": {"category": "Fruit", "expected_yield_range": "40-70 quintal/acre", "growing_season": "Year-round", "water_need": "Medium"},
    "grapes": {"category": "Fruit", "expected_yield_range": "80-120 quintal/acre", "growing_season": "Rabi", "water_need": "Medium"},
    "watermelon": {"category": "Fruit", "expected_yield_range": "150-250 quintal/acre", "growing_season": "Zaid", "water_need": "High"},
    "muskmelon": {"category": "Fruit", "expected_yield_range": "100-150 quintal/acre", "growing_season": "Zaid", "water_need": "High"},
    "apple": {"category": "Fruit", "expected_yield_range": "50-80 quintal/acre", "growing_season": "Temperate/Year-round", "water_need": "Medium"},
    "orange": {"category": "Fruit", "expected_yield_range": "60-90 quintal/acre", "growing_season": "Year-round", "water_need": "Medium"},
    "papaya": {"category": "Fruit", "expected_yield_range": "300-500 quintal/acre", "growing_season": "Year-round", "water_need": "Medium"},
    "coconut": {"category": "Plantation", "expected_yield_range": "80-120 nuts/tree/year", "growing_season": "Year-round", "water_need": "High"},
    "cotton": {"category": "Fiber", "expected_yield_range": "6-10 quintal/acre", "growing_season": "Kharif", "water_need": "Medium"},
    "jute": {"category": "Fiber", "expected_yield_range": "8-12 quintal/acre", "growing_season": "Kharif", "water_need": "High"},
    "coffee": {"category": "Plantation", "expected_yield_range": "4-8 quintal/acre", "growing_season": "Year-round", "water_need": "Medium"},
}

# Generic, non-brand-specific prevention/treatment guidance keyed by disease name.
# Provided as general agronomic reference information, not medical/legal advice.
DISEASE_GUIDANCE = {
    "default": {
        "prevention": [
            "Use certified, disease-resistant seed varieties.",
            "Maintain proper field drainage and avoid waterlogging.",
            "Practice crop rotation to break pest/disease cycles.",
            "Ensure adequate plant spacing for airflow.",
        ],
        "organic_treatment": ["Neem oil spray", "Trichoderma-based bio-fungicide", "Remove and destroy infected plant debris"],
        "chemical_treatment": ["Consult local agricultural extension office for a registered fungicide/bactericide suited to the confirmed disease"],
    },
    "Healthy": {
        "prevention": ["Continue current practices.", "Monitor weekly for early signs of stress."],
        "organic_treatment": ["No treatment needed."],
        "chemical_treatment": ["No treatment needed."],
    },
    "Rice Blast": {
        "prevention": ["Avoid excess nitrogen application.", "Use resistant varieties.", "Maintain balanced field water levels."],
        "organic_treatment": ["Neem-based foliar spray", "Silicon soil amendment"],
        "chemical_treatment": ["Tricyclazole-based fungicide (as per local agri-extension guidance)"],
    },
    "Bacterial Leaf Blight": {
        "prevention": ["Use disease-free seed.", "Avoid clipping seedlings before transplanting.", "Balanced N-P-K application."],
        "organic_treatment": ["Copper-based bio-bactericide", "Field sanitation"],
        "chemical_treatment": ["Copper oxychloride-based bactericide"],
    },
    "Powdery Mildew": {
        "prevention": ["Improve air circulation via pruning.", "Avoid overhead irrigation.", "Sunlight exposure to canopy."],
        "organic_treatment": ["Sulfur dust", "Diluted milk spray", "Neem oil"],
        "chemical_treatment": ["Sulfur or triazole-based fungicide"],
    },
    "Downy Mildew": {
        "prevention": ["Avoid dense planting.", "Improve drainage.", "Remove infected leaves promptly."],
        "organic_treatment": ["Copper-based bio-fungicide", "Baking soda spray"],
        "chemical_treatment": ["Metalaxyl or mancozeb-based fungicide"],
    },
}


def get_disease_guidance(disease_name: str) -> dict:
    return DISEASE_GUIDANCE.get(disease_name, DISEASE_GUIDANCE["default"])


def get_crop_details(crop_name: str) -> dict:
    return CROP_DETAILS.get(
        crop_name,
        {"category": "General", "expected_yield_range": "Data not available", "growing_season": "Varies", "water_need": "Medium"},
    )
