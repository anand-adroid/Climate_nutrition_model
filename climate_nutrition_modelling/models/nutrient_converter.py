"""
Nutrient Converter Module.

Converts crop production (tonnes) into nutritional content using
FAO/INFOODS food composition data and USDA FoodData Central values.

This is the bridge between agricultural output and human nutrition.

Data Sources:
- FAO/INFOODS Food Composition Tables (2022)
  https://www.fao.org/infoods/infoods/tables-and-databases/en/
- USDA FoodData Central
  https://fdc.nal.usda.gov/
- West African Food Composition Table (FAO, 2019)
  https://www.fao.org/3/ca2698en/ca2698en.pdf

Author: Climate-Nutrition Modelling Project
License: MIT
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class NutrientProfile:
    """
    Nutritional content per 1000 kg (1 tonne) of raw crop.
    Values account for average moisture content and edible portion.

    Units:
    - energy_kcal: kilocalories
    - protein_kg, fat_kg, carbs_kg, fiber_kg: kilograms
    - iron_g, zinc_g, calcium_g: grams
    - vitamin_a_mg_rae: milligrams Retinol Activity Equivalents
    - folate_mg: milligrams
    - vitamin_c_g: grams
    """
    crop_name: str
    energy_kcal: float = 0.0
    protein_kg: float = 0.0
    fat_kg: float = 0.0
    carbs_kg: float = 0.0
    fiber_kg: float = 0.0
    iron_g: float = 0.0
    zinc_g: float = 0.0
    calcium_g: float = 0.0
    vitamin_a_mg_rae: float = 0.0
    folate_mg: float = 0.0
    vitamin_c_g: float = 0.0

    # Post-harvest loss fraction (0 to 1)
    post_harvest_loss: float = 0.0

    # Fraction used for human food (rest is feed, seed, industrial)
    food_use_fraction: float = 1.0


# ============================================================
# FAO/INFOODS + USDA NUTRIENT PROFILES PER TONNE OF RAW CROP
# ============================================================
# Values are per 1000 kg of raw crop as harvested.
# Moisture content is accounted for in the per-tonne values.

CROP_NUTRIENT_PROFILES = {
    # --- CEREALS ---
    'maize_grain': NutrientProfile(
        crop_name='Maize (Corn) Grain',
        energy_kcal=3_650_000,  # ~365 kcal/100g dry
        protein_kg=94.0,       # ~9.4%
        fat_kg=47.0,           # ~4.7%
        carbs_kg=743.0,        # ~74.3%
        fiber_kg=73.0,         # ~7.3%
        iron_g=2_700,          # ~2.7 mg/100g
        zinc_g=2_200,          # ~2.2 mg/100g
        calcium_g=7_000,       # ~7 mg/100g
        vitamin_a_mg_rae=214,  # ~214 µg RAE/100g (yellow maize)
        folate_mg=190,         # ~19 µg/100g
        vitamin_c_g=0,         # negligible
        post_harvest_loss=0.10,
        food_use_fraction=0.40,  # ~40% direct food, rest is feed/industrial
    ),
    'rice_paddy': NutrientProfile(
        crop_name='Rice (Paddy)',
        energy_kcal=3_600_000,
        protein_kg=75.0,
        fat_kg=6.0,
        carbs_kg=800.0,
        fiber_kg=13.0,
        iron_g=800,
        zinc_g=1_100,
        calcium_g=10_000,
        vitamin_a_mg_rae=0,
        folate_mg=80,
        vitamin_c_g=0,
        post_harvest_loss=0.08,
        food_use_fraction=0.85,
    ),
    'sorghum': NutrientProfile(
        crop_name='Sorghum',
        energy_kcal=3_390_000,
        protein_kg=112.0,
        fat_kg=33.0,
        carbs_kg=729.0,
        fiber_kg=68.0,
        iron_g=4_400,
        zinc_g=1_600,
        calcium_g=28_000,
        vitamin_a_mg_rae=0,
        folate_mg=200,
        vitamin_c_g=0,
        post_harvest_loss=0.12,
        food_use_fraction=0.70,
    ),
    'millet': NutrientProfile(
        crop_name='Pearl Millet',
        energy_kcal=3_780_000,
        protein_kg=110.0,
        fat_kg=42.0,
        carbs_kg=730.0,
        fiber_kg=85.0,
        iron_g=8_000,
        zinc_g=3_100,
        calcium_g=42_000,
        vitamin_a_mg_rae=0,
        folate_mg=850,
        vitamin_c_g=0,
        post_harvest_loss=0.15,
        food_use_fraction=0.80,
    ),
    'wheat': NutrientProfile(
        crop_name='Wheat',
        energy_kcal=3_270_000,
        protein_kg=127.0,
        fat_kg=17.0,
        carbs_kg=714.0,
        fiber_kg=122.0,
        iron_g=3_200,
        zinc_g=2_500,
        calcium_g=29_000,
        vitamin_a_mg_rae=0,
        folate_mg=380,
        vitamin_c_g=0,
        post_harvest_loss=0.05,
        food_use_fraction=0.75,
    ),

    # --- ROOT CROPS ---
    'cassava': NutrientProfile(
        crop_name='Cassava (Fresh)',
        energy_kcal=1_600_000,  # High moisture, ~160 kcal/100g
        protein_kg=14.0,
        fat_kg=3.0,
        carbs_kg=380.0,
        fiber_kg=18.0,
        iron_g=2_700,
        zinc_g=3_400,
        calcium_g=16_000,
        vitamin_a_mg_rae=13,   # white cassava, very low
        folate_mg=270,
        vitamin_c_g=20_600,    # ~20.6 mg/100g
        post_harvest_loss=0.25,  # High — perishable
        food_use_fraction=0.70,
    ),
    'yam': NutrientProfile(
        crop_name='Yam (Fresh)',
        energy_kcal=1_180_000,
        protein_kg=15.0,
        fat_kg=2.0,
        carbs_kg=278.0,
        fiber_kg=41.0,
        iron_g=5_400,
        zinc_g=2_400,
        calcium_g=17_000,
        vitamin_a_mg_rae=83,
        folate_mg=230,
        vitamin_c_g=17_100,
        post_harvest_loss=0.22,
        food_use_fraction=0.85,
    ),
    'sweet_potato': NutrientProfile(
        crop_name='Sweet Potato (Orange-fleshed)',
        energy_kcal=860_000,
        protein_kg=16.0,
        fat_kg=1.0,
        carbs_kg=201.0,
        fiber_kg=30.0,
        iron_g=6_100,
        zinc_g=3_000,
        calcium_g=30_000,
        vitamin_a_mg_rae=7_090,  # Very high in orange-fleshed
        folate_mg=110,
        vitamin_c_g=24_000,
        post_harvest_loss=0.20,
        food_use_fraction=0.90,
    ),

    # --- LEGUMES ---
    'cowpea': NutrientProfile(
        crop_name='Cowpea (Black-eyed Pea)',
        energy_kcal=3_360_000,
        protein_kg=236.0,   # Very high protein
        fat_kg=15.0,
        carbs_kg=604.0,
        fiber_kg=108.0,
        iron_g=8_300,
        zinc_g=3_400,
        calcium_g=110_000,
        vitamin_a_mg_rae=13,
        folate_mg=6_330,   # Extremely high in folate
        vitamin_c_g=1_500,
        post_harvest_loss=0.08,
        food_use_fraction=0.90,
    ),
    'groundnut': NutrientProfile(
        crop_name='Groundnut (Peanut)',
        energy_kcal=5_670_000,
        protein_kg=258.0,
        fat_kg=492.0,
        carbs_kg=161.0,
        fiber_kg=85.0,
        iron_g=4_600,
        zinc_g=3_300,
        calcium_g=92_000,
        vitamin_a_mg_rae=0,
        folate_mg=2_400,
        vitamin_c_g=0,
        post_harvest_loss=0.06,
        food_use_fraction=0.60,
    ),
    'soybean': NutrientProfile(
        crop_name='Soybean',
        energy_kcal=4_460_000,
        protein_kg=364.0,
        fat_kg=199.0,
        carbs_kg=301.0,
        fiber_kg=92.0,
        iron_g=15_700,
        zinc_g=4_900,
        calcium_g=277_000,
        vitamin_a_mg_rae=10,
        folate_mg=3_750,
        vitamin_c_g=6_000,
        post_harvest_loss=0.05,
        food_use_fraction=0.30,
    ),

    # --- VEGETABLES ---
    'tomato': NutrientProfile(
        crop_name='Tomato',
        energy_kcal=180_000,
        protein_kg=9.0,
        fat_kg=2.0,
        carbs_kg=39.0,
        fiber_kg=12.0,
        iron_g=2_700,
        zinc_g=1_700,
        calcium_g=10_000,
        vitamin_a_mg_rae=4_200,
        folate_mg=150,
        vitamin_c_g=138_000,  # Very high vitamin C
        post_harvest_loss=0.30,  # Very perishable
        food_use_fraction=0.95,
    ),
    'leafy_greens': NutrientProfile(
        crop_name='Leafy Greens (Amaranth/Spinach)',
        energy_kcal=230_000,
        protein_kg=28.0,
        fat_kg=4.0,
        carbs_kg=36.0,
        fiber_kg=22.0,
        iron_g=27_000,   # Very high iron
        zinc_g=9_000,
        calcium_g=215_000,  # Very high calcium
        vitamin_a_mg_rae=52_920,  # Extremely high vitamin A
        folate_mg=1_460,
        vitamin_c_g=281_000,
        post_harvest_loss=0.35,  # Very perishable
        food_use_fraction=0.95,
    ),
    'pepper': NutrientProfile(
        crop_name='Pepper (Chili/Bell)',
        energy_kcal=400_000,
        protein_kg=19.0,
        fat_kg=4.0,
        carbs_kg=90.0,
        fiber_kg=15.0,
        iron_g=10_000,
        zinc_g=2_500,
        calcium_g=14_000,
        vitamin_a_mg_rae=18_000,
        folate_mg=460,
        vitamin_c_g=1_440_000,  # Extremely high vitamin C
        post_harvest_loss=0.25,
        food_use_fraction=0.95,
    ),
    'okra': NutrientProfile(
        crop_name='Okra',
        energy_kcal=330_000,
        protein_kg=19.0,
        fat_kg=2.0,
        carbs_kg=72.0,
        fiber_kg=32.0,
        iron_g=6_200,
        zinc_g=5_800,
        calcium_g=82_000,
        vitamin_a_mg_rae=3_600,
        folate_mg=600,
        vitamin_c_g=232_000,
        post_harvest_loss=0.28,
        food_use_fraction=0.95,
    ),

    # --- FRUITS ---
    'plantain': NutrientProfile(
        crop_name='Plantain',
        energy_kcal=1_220_000,
        protein_kg=13.0,
        fat_kg=4.0,
        carbs_kg=318.0,
        fiber_kg=23.0,
        iron_g=6_000,
        zinc_g=1_400,
        calcium_g=3_000,
        vitamin_a_mg_rae=11_280,
        folate_mg=220,
        vitamin_c_g=184_000,
        post_harvest_loss=0.30,
        food_use_fraction=0.90,
    ),
}


class NutrientConverter:
    """
    Converts crop production data into nutritional availability.

    Takes production in tonnes for each crop and calculates total
    nutrient availability after accounting for:
    1. Post-harvest losses
    2. Food vs non-food use allocation
    3. Nutrient content per tonne

    Usage:
        converter = NutrientConverter()
        converter.add_crop('maize_grain', production_tonnes=15_000_000)
        converter.add_crop('cassava', production_tonnes=60_000_000)
        nutrients = converter.get_total_nutrients()
    """

    TRACKED_NUTRIENTS = [
        'energy_kcal', 'protein_kg', 'fat_kg', 'carbs_kg', 'fiber_kg',
        'iron_g', 'zinc_g', 'calcium_g', 'vitamin_a_mg_rae', 'folate_mg',
        'vitamin_c_g',
    ]

    def __init__(self):
        self.crop_contributions: Dict[str, Dict[str, float]] = {}
        self.profiles = CROP_NUTRIENT_PROFILES.copy()

    def add_custom_profile(self, key: str, profile: NutrientProfile):
        """Add or override a nutrient profile for a crop."""
        self.profiles[key] = profile

    def convert_crop(
        self,
        crop_key: str,
        production_tonnes: float,
        food_fraction_override: Optional[float] = None,
        loss_fraction_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Convert crop production to nutrients available for human consumption.

        Args:
            crop_key: Key into CROP_NUTRIENT_PROFILES
            production_tonnes: Total production in tonnes
            food_fraction_override: Override the default food use fraction
            loss_fraction_override: Override post-harvest loss fraction

        Returns:
            Dict of nutrient name → total amount available
        """
        if crop_key not in self.profiles:
            raise ValueError(f"Unknown crop: {crop_key}. "
                           f"Available: {list(self.profiles.keys())}")

        profile = self.profiles[crop_key]
        loss = loss_fraction_override if loss_fraction_override is not None else profile.post_harvest_loss
        food_frac = food_fraction_override if food_fraction_override is not None else profile.food_use_fraction

        # Effective tonnes for human food
        effective_tonnes = production_tonnes * (1 - loss) * food_frac

        nutrients = {}
        for nutrient in self.TRACKED_NUTRIENTS:
            per_tonne = getattr(profile, nutrient, 0.0)
            nutrients[nutrient] = effective_tonnes * per_tonne

        nutrients['effective_food_tonnes'] = effective_tonnes
        nutrients['total_production_tonnes'] = production_tonnes
        nutrients['crop_name'] = profile.crop_name

        self.crop_contributions[crop_key] = nutrients
        return nutrients

    def convert_production_timeseries(
        self,
        crop_key: str,
        production_series: pd.Series,
        years: pd.Series,
    ) -> pd.DataFrame:
        """
        Convert a time series of crop production to nutrient availability.

        Args:
            crop_key: Key into CROP_NUTRIENT_PROFILES
            production_series: Annual production in tonnes
            years: Corresponding years

        Returns:
            DataFrame with Year and all nutrient columns
        """
        if crop_key not in self.profiles:
            raise ValueError(f"Unknown crop: {crop_key}")

        profile = self.profiles[crop_key]
        effective = production_series * (1 - profile.post_harvest_loss) * profile.food_use_fraction

        result = pd.DataFrame({'Year': years})
        result['Effective_Food_Tonnes'] = effective.values

        for nutrient in self.TRACKED_NUTRIENTS:
            per_tonne = getattr(profile, nutrient, 0.0)
            result[nutrient] = effective.values * per_tonne

        result['Crop'] = crop_key
        return result

    def get_total_nutrients(self) -> Dict[str, float]:
        """Get total nutrients from all crops added so far."""
        totals = {n: 0.0 for n in self.TRACKED_NUTRIENTS}
        totals['total_food_tonnes'] = 0.0

        for crop_key, nutrients in self.crop_contributions.items():
            for n in self.TRACKED_NUTRIENTS:
                totals[n] += nutrients.get(n, 0.0)
            totals['total_food_tonnes'] += nutrients.get('effective_food_tonnes', 0.0)

        return totals

    def get_contribution_table(self) -> pd.DataFrame:
        """Get a breakdown table showing each crop's contribution."""
        rows = []
        for crop_key, nutrients in self.crop_contributions.items():
            row = {'Crop': nutrients.get('crop_name', crop_key)}
            row['Production_tonnes'] = nutrients.get('total_production_tonnes', 0)
            row['Food_tonnes'] = nutrients.get('effective_food_tonnes', 0)
            for n in self.TRACKED_NUTRIENTS:
                row[n] = nutrients.get(n, 0)
            rows.append(row)

        return pd.DataFrame(rows)
