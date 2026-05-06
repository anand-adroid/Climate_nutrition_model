"""
Climate-Nutrition Modelling - Core Models

Modules:
    crop_model: Generic system dynamics crop production model
    climate_scenarios: IPCC SSP climate projection generator
    nutrient_converter: Crop production → nutrient availability converter
    nutrition_gap: Nutritional gap analysis (supply vs. demand)
"""

from .crop_model import CropModel, CropParameters
from .climate_scenarios import ClimateScenarioManager, RegionalClimateBaseline, REGIONAL_BASELINES
from .nutrient_converter import NutrientConverter, CROP_NUTRIENT_PROFILES, NutrientProfile
from .nutrition_gap import NutritionalGapAnalyzer, POPULATION_PROFILES, DailyNutrientRequirement

__all__ = [
    'CropModel', 'CropParameters',
    'ClimateScenarioManager', 'RegionalClimateBaseline', 'REGIONAL_BASELINES',
    'NutrientConverter', 'CROP_NUTRIENT_PROFILES', 'NutrientProfile',
    'NutritionalGapAnalyzer', 'POPULATION_PROFILES', 'DailyNutrientRequirement',
]
