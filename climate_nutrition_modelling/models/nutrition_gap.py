"""
Nutritional Gap Analyzer.

Compares nutrient supply (from crop production) against nutrient demand
(from population requirements) to identify food security gaps.

Follows WHO/FAO recommended nutrient intake standards and the
EAT-Lancet planetary health diet framework.

Data Sources:
- WHO/FAO Vitamin and Mineral Requirements (2004)
  https://www.who.int/publications/i/item/9241546123
- FAO/WHO/UNU Energy and Protein Requirements
- EAT-Lancet Commission on Food, Planet, Health (2019)
  https://eatforum.org/eat-lancet-commission/

Author: Climate-Nutrition Modelling Project
License: MIT
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .nutrient_converter import NutrientConverter


@dataclass
class DailyNutrientRequirement:
    """
    Daily nutrient requirements per person.
    Based on WHO/FAO recommended intakes for adult population average.
    Values represent weighted average across age/sex groups.
    """
    energy_kcal: float = 2100.0
    protein_kg: float = 0.050  # 50g = 0.05kg
    fat_kg: float = 0.065     # 65g
    carbs_kg: float = 0.300   # 300g
    fiber_kg: float = 0.025   # 25g
    iron_g: float = 0.018     # 18mg = 0.018g (women need more)
    zinc_g: float = 0.011     # 11mg
    calcium_g: float = 1.0    # 1000mg = 1g
    vitamin_a_mg_rae: float = 0.0007  # 700µg = 0.7mg
    folate_mg: float = 0.0004  # 400µg = 0.4mg
    vitamin_c_g: float = 0.090  # 90mg = 0.09g


# Pre-defined population nutritional profiles
POPULATION_PROFILES = {
    'canada': DailyNutrientRequirement(
        energy_kcal=2200,  # Higher due to cold climate
        protein_kg=0.056,
        fat_kg=0.078,
        iron_g=0.014,  # Fortified food supply
    ),
    'nigeria': DailyNutrientRequirement(
        energy_kcal=2100,
        protein_kg=0.050,
        fat_kg=0.058,
        iron_g=0.020,  # Higher need (malaria regions)
        zinc_g=0.012,
        vitamin_a_mg_rae=0.0008,  # Higher need (deficiency prevalent)
    ),
    'global_average': DailyNutrientRequirement(),
}


class NutritionalGapAnalyzer:
    """
    Analyzes the gap between nutrient supply and population requirements.

    The gap for each nutrient is:
        Gap = Supply - Demand
        Gap_Ratio = Supply / Demand

    A ratio > 1.0 means sufficient supply.
    A ratio < 1.0 means deficit (food insecurity for that nutrient).

    Usage:
        analyzer = NutritionalGapAnalyzer(population=200_000_000, profile='nigeria')
        analyzer.set_nutrient_supply(converter.get_total_nutrients())
        gaps = analyzer.analyze()
    """

    def __init__(
        self,
        population: float,
        profile: str = 'global_average',
        custom_requirements: Optional[DailyNutrientRequirement] = None,
    ):
        self.population = population
        if custom_requirements:
            self.requirements = custom_requirements
        elif profile in POPULATION_PROFILES:
            self.requirements = POPULATION_PROFILES[profile]
        else:
            self.requirements = DailyNutrientRequirement()

        self.nutrient_supply: Dict[str, float] = {}
        self._tracked_nutrients = [
            'energy_kcal', 'protein_kg', 'fat_kg', 'carbs_kg', 'fiber_kg',
            'iron_g', 'zinc_g', 'calcium_g', 'vitamin_a_mg_rae', 'folate_mg',
            'vitamin_c_g',
        ]

    def set_population(self, population: float):
        """Update population count."""
        self.population = population

    def set_nutrient_supply(self, supply: Dict[str, float]):
        """Set total annual nutrient supply from crops."""
        self.nutrient_supply = supply

    def _annual_requirement(self, nutrient: str) -> float:
        """Calculate total annual nutrient requirement for the population."""
        daily = getattr(self.requirements, nutrient, 0)
        return daily * self.population * 365

    def analyze(self) -> pd.DataFrame:
        """
        Perform nutritional gap analysis.

        Returns DataFrame with columns:
            Nutrient, Annual_Supply, Annual_Demand, Gap, Gap_Ratio, Status
        """
        rows = []

        nutrient_labels = {
            'energy_kcal': ('Energy', 'billion kcal'),
            'protein_kg': ('Protein', 'thousand tonnes'),
            'fat_kg': ('Fat', 'thousand tonnes'),
            'carbs_kg': ('Carbohydrates', 'thousand tonnes'),
            'fiber_kg': ('Dietary Fiber', 'thousand tonnes'),
            'iron_g': ('Iron', 'tonnes'),
            'zinc_g': ('Zinc', 'tonnes'),
            'calcium_g': ('Calcium', 'tonnes'),
            'vitamin_a_mg_rae': ('Vitamin A', 'kg RAE'),
            'folate_mg': ('Folate', 'kg'),
            'vitamin_c_g': ('Vitamin C', 'tonnes'),
        }

        for nutrient in self._tracked_nutrients:
            supply = self.nutrient_supply.get(nutrient, 0)
            demand = self._annual_requirement(nutrient)

            gap = supply - demand
            ratio = supply / demand if demand > 0 else float('inf')

            label, unit = nutrient_labels.get(nutrient, (nutrient, ''))

            # Determine status
            if ratio >= 1.2:
                status = 'Surplus'
            elif ratio >= 1.0:
                status = 'Adequate'
            elif ratio >= 0.8:
                status = 'Mild Deficit'
            elif ratio >= 0.5:
                status = 'Moderate Deficit'
            else:
                status = 'Severe Deficit'

            rows.append({
                'Nutrient': label,
                'Nutrient_Key': nutrient,
                'Unit': unit,
                'Annual_Supply': supply,
                'Annual_Demand': demand,
                'Gap': gap,
                'Gap_Ratio': round(ratio, 3),
                'Status': status,
            })

        return pd.DataFrame(rows)

    def analyze_timeseries(
        self,
        crop_results: Dict[str, pd.DataFrame],
        converter: NutrientConverter,
        population_series: pd.Series,
        years: pd.Series,
    ) -> pd.DataFrame:
        """
        Run gap analysis across a time series.

        Args:
            crop_results: Dict of crop_key → DataFrame with 'Year' and 'Production_tonnes'
            converter: NutrientConverter instance
            population_series: Population for each year
            years: Year values

        Returns:
            DataFrame with Year and gap ratios for each nutrient
        """
        all_rows = []

        for i, year in enumerate(years):
            # Convert all crops for this year
            year_supply = {n: 0.0 for n in self._tracked_nutrients}

            for crop_key, crop_df in crop_results.items():
                year_row = crop_df[crop_df['Year'] == year]
                if len(year_row) == 0:
                    continue

                prod = year_row['Production_tonnes'].values[0]
                nutrients = converter.convert_crop(crop_key, prod)

                for n in self._tracked_nutrients:
                    year_supply[n] += nutrients.get(n, 0)

            # Calculate demand
            pop = population_series.iloc[i] if i < len(population_series) else population_series.iloc[-1]
            self.set_population(pop)
            self.set_nutrient_supply(year_supply)

            row = {'Year': year, 'Population': pop}
            analysis = self.analyze()
            for _, arow in analysis.iterrows():
                row[f"{arow['Nutrient_Key']}_ratio"] = arow['Gap_Ratio']
                row[f"{arow['Nutrient_Key']}_status"] = arow['Status']

            all_rows.append(row)

        return pd.DataFrame(all_rows)

    def get_priority_interventions(self) -> List[Dict]:
        """
        Identify which nutrients need intervention most urgently.

        Returns list of interventions sorted by severity.
        """
        analysis = self.analyze()
        deficits = analysis[analysis['Gap_Ratio'] < 1.0].copy()
        deficits = deficits.sort_values('Gap_Ratio')

        interventions = []
        # Map nutrients to best crop sources
        best_sources = {
            'energy_kcal': ['maize_grain', 'cassava', 'rice_paddy', 'sorghum'],
            'protein_kg': ['cowpea', 'soybean', 'groundnut', 'millet'],
            'fat_kg': ['groundnut', 'soybean'],
            'iron_g': ['leafy_greens', 'cowpea', 'millet', 'soybean'],
            'zinc_g': ['cowpea', 'groundnut', 'millet', 'leafy_greens'],
            'calcium_g': ['leafy_greens', 'soybean', 'cowpea', 'okra'],
            'vitamin_a_mg_rae': ['sweet_potato', 'leafy_greens', 'pepper', 'plantain'],
            'folate_mg': ['cowpea', 'leafy_greens', 'soybean', 'groundnut'],
            'vitamin_c_g': ['pepper', 'leafy_greens', 'tomato', 'okra'],
        }

        for _, row in deficits.iterrows():
            key = row['Nutrient_Key']
            interventions.append({
                'nutrient': row['Nutrient'],
                'current_ratio': row['Gap_Ratio'],
                'status': row['Status'],
                'tonnes_deficit': abs(row['Gap']),
                'recommended_crops': best_sources.get(key, ['multiple sources']),
                'priority': 'HIGH' if row['Gap_Ratio'] < 0.5 else
                           'MEDIUM' if row['Gap_Ratio'] < 0.8 else 'LOW',
            })

        return interventions
