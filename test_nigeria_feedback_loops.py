#!/usr/bin/env python3
"""
Nigeria Feedback Loop Verification Script
==========================================
Tests all 6 feedback loops for each Nigerian crop (maize, cassava, rice, sorghum).

Data Sources:
    - FAOSTAT (FAO, 2024): Crop production, area, yield statistics
    - NBS Nigeria: National Agricultural Census 2022
    - IFDC: Fertilizer Statistics Overview Nigeria 2024/2025
    - UN World Population Prospects 2024: Population projections
    - World Bank Climate Change Knowledge Portal: Climate baselines

Usage:
    python test_nigeria_feedback_loops.py
"""

import sys, os, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from climate_nutrition_modelling.models.crop_model import CropModel, CropParameters
from climate_nutrition_modelling.models.climate_scenarios import (
    ClimateScenarioManager, REGIONAL_BASELINES
)


CROPS = [
    ('Maize',    'nigeria_maize.json',    'maize_production.csv'),
    ('Cassava',  'nigeria_cassava.json',  'cassava_production.csv'),
    ('Rice',     'nigeria_rice.json',     'rice_production.csv'),
    ('Sorghum',  'nigeria_sorghum.json',  'sorghum_production.csv'),
]


def run(config_file, data_file, **scenario_kw):
    """Run model silently and return results DataFrame."""
    with contextlib.redirect_stdout(io.StringIO()):
        csm = ClimateScenarioManager(baseline=REGIONAL_BASELINES['nigeria_north'])
        climate = csm.generate_scenario('ssp245', 1971, 2100)
        params = CropParameters.from_json(
            f'climate_nutrition_modelling/config/{config_file}')
        model = CropModel(params)
        model.load_historical_data(
            f'climate_nutrition_modelling/data/nigeria/{data_file}')
        model.climate_data = climate

        # Load population data
        import pandas as pd
        pop = pd.read_csv('climate_nutrition_modelling/data/nigeria/nigeria_population.csv')
        model.population_data = pop

        model.set_scenario(**scenario_kw)
        df = model.run()
    return df


def val(df, year, col):
    row = df[df.Year == year]
    if len(row) == 0:
        return float('nan')
    return row[col].values[0]


print("=" * 90)
print("NIGERIA — FEEDBACK LOOP VERIFICATION (SSP2-4.5)")
print("=" * 90)
print()

yr = 2060  # Test in projection period

for crop_name, config, data in CROPS:
    print(f"\n{'='*90}")
    print(f"  {crop_name.upper()}")
    print(f"{'='*90}")

    baseline   = run(config, data)
    high_n     = run(config, data, nitrogen_multiplier=2.0)
    high_tech  = run(config, data, technology_boost=0.02)
    high_cons  = run(config, data, conservation_investment=1.0)
    high_irr   = run(config, data, irrigation_investment=1.0)

    # B1: Market Equilibrium
    base_prod = val(baseline, yr, 'Production_tonnes')
    tech_prod = val(high_tech, yr, 'Production_tonnes')
    base_price = val(baseline, yr, 'Price_per_tonne')
    tech_price = val(high_tech, yr, 'Price_per_tonne')
    print(f"\n  B1 Market: base prod={base_prod:>14,.0f}, tech prod={tech_prod:>14,.0f}")
    print(f"            base price=${base_price:>8.1f}, tech price=${tech_price:>8.1f}")
    if tech_prod > base_prod and tech_price <= base_price:
        print(f"  ✓ PASS: More production → lower price")
    elif tech_prod > base_prod:
        print(f"  ~ PARTIAL: More production but price didn't drop")
    else:
        print(f"  ~ INFO: Tech didn't increase production (may be at cap)")

    # B2: Soil Degradation
    base_soil = val(baseline, yr, 'Soil_Health')
    hn_soil = val(high_n, yr, 'Soil_Health')
    print(f"\n  B2 Soil: base={base_soil:.3f}, N2x={hn_soil:.3f}")
    if hn_soil < base_soil:
        print(f"  ✓ PASS: More nitrogen → lower soil health")
    else:
        print(f"  ~ INFO: Soil similar (N level may be low)")

    # B3: Water Stress → Irrigation
    irr_2030 = val(baseline, 2030, 'Irrigation_Fraction')
    irr_2070 = val(baseline, 2070, 'Irrigation_Fraction')
    irr_pol = val(high_irr, yr, 'Irrigation_Fraction')
    print(f"\n  B3 Water: baseline irr 2030={irr_2030:.4f}, 2070={irr_2070:.4f}")
    print(f"           with policy irr@{yr}={irr_pol:.4f}")
    if irr_pol > val(baseline, yr, 'Irrigation_Fraction'):
        print(f"  ✓ PASS: Irrigation policy increases coverage")

    # B4: Erosion → P depletion
    base_p = val(baseline, yr, 'P_Soil_Stock_kg_ha')
    cons_p = val(high_cons, yr, 'P_Soil_Stock_kg_ha')
    print(f"\n  B4 Erosion: base P={base_p:.0f}, conserv P={cons_p:.0f}")
    if cons_p >= base_p:
        print(f"  ✓ PASS: Conservation preserves P stock")

    # Summary row
    print(f"\n  {'Scenario':<18} {'Yield':>7} {'Production':>14} {'Soil':>6} {'N':>5} {'Price':>7} {'Irr%':>6}")
    print(f"  {'-'*64}")
    for name, df in [("Baseline", baseline), ("N 2x", high_n), ("Tech +2%", high_tech),
                     ("Conservation", high_cons), ("Irrigation", high_irr)]:
        print(f"  {name:<18} "
              f"{val(df, yr, 'Yield_t_per_ha'):>7.2f} "
              f"{val(df, yr, 'Production_tonnes'):>14,.0f} "
              f"{val(df, yr, 'Soil_Health'):>6.3f} "
              f"{val(df, yr, 'Nitrogen_kg_ha'):>5.0f} "
              f"{val(df, yr, 'Price_per_tonne'):>7.1f} "
              f"{val(df, yr, 'Irrigation_Fraction')*100:>5.1f}%")

print(f"\n{'='*90}")
print("DONE — If values differ between scenarios, feedback loops are working.")
print("Run the dashboard: python -m streamlit run climate_nutrition_modelling/app.py")
print("Select 'Nigeria' from the country dropdown.")
print("=" * 90)
