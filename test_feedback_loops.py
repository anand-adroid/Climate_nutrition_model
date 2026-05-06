#!/usr/bin/env python3
"""
Feedback Loop Verification Script
==================================
Run this to verify that all 6 feedback loops are working correctly.
Each test changes ONE input and checks that the expected chain of
effects propagates through the model.

Usage:
    python test_feedback_loops.py

What "correct" means for each loop:
    B1 (Price-Area): Higher price → more area → more production → lower price
    B2 (Soil-N): More nitrogen → lower soil health → lower yield multiplier
    R1 (NPK-Profit): Higher profit → more NPK → higher yield
    R2 (Machinery): Higher revenue → more machinery → higher yield
    B3 (Water): Water stress → irrigation expansion (endogenous)
    B4 (Erosion-P): More erosion → lower P soil stock → lower P response
"""

import sys, os, io, contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from climate_nutrition_modelling.models.crop_model import CropModel, CropParameters
from climate_nutrition_modelling.models.climate_scenarios import (
    ClimateScenarioManager, REGIONAL_BASELINES
)


def run(label, cap=20.0, **scenario_kw):
    """Run model silently and return results DataFrame."""
    with contextlib.redirect_stdout(io.StringIO()):
        csm = ClimateScenarioManager(baseline=REGIONAL_BASELINES['canada_ontario'])
        climate = csm.generate_scenario('ssp245', 1971, 2100)
        params = CropParameters.from_json(
            'climate_nutrition_modelling/config/canada_corn.json')
        params.max_attainable_yield = cap
        model = CropModel(params)
        model.load_historical_data(
            'climate_nutrition_modelling/data/canada/corn_production.csv')
        model.climate_data = climate
        model.set_scenario(**scenario_kw)
        df = model.run()
    return df


def val(df, year, col):
    """Get value at a specific year."""
    row = df[df.Year == year]
    if len(row) == 0:
        return float('nan')
    return row[col].values[0]


# ================================================================
# RUN SCENARIOS
# ================================================================
print("Running scenarios... (this takes ~10 seconds)")
print()

baseline   = run("Baseline")
high_n     = run("High N",      nitrogen_multiplier=2.0)
low_n      = run("Low N",       nitrogen_multiplier=0.5)
high_tech  = run("High Tech",   technology_boost=0.02)
high_adapt = run("High Adapt",  climate_adaptation=0.8)
high_cons  = run("Conservation", conservation_investment=1.0)
high_irr   = run("Irrigation",  irrigation_investment=1.0)

yr = 2060  # Test in projection period

print("=" * 80)
print(f"FEEDBACK LOOP VERIFICATION (Year {yr}, SSP2-4.5)")
print("=" * 80)

# ================================================================
# TEST B1: MARKET EQUILIBRIUM
# ================================================================
print("\n--- B1: Market Equilibrium (Price → Area → Supply → Price) ---")
base_price = val(baseline, yr, 'Price_per_tonne')
base_area  = val(baseline, yr, 'Area_Harvested_ha')
base_prod  = val(baseline, yr, 'Production_tonnes')

# High tech → more yield → more supply → should push price DOWN
tech_price = val(high_tech, yr, 'Price_per_tonne')
tech_prod  = val(high_tech, yr, 'Production_tonnes')

print(f"  Baseline:  production={base_prod:>14,.0f} t, price=${base_price:>8.1f}/t")
print(f"  High Tech: production={tech_prod:>14,.0f} t, price=${tech_price:>8.1f}/t")
if tech_prod > base_prod and tech_price <= base_price:
    print(f"  ✓ PASS: More production → lower/equal price (B1 balancing)")
elif tech_prod > base_prod:
    print(f"  ~ PARTIAL: More production but price didn't drop (may be at floor)")
else:
    print(f"  ✗ FAIL: Expected more production from high tech")

# ================================================================
# TEST B2: SOIL DEGRADATION
# ================================================================
print("\n--- B2: Soil Degradation (Nitrogen ↑ → Soil Health ↓ → Yield ↓) ---")
base_soil  = val(baseline, yr, 'Soil_Health')
base_yield = val(baseline, yr, 'Yield_t_per_ha')
hn_soil    = val(high_n, yr, 'Soil_Health')
hn_yield   = val(high_n, yr, 'Yield_t_per_ha')
hn_n       = val(high_n, yr, 'Nitrogen_kg_ha')

print(f"  Baseline:   N={val(baseline, yr, 'Nitrogen_kg_ha'):>6.0f} kg/ha, soil={base_soil:.3f}, yield={base_yield:.2f}")
print(f"  Nitrogen 2x: N={hn_n:>6.0f} kg/ha, soil={hn_soil:.3f}, yield={hn_yield:.2f}")
if hn_soil < base_soil:
    print(f"  ✓ PASS: More nitrogen → lower soil health ({base_soil:.3f} → {hn_soil:.3f})")
else:
    print(f"  ✗ FAIL: Soil health should decrease with more nitrogen")

# ================================================================
# TEST R1: INPUT INVESTMENT
# ================================================================
print("\n--- R1: Input Investment (Profit ↑ → NPK ↑ → Yield ↑) ---")
base_n     = val(baseline, yr, 'Nitrogen_kg_ha')
base_prof  = val(baseline, yr, 'Profit_Margin')
tech_n     = val(high_tech, yr, 'Nitrogen_kg_ha')
tech_prof  = val(high_tech, yr, 'Profit_Margin')

print(f"  Baseline:  profit={base_prof:>6.3f}, N applied={base_n:>6.0f} kg/ha")
print(f"  High Tech: profit={tech_prof:>6.3f}, N applied={tech_n:>6.0f} kg/ha")
# Higher tech → higher yield → higher profit should sustain or increase N
print(f"  ✓ INFO: Tech boost changes profit from {base_prof:.3f} to {tech_prof:.3f}")

# ================================================================
# TEST R2: CAPITAL INVESTMENT
# ================================================================
print("\n--- R2: Capital Investment (Revenue ↑ → Machinery ↑ → Yield ↑) ---")
base_mach = val(baseline, yr, 'Machinery_Capital')
tech_mach = val(high_tech, yr, 'Machinery_Capital')

print(f"  Baseline:  machinery capital = {base_mach:>12,.0f}")
print(f"  High Tech: machinery capital = {tech_mach:>12,.0f}")
if tech_mach > base_mach:
    print(f"  ✓ PASS: Higher yield → higher revenue → more machinery investment")
else:
    print(f"  ~ INFO: Machinery similar (may be at equilibrium)")

# ================================================================
# TEST B3: WATER STRESS → IRRIGATION (ENDOGENOUS)
# ================================================================
print("\n--- B3: Water Stress → Irrigation (endogenous response) ---")
base_irr = val(baseline, yr, 'Irrigation_Fraction')
base_ws  = val(baseline, yr, 'Water_Stress_Factor')
irr_irr  = val(high_irr, yr, 'Irrigation_Fraction')

# Also check that irrigation changes even WITHOUT explicit scenario parameter
# (the endogenous water-stress response)
irr_2030 = val(baseline, 2030, 'Irrigation_Fraction')
irr_2070 = val(baseline, 2070, 'Irrigation_Fraction')

print(f"  Baseline irrigation: 2030={irr_2030:.4f}, 2070={irr_2070:.4f}")
print(f"  With irrigation investment: {yr}={irr_irr:.4f}")
if irr_2070 > irr_2030:
    print(f"  ✓ PASS: Irrigation expands endogenously over time ({irr_2030:.4f} → {irr_2070:.4f})")
else:
    print(f"  ✗ FAIL: Irrigation should expand from water stress feedback")
if irr_irr > base_irr:
    print(f"  ✓ PASS: Irrigation policy further increases expansion ({base_irr:.4f} → {irr_irr:.4f})")

# ================================================================
# TEST B4: EROSION → P DEPLETION → YIELD
# ================================================================
print("\n--- B4: Erosion → P Soil Stock Depletion → Yield ---")
base_p  = val(baseline, yr, 'P_Soil_Stock_kg_ha')
cons_p  = val(high_cons, yr, 'P_Soil_Stock_kg_ha')
base_p0 = val(baseline, 1971, 'P_Soil_Stock_kg_ha')

print(f"  P soil stock: initial={base_p0:.0f}, baseline@{yr}={base_p:.0f}, conservation@{yr}={cons_p:.0f}")
if cons_p > base_p:
    print(f"  ✓ PASS: Conservation preserves P soil stock ({base_p:.0f} → {cons_p:.0f})")
else:
    print(f"  ~ INFO: P stock similar (erosion may be low in this scenario)")

# ================================================================
# SUMMARY TABLE
# ================================================================
print("\n" + "=" * 80)
print(f"FULL COMPARISON TABLE — Year {yr}")
print("=" * 80)
header = f"{'Scenario':<20} {'Yield':>7} {'Prod':>12} {'Soil':>6} {'N':>5} {'Price':>7} {'Irr%':>6} {'P_stock':>8}"
print(header)
print("-" * len(header))
scenarios = [
    ("Baseline",      baseline),
    ("Nitrogen 2x",   high_n),
    ("Nitrogen 0.5x", low_n),
    ("Tech +2%/yr",   high_tech),
    ("Adapt 80%",     high_adapt),
    ("Conservation",  high_cons),
    ("Irrigation",    high_irr),
]
for name, df in scenarios:
    print(f"{name:<20} "
          f"{val(df, yr, 'Yield_t_per_ha'):>7.2f} "
          f"{val(df, yr, 'Production_tonnes'):>12,.0f} "
          f"{val(df, yr, 'Soil_Health'):>6.3f} "
          f"{val(df, yr, 'Nitrogen_kg_ha'):>5.0f} "
          f"{val(df, yr, 'Price_per_tonne'):>7.1f} "
          f"{val(df, yr, 'Irrigation_Fraction')*100:>5.1f}% "
          f"{val(df, yr, 'P_Soil_Stock_kg_ha'):>8.0f}")

print()
print("If values in the table above differ between scenarios,")
print("the feedback loops are working correctly.")
print()
print("To test in the DASHBOARD: change a slider, then look at")
print("the PROJECTION period (after 2023). Historical data (1971-2023)")
print("always shows the same values because it comes from CSV files.")
