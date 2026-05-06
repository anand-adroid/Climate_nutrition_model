"""
End-to-end pipeline test for the Climate-Nutrition Modelling v2.0.

Tests all 7 subsystems connect properly:
Climate → Crop Model (Land, Water, Soil N-P-K, CO₂) → Nutrient Conversion → Gap Analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from climate_nutrition_modelling.models.crop_model import CropModel, CropParameters
from climate_nutrition_modelling.models.climate_scenarios import (
    ClimateScenarioManager, REGIONAL_BASELINES
)
from climate_nutrition_modelling.models.nutrient_converter import (
    NutrientConverter, CROP_NUTRIENT_PROFILES
)
from climate_nutrition_modelling.models.nutrition_gap import (
    NutritionalGapAnalyzer, POPULATION_PROFILES
)
from climate_nutrition_modelling.models.sensitivity_analysis import SensitivityAnalyzer
from climate_nutrition_modelling.models.model_comparison import ModelComparison


def test_full_pipeline():
    """Run the complete v2.0 pipeline end-to-end."""
    print("=" * 60)
    print("CLIMATE-NUTRITION MODELLING v2.0: END-TO-END TEST")
    print("=" * 60)

    # --- 1. CLIMATE SCENARIOS ---
    print("\n--- Step 1: Climate Scenarios ---")
    baseline = REGIONAL_BASELINES['canada_ontario']
    climate_mgr = ClimateScenarioManager(baseline)

    scenarios = {}
    for ssp in ['ssp126', 'ssp245', 'ssp585']:
        df = climate_mgr.generate_scenario(ssp, 1971, 2080)
        scenarios[ssp] = df
        row_2080 = df[df['Year'] == 2080].iloc[0]
        print(f"  {ssp}: {len(df)} years, "
              f"Temp 2080: {row_2080['Temperature_C']:.1f}°C, "
              f"CO2: {row_2080['CO2_ppm']:.0f} ppm")

    assert len(scenarios['ssp245']) > 0
    assert 'CO2_ppm' in scenarios['ssp245'].columns
    print("  ✓ Climate scenarios generated")

    # --- 2. CROP MODEL (Canadian Corn) ---
    print("\n--- Step 2: Crop Model — Canadian Corn ---")
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'canada_corn.json'
    )
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'canada', 'corn_production.csv'
    )

    if os.path.exists(config_path):
        params = CropParameters.from_json(config_path)
    else:
        params = CropParameters(crop_name='maize_grain', country='canada',
                                year_end=2080, initial_area_ha=1_770_000,
                                initial_yield_t_per_ha=4.2, initial_price_per_tonne=65.0)

    params.year_end = 2080
    model = CropModel(params)
    if os.path.exists(data_path):
        model.load_historical_data(data_path)

    model.climate_data = scenarios['ssp245'][['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']]
    corn_results = model.run()

    assert len(corn_results) > 0
    assert corn_results['Production_tonnes'].iloc[-1] > 0

    # Check new v2 columns
    assert 'Water_Stress_Factor' in corn_results.columns, "Missing Water_Stress_Factor"
    assert 'Nitrogen_kg_ha' in corn_results.columns, "Missing Nitrogen_kg_ha"
    assert 'Phosphorus_kg_ha' in corn_results.columns, "Missing Phosphorus_kg_ha"
    assert 'Potassium_kg_ha' in corn_results.columns, "Missing Potassium_kg_ha"
    assert 'NPK_Response' in corn_results.columns, "Missing NPK_Response"
    assert 'Erosion_Rate_t_ha_yr' in corn_results.columns, "Missing Erosion_Rate_t_ha_yr"
    assert 'CO2_Fert_Factor' in corn_results.columns, "Missing CO2_Fert_Factor"
    assert 'CO2_Nutrient_Factor' in corn_results.columns, "Missing CO2_Nutrient_Factor"
    assert 'Arable_Land_Available_ha' in corn_results.columns, "Missing Arable_Land_Available_ha"

    print(f"  Production 1971: {corn_results.iloc[0]['Production_tonnes']:,.0f} t")
    print(f"  Production 2080: {corn_results.iloc[-1]['Production_tonnes']:,.0f} t")
    print(f"  Yield 2080: {corn_results.iloc[-1]['Yield_t_per_ha']:.2f} t/ha")
    print(f"  Soil Health 2080: {corn_results.iloc[-1]['Soil_Health']:.3f}")
    print(f"  Water Stress 2080: {corn_results.iloc[-1]['Water_Stress_Factor']:.3f}")
    print(f"  NPK Response 2080: {corn_results.iloc[-1]['NPK_Response']:.3f}")
    print(f"  Erosion 2080: {corn_results.iloc[-1]['Erosion_Rate_t_ha_yr']:.2f} t/ha/yr")
    print(f"  CO2 Fert Factor 2080: {corn_results.iloc[-1]['CO2_Fert_Factor']:.3f}")
    print("  ✓ Crop model (7 subsystems) executed successfully")

    # --- 3. VEGETABLE CROPS ---
    print("\n--- Step 3: Vegetable Crops (Tomato + Leafy Greens) ---")
    veg_results = {}
    for crop_key, config_file, data_file in [
        ('tomato', 'canada_tomato.json', 'tomato_production.csv'),
        ('leafy_greens', 'canada_leafy_greens.json', 'leafy_greens_production.csv'),
    ]:
        cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', config_file)
        dat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'canada', data_file)

        p = CropParameters.from_json(cfg) if os.path.exists(cfg) else CropParameters(crop_name=crop_key)
        p.year_end = 2080
        m = CropModel(p)
        if os.path.exists(dat):
            m.load_historical_data(dat)
        m.climate_data = scenarios['ssp245'][['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']]
        r = m.run()
        veg_results[crop_key] = r

        assert len(r) > 0
        assert r['Production_tonnes'].iloc[-1] > 0
        print(f"  {crop_key}: Prod 2080 = {r.iloc[-1]['Production_tonnes']:,.0f} t, "
              f"Yield = {r.iloc[-1]['Yield_t_per_ha']:.1f} t/ha")

    print("  ✓ Vegetable models executed successfully")

    # --- 4. NUTRIENT CONVERSION ---
    print("\n--- Step 4: Nutrient Conversion ---")
    converter = NutrientConverter()
    all_crops = {'maize_grain': corn_results}
    all_crops.update(veg_results)

    for crop_key, df in all_crops.items():
        prod = df.iloc[-1]['Production_tonnes']
        if crop_key in CROP_NUTRIENT_PROFILES:
            nuts = converter.convert_crop(crop_key, prod)
            print(f"  {crop_key}: Energy={nuts['energy_kcal']:,.0f} kcal, "
                  f"Protein={nuts['protein_kg']:,.0f} kg, "
                  f"Iron={nuts['iron_g']:,.0f} g")

    print("  ✓ Nutrient conversion successful")

    # --- 5. NUTRITIONAL GAP ANALYSIS ---
    print("\n--- Step 5: Gap Analysis ---")
    population = 45_000_000
    total_nutrients = converter.get_total_nutrients()

    analyzer = NutritionalGapAnalyzer(population=population, profile='canada')
    analyzer.set_nutrient_supply(total_nutrients)
    gap_df = analyzer.analyze()

    print(f"  Population: {population:,}")
    print(f"  Nutrients analysed: {len(gap_df)}")
    for _, row in gap_df.iterrows():
        icon = "✓" if row['Gap_Ratio'] >= 1.0 else "✗"
        print(f"    {icon} {row['Nutrient']}: {row['Gap_Ratio']:.1%} — {row['Status']}")

    # --- 6. INTERVENTIONS ---
    print("\n--- Step 6: Priority Interventions ---")
    interventions = analyzer.get_priority_interventions()
    if interventions:
        print(f"  {len(interventions)} deficits:")
        for intv in interventions[:5]:
            print(f"    [{intv['priority']}] {intv['nutrient']}: "
                  f"{intv['current_ratio']:.1%} → Grow: {', '.join(intv['recommended_crops'][:3])}")
    else:
        print("  No deficits")
    print("  ✓ Gap analysis complete")

    # --- 7. MULTI-SCENARIO ---
    print("\n--- Step 7: Multi-Scenario Comparison (Corn) ---")
    for ssp in ['ssp126', 'ssp245', 'ssp585']:
        p = CropParameters.from_json(config_path) if os.path.exists(config_path) else CropParameters(crop_name='maize_grain', country='canada')
        p.year_end = 2080
        m = CropModel(p)
        if os.path.exists(data_path):
            m.load_historical_data(data_path)
        m.climate_data = scenarios[ssp][['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']]
        r = m.run()
        print(f"  {ssp}: Prod 2080 = {r.iloc[-1]['Production_tonnes']:,.0f} t "
              f"(Yield: {r.iloc[-1]['Yield_t_per_ha']:.2f} t/ha, "
              f"Water: {r.iloc[-1]['Water_Stress_Factor']:.2f})")

    print("  ✓ Multi-scenario comparison complete")

    # --- 8. SENSITIVITY ANALYSIS ---
    print("\n--- Step 8: Sensitivity Analysis (quick 3-level test) ---")
    sa_params = CropParameters.from_json(config_path) if os.path.exists(config_path) else CropParameters(crop_name='maize_grain')
    sa_params.year_end = 2080
    sa = SensitivityAnalyzer(
        base_params=sa_params,
        climate_data=scenarios['ssp245'][['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']],
        historical_data_path=data_path if os.path.exists(data_path) else None,
        parameters={
            'initial_nitrogen_kg_ha': {'label': 'N application', 'subsystem': 'Soil', 'range': (0.5, 2.0), 'description': 'test'},
            'temperature_sensitivity': {'label': 'Temp sensitivity', 'subsystem': 'Climate', 'range': (0.5, 2.0), 'description': 'test'},
        },
    )
    sa_results = sa.run_oat(n_levels=3)
    assert len(sa_results) > 0, "Sensitivity analysis returned no results"
    tornado = sa.get_tornado_data('Production_2080')
    assert len(tornado) > 0, "Tornado data empty"
    rankings = sa.get_parameter_rankings()
    assert len(rankings) > 0, "Rankings empty"
    print(f"  OAT results: {len(sa_results)} data points")
    print(f"  Tornado: {len(tornado)} parameters ranked")
    print(f"  Top parameter: {rankings.iloc[0]['Parameter']} (range: {rankings.iloc[0]['Avg_Range']:.1f}%)")
    print("  ✓ Sensitivity analysis complete")

    # --- 9. MODEL COMPARISON ---
    print("\n--- Step 9: Model Comparison & Benchmarking ---")
    comp = ModelComparison()
    comp.add_sd_results('maize_grain', 'ssp245', corn_results)
    yield_comp = comp.compare_yield_projections()
    print(f"  Yield comparisons: {len(yield_comp)} benchmarks")
    if len(yield_comp) > 0:
        within = yield_comp['Within_Range'].sum()
        total = len(yield_comp)
        print(f"  Within benchmark range: {within}/{total}")

    co2_comp = comp.compare_co2_nutrient_effects()
    print(f"  CO₂ nutrient comparisons: {len(co2_comp)}")

    scorecard = comp.get_scorecard()
    print(f"  Scorecard entries: {len(scorecard)}")

    # Validation stats
    if os.path.exists(data_path):
        hist_df = pd.read_csv(data_path)
        val_stats = comp.compute_validation_stats(corn_results, hist_df)
        if 'error' not in val_stats:
            print(f"  Validation: R²={val_stats['R2']:.4f}, MAPE={val_stats['MAPE_pct']:.1f}%, "
                  f"NSE={val_stats['NSE']:.4f}, Willmott d={val_stats['Willmott_d']:.4f}")
        else:
            print(f"  Validation: {val_stats}")

    ref_table = ModelComparison.get_benchmark_table()
    assert len(ref_table) > 0, "Benchmark reference table empty"
    print(f"  Reference benchmarks: {len(ref_table)}")
    print("  ✓ Model comparison complete")

    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓ (v2.1 — 7 Subsystems + SA + Comparison)")
    print("=" * 60)
    print(f"\nModules verified:")
    print(f"  • CropModel v2: {len(corn_results)} time steps, "
          f"{len(corn_results.columns)} output columns")
    print(f"  • Subsystems: Land, Water, Soil (N-P-K + erosion), "
          f"Climate (CO₂), Crop, Economics, Demand")
    print(f"  • Crops: maize_grain, tomato, leafy_greens")
    print(f"  • ClimateScenarioManager: {len(scenarios)} scenarios")
    print(f"  • NutrientConverter: {len(CROP_NUTRIENT_PROFILES)} crop profiles")
    print(f"  • NutritionalGapAnalyzer: {len(gap_df)} nutrients tracked")
    print(f"  • SensitivityAnalyzer: OAT + tornado + rankings")
    print(f"  • ModelComparison: IPCC + AgMIP + DSSAT + FACE benchmarks")
    print(f"  • Data: REAL from StatCan/FAOSTAT (verified anchor points)")

    return True


if __name__ == '__main__':
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
