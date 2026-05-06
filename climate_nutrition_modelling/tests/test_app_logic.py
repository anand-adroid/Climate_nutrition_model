"""
Test the exact logic flow that app.py v2.0 uses, without Streamlit.
Catches any errors the dashboard would hit at runtime.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from climate_nutrition_modelling.models.crop_model import CropModel, CropParameters
from climate_nutrition_modelling.models.climate_scenarios import ClimateScenarioManager, REGIONAL_BASELINES
from climate_nutrition_modelling.models.nutrient_converter import NutrientConverter, CROP_NUTRIENT_PROFILES
from climate_nutrition_modelling.models.nutrition_gap import NutritionalGapAnalyzer, POPULATION_PROFILES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

def test_app_pipeline():
    """Simulate exactly what app.py run_pipeline() does."""
    print("Testing app.py v2.0 pipeline logic...\n")

    country_key = "Canada"
    ssp = "ssp245"
    y_end = 2080

    # 1. Climate
    baseline = REGIONAL_BASELINES['canada_ontario']
    climate_mgr = ClimateScenarioManager(baseline)
    climate_df = climate_mgr.generate_scenario(ssp, 1971, y_end)
    print(f"  Climate: {len(climate_df)} rows, cols={list(climate_df.columns)}")

    climate_summary = climate_mgr.get_scenario_summary()
    print(f"  Summary: {len(climate_summary)} scenarios")

    # 2. Population
    pop_path = os.path.join(DATA_DIR, 'canada', 'canada_population.csv')
    pop_df = pd.read_csv(pop_path)
    print(f"  Population: {len(pop_df)} rows")

    # 3. Crop models (v2.0: tomato + leafy_greens instead of vegetables_mixed)
    crop_configs = [
        ('maize_grain', 'canada_corn.json', 'corn_production.csv'),
        ('tomato', 'canada_tomato.json', 'tomato_production.csv'),
        ('leafy_greens', 'canada_leafy_greens.json', 'leafy_greens_production.csv'),
    ]

    crop_results = {}
    for crop_key, config_file, data_file in crop_configs:
        config_path = os.path.join(CONFIG_DIR, config_file)
        params = CropParameters.from_json(config_path)
        params.year_end = y_end

        model = CropModel(params)

        data_path = os.path.join(DATA_DIR, 'canada', data_file)
        if os.path.exists(data_path):
            model.load_historical_data(data_path)

        model.climate_data = climate_df[['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']].copy()
        model.population_data = pop_df.copy()

        model.set_scenario(
            nitrogen_multiplier=1.0,
            phosphorus_multiplier=1.0,
            potassium_multiplier=1.0,
            technology_boost=0.0,
            climate_adaptation=0.0,
            conservation_investment=0.0,
            irrigation_investment=0.0,
            area_constraint=1.0,
        )

        sim_df = model.run()
        crop_results[crop_key] = sim_df

        validation = model.validate()
        print(f"  {crop_key}: {len(sim_df)} rows, {len(sim_df.columns)} cols, "
              f"validation={list(validation.keys()) if isinstance(validation, dict) and 'error' not in validation else validation}")

    # 4. Nutrient conversion
    converter = NutrientConverter()
    nutrient_timeseries = {}

    for crop_key, sim_df in crop_results.items():
        profile_key = crop_key
        if profile_key in CROP_NUTRIENT_PROFILES:
            nut_ts = converter.convert_production_timeseries(
                profile_key, sim_df['Production_tonnes'], sim_df['Year'])
            nutrient_timeseries[crop_key] = nut_ts
            print(f"  Nutrients {crop_key}: {len(nut_ts)} rows, cols={len(nut_ts.columns)}")

    # 5. Gap analysis
    final_year_supply = {}
    for crop_key, nut_df in nutrient_timeseries.items():
        last_row = nut_df.iloc[-1]
        for col in NutrientConverter.TRACKED_NUTRIENTS:
            if col in nut_df.columns:
                final_year_supply[col] = final_year_supply.get(col, 0) + last_row[col]

    pop_at_end = np.interp(y_end, pop_df['Year'], pop_df['Population'])
    print(f"  Population at {y_end}: {pop_at_end:,.0f}")

    analyzer = NutritionalGapAnalyzer(population=pop_at_end, profile='canada')
    analyzer.set_nutrient_supply(final_year_supply)
    gap_df = analyzer.analyze()
    print(f"  Gap analysis: {len(gap_df)} nutrients")

    interventions = analyzer.get_priority_interventions()
    print(f"  Interventions: {len(interventions)} deficits")

    # 6. Gap timeseries
    gap_years = list(crop_results.values())[0]['Year'].values
    gap_ts_rows = []
    for i, yr in enumerate(gap_years):
        yr_supply = {}
        for crop_key, nut_df in nutrient_timeseries.items():
            yr_row = nut_df[nut_df['Year'] == yr]
            if len(yr_row) > 0:
                for col in NutrientConverter.TRACKED_NUTRIENTS:
                    if col in nut_df.columns:
                        yr_supply[col] = yr_supply.get(col, 0) + yr_row[col].values[0]

        pop = np.interp(yr, pop_df['Year'], pop_df['Population'])
        analyzer_ts = NutritionalGapAnalyzer(population=pop, profile='canada')
        analyzer_ts.set_nutrient_supply(yr_supply)
        gap_result = analyzer_ts.analyze()

        row = {'Year': yr, 'Population': pop}
        for _, g in gap_result.iterrows():
            row[g['Nutrient']] = g['Gap_Ratio']
        gap_ts_rows.append(row)

    gap_ts = pd.DataFrame(gap_ts_rows)
    print(f"  Gap timeseries: {len(gap_ts)} rows, cols={list(gap_ts.columns)[:5]}...")

    # 7. Test tab data access
    print("\n  Testing tab data access...")

    # Overview tab
    total_production = sum(df.iloc[-1]['Production_tonnes'] for df in crop_results.values())
    energy_ratio = gap_df[gap_df['Nutrient'] == 'Energy']['Gap_Ratio'].values
    protein_ratio = gap_df[gap_df['Nutrient'] == 'Protein']['Gap_Ratio'].values
    print(f"    Overview: prod={total_production:,.0f}, energy={energy_ratio[0]:.3f}, protein={protein_ratio[0]:.3f}")

    # Crop tab — check all required columns
    for crop_key, df in crop_results.items():
        assert 'Production_tonnes' in df.columns, f"Missing Production_tonnes in {crop_key}"
        assert 'Yield_t_per_ha' in df.columns, f"Missing Yield_t_per_ha in {crop_key}"
        assert 'Area_Harvested_ha' in df.columns, f"Missing Area_Harvested_ha in {crop_key}"
        assert 'Price_per_tonne' in df.columns, f"Missing Price_per_tonne in {crop_key}"
        assert 'Soil_Health' in df.columns, f"Missing Soil_Health in {crop_key}"
        # v2.0 new columns
        assert 'Water_Stress_Factor' in df.columns, f"Missing Water_Stress_Factor in {crop_key}"
        assert 'Nitrogen_kg_ha' in df.columns, f"Missing Nitrogen_kg_ha in {crop_key}"
        assert 'Phosphorus_kg_ha' in df.columns, f"Missing Phosphorus_kg_ha in {crop_key}"
        assert 'Potassium_kg_ha' in df.columns, f"Missing Potassium_kg_ha in {crop_key}"
        assert 'NPK_Response' in df.columns, f"Missing NPK_Response in {crop_key}"
        assert 'Erosion_Rate_t_ha_yr' in df.columns, f"Missing Erosion_Rate_t_ha_yr in {crop_key}"
        assert 'CO2_Fert_Factor' in df.columns, f"Missing CO2_Fert_Factor in {crop_key}"
        assert 'CO2_Nutrient_Factor' in df.columns, f"Missing CO2_Nutrient_Factor in {crop_key}"
        assert 'Arable_Land_Available_ha' in df.columns, f"Missing Arable_Land_Available_ha in {crop_key}"
        assert 'Irrigation_Fraction' in df.columns, f"Missing Irrigation_Fraction in {crop_key}"
    print("    Crop Production tab: all columns present ✓")

    # Climate tab
    assert 'Temperature_C' in climate_df.columns
    assert 'Precipitation_mm' in climate_df.columns
    assert 'GDD' in climate_df.columns
    assert 'CO2_ppm' in climate_df.columns
    assert 'Climate_Stress_Factor' in climate_df.columns
    print("    Climate tab: all columns present ✓")

    # Nutrient tab
    for crop_key, nut_df in nutrient_timeseries.items():
        assert 'energy_kcal' in nut_df.columns
        assert 'protein_kg' in nut_df.columns
    print("    Nutrient Supply tab: all columns present ✓")

    # Gap tab
    assert 'Nutrient' in gap_df.columns
    assert 'Gap_Ratio' in gap_df.columns
    assert 'Status' in gap_df.columns
    nutrient_cols = [c for c in gap_ts.columns if c not in ['Year', 'Population']]
    assert len(nutrient_cols) > 0
    print(f"    Gaps tab: {len(nutrient_cols)} nutrient time series ✓")

    # Interventions tab
    for intv in interventions:
        assert 'nutrient' in intv
        assert 'current_ratio' in intv
        assert 'recommended_crops' in intv
    print(f"    Interventions tab: {len(interventions)} items ✓")

    print("\n" + "=" * 50)
    print("ALL APP LOGIC TESTS PASSED ✓ (v2.0)")
    print("=" * 50)

if __name__ == '__main__':
    test_app_pipeline()
