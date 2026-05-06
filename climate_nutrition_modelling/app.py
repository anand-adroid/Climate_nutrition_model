"""
Climate-Nutrition Modelling Dashboard — v2.1 (Production)

Streamlit application for analysing crop production under climate change
and assessing nutritional adequacy for a given population.

Pipeline: Climate Scenarios → Crop Models (7 subsystems) → Nutrient Conversion → Gap Analysis

Run:  python -m streamlit run climate_nutrition_modelling/app.py

Author: Climate-Nutrition Modelling Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from climate_nutrition_modelling.models.crop_model import CropModel, CropParameters
from climate_nutrition_modelling.models.climate_scenarios import (
    ClimateScenarioManager, REGIONAL_BASELINES
)
from climate_nutrition_modelling.models.nutrient_converter import (
    NutrientConverter, CROP_NUTRIENT_PROFILES
)
from climate_nutrition_modelling.models.nutrition_gap import (
    NutritionalGapAnalyzer, POPULATION_PROFILES, DailyNutrientRequirement
)
from climate_nutrition_modelling.models.sensitivity_analysis import (
    SensitivityAnalyzer, SENSITIVITY_PARAMETERS, OUTPUT_METRICS
)
from climate_nutrition_modelling.models.model_comparison import (
    ModelComparison, IPCC_YIELD_CHANGE_PER_DEGREE, AGMIP_PROJECTIONS,
    CO2_NUTRIENT_BENCHMARKS, OBSERVED_YIELD_TRENDS
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Climate-Nutrition Model v2.1",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: fix sidebar caption overlap on slider hover ──
st.markdown("""
<style>
/* ── Sidebar spacing: prevent slider tooltips overlapping captions ── */

/* Give each slider/select_slider extra bottom margin so the tooltip
   bubble (which appears below the thumb) has room before the next element. */
[data-testid="stSidebar"] .stSlider {
    margin-bottom: 1.2rem !important;
}
[data-testid="stSidebar"] .stSelectSlider {
    margin-bottom: 1.2rem !important;
}

/* Captions (the "→ 50 kg N/ha" helper lines) — compact but clear */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    margin-top: -0.8rem !important;
    padding-bottom: 0.5rem !important;
}

/* Body text in sidebar */
[data-testid="stSidebar"] .stMarkdown p {
    margin-bottom: 0.2rem !important;
    line-height: 1.35 !important;
}

/* Section headings */
[data-testid="stSidebar"] h3 {
    margin-top: 0.4rem !important;
    margin-bottom: 0.4rem !important;
}

/* Dividers */
[data-testid="stSidebar"] hr {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}

/* Ensure slider tooltip (popover) renders above other elements */
[data-testid="stSidebar"] [data-testid="stThumbValue"] {
    z-index: 999 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")


HIST_END_YEAR = 2023  # Last year of historical CSV data


def add_projection_boundary(fig, hist_end=HIST_END_YEAR):
    """Add a vertical dashed line separating historical data from projections."""
    fig.add_vline(
        x=hist_end + 0.5, line_dash="dot", line_color="rgba(100,100,100,0.5)",
        annotation_text="← Historical | Projection →",
        annotation_position="top",
        annotation_font_size=10,
        annotation_font_color="gray",
    )
    return fig


def load_config(filename):
    path = os.path.join(CONFIG_DIR, filename)
    if os.path.exists(path):
        return CropParameters.from_json(path)
    return None


def load_csv(country, filename):
    path = os.path.join(DATA_DIR, country, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def fmt(n, d=1):
    if abs(n) >= 1e9:
        return f"{n/1e9:.{d}f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.{d}f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.{d}f}K"
    return f"{n:.{d}f}"


COLORS = {
    'ssp126': '#2ecc71', 'ssp245': '#f39c12',
    'ssp370': '#e74c3c', 'ssp585': '#8e44ad',
    'primary': '#2c3e50', 'secondary': '#3498db',
    'success': '#27ae60', 'warning': '#f39c12', 'danger': '#e74c3c',
}

# ============================================================
# SIDEBAR — Policy Control Panel
# ============================================================
st.sidebar.title("Climate-Nutrition Model")
st.sidebar.markdown("---")

country = st.sidebar.selectbox("Country", ["Canada", "Nigeria"], help="Select country")

# Load baseline config values for real-unit display
_baseline_configs = {}
if country == "Canada":
    for _cname, _cfile in [('Corn', 'canada_corn.json'), ('Tomato', 'canada_tomato.json'), ('Leafy Greens', 'canada_leafy_greens.json')]:
        _p = load_config(_cfile)
        if _p:
            _baseline_configs[_cname] = _p
else:  # Nigeria
    for _cname, _cfile in [('Maize', 'nigeria_maize.json'), ('Cassava', 'nigeria_cassava.json'), ('Rice', 'nigeria_rice.json'), ('Sorghum', 'nigeria_sorghum.json')]:
        _p = load_config(_cfile)
        if _p:
            _baseline_configs[_cname] = _p
_corn = _baseline_configs.get('Corn') or _baseline_configs.get('Maize')

# ── Climate Scenario ──
st.sidebar.markdown("### 1. Climate Scenario")
ssp_scenario = st.sidebar.selectbox(
    "IPCC Pathway",
    ["ssp126", "ssp245", "ssp370", "ssp585"],
    index=1,
    format_func=lambda x: {
        'ssp126': 'SSP1-2.6 — Sustainability',
        'ssp245': 'SSP2-4.5 — Middle Road',
        'ssp370': 'SSP3-7.0 — Regional Rivalry',
        'ssp585': 'SSP5-8.5 — Fossil Fuel Dev.',
    }[x]
)
_ssp_warming = {'ssp126': '+1.8°C', 'ssp245': '+2.7°C', 'ssp370': '+3.6°C', 'ssp585': '+4.4°C'}
_ssp_co2 = {'ssp126': '430', 'ssp245': '600', 'ssp370': '850', 'ssp585': '1135'}
_amp_label = "Canada: 1.4x amplification" if country == "Canada" else "Nigeria: 1.1x amplification"
st.sidebar.caption(
    f"→ Warming: {_ssp_warming[ssp_scenario]} by 2100 | "
    f"CO₂: {_ssp_co2[ssp_scenario]} ppm | "
    f"{_amp_label}"
)

# ── Fertiliser Policy ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Fertiliser Policy")

# Baseline values for corn (largest crop)
_n_base = _corn.initial_nitrogen_kg_ha if _corn else 50
_p_base = _corn.initial_phosphorus_kg_ha if _corn else 22
_k_base = _corn.initial_potassium_kg_ha if _corn else 35

nitrogen_mult = st.sidebar.slider(
    f"Nitrogen (N) — base {_n_base:.0f} kg/ha",
    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    format="%.1fx",
    help=f"Multiplier on baseline N application. Corn baseline = {_n_base:.0f} kg/ha (1971)."
)
_n_actual = _n_base * nitrogen_mult
st.sidebar.caption(f"→ {_n_actual:.0f} kg N/ha")

phosphorus_mult = st.sidebar.slider(
    f"Phosphorus (P) — base {_p_base:.0f} kg/ha",
    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    format="%.1fx",
    help=f"Multiplier on baseline P application. Corn baseline = {_p_base:.0f} kg/ha."
)
_p_actual = _p_base * phosphorus_mult
st.sidebar.caption(f"→ {_p_actual:.0f} kg P/ha")

potassium_mult = st.sidebar.slider(
    f"Potassium (K) — base {_k_base:.0f} kg/ha",
    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
    format="%.1fx",
    help=f"Multiplier on baseline K application. Corn baseline = {_k_base:.0f} kg/ha."
)
_k_actual = _k_base * potassium_mult
st.sidebar.caption(f"→ {_k_actual:.0f} kg K/ha")

# ── Estimated fertiliser cost
_n_cost = _corn.nitrogen_cost_per_kg if _corn else 1.1
_p_cost = _corn.phosphorus_cost_per_kg if _corn else 2.8
_k_cost = _corn.potassium_cost_per_kg if _corn else 0.75
_fert_cost_ha = (_n_actual * _n_cost + _p_actual * _p_cost + _k_actual * _k_cost)
_fert_cost_baseline = _n_base * _n_cost + _p_base * _p_cost + _k_base * _k_cost
st.sidebar.caption(f"Cost: ${_fert_cost_ha:.0f}/ha (base ${_fert_cost_baseline:.0f}/ha)")

# ── Technology & Adaptation ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Technology & Adaptation")

tech_boost = st.sidebar.slider(
    "R&D investment boost (%/yr)",
    min_value=0.0, max_value=2.0, value=0.0, step=0.2,
    format="%.1f%%",
    help="Additional annual yield improvement from R&D. 0% = no extra R&D. "
         "Baseline tech growth is already ~0.8%/yr for corn."
)
tech_boost_val = tech_boost / 100.0  # Convert % to decimal for the model
_base_tech = (_corn.technology_growth_rate if _corn else 0.008)
_total_tech = _base_tech + tech_boost_val
st.sidebar.caption(f"→ {_total_tech*100:.1f}%/yr total (+{((1+_total_tech)**30 - 1)*100:.0f}% by 2053)")

climate_adapt = st.sidebar.select_slider(
    "Climate adaptation strategy",
    options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    value=0.0,
    format_func=lambda x: {
        0.0: "None",
        0.2: "Minimal — seed selection",
        0.4: "Moderate — drought-tolerant varieties",
        0.6: "Strong — varieties + adjusted planting dates",
        0.8: "Intensive — above + supplemental irrigation",
        1.0: "Maximum — full adaptation package",
    }[x]
)
_adapt_effect = climate_adapt * 50  # rough % stress reduction
st.sidebar.caption(f"→ Stress reduction: ~{_adapt_effect:.0f}%")

# ── Soil & Land Management ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 4. Soil & Land Management")

conservation = st.sidebar.select_slider(
    "Conservation practices",
    options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    value=0.0,
    format_func=lambda x: {
        0.0: "None — conventional tillage",
        0.2: "Minimal — crop rotation",
        0.4: "Moderate — reduced tillage + rotation",
        0.6: "Strong — no-till + cover crops",
        0.8: "Intensive — above + organic amendments",
        1.0: "Maximum — regenerative agriculture",
    }[x]
)
_erosion_reduction = conservation * 50
st.sidebar.caption(f"→ Erosion -{_erosion_reduction:.0f}% | Soil regen +{conservation * 50:.0f}%")

irrigation_inv = st.sidebar.select_slider(
    "Irrigation expansion",
    options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    value=0.0,
    format_func=lambda x: {
        0.0: "None — current irrigated area",
        0.2: "Slow — +0.1%/yr expansion",
        0.4: "Moderate — +0.2%/yr expansion",
        0.6: "Fast — +0.3%/yr expansion",
        0.8: "Rapid — +0.4%/yr expansion",
        1.0: "Maximum — +0.5%/yr expansion",
    }[x]
)
_corn_irr_base = (_corn.irrigation_fraction if _corn else 0.05) * 100
_corn_irr_2080 = min(95, _corn_irr_base + irrigation_inv * 0.5 * (2080 - 2023))
st.sidebar.caption(f"→ Corn: {_corn_irr_base:.0f}% now → ~{_corn_irr_2080:.0f}% by 2080")

area_constraint = st.sidebar.slider(
    "Land use policy",
    min_value=0.5, max_value=1.5, value=1.0, step=0.1,
    format="%.1fx",
    help="1.0 = current max farmland. <1 = protect land from farming. >1 = allow expansion."
)
_main_crop_name = "Corn" if country == "Canada" else "Maize"
_corn_max = (_corn.area_max_ha if _corn else 3_500_000) * area_constraint
_area_note = "baseline" if area_constraint == 1.0 else f"{(area_constraint-1)*100:+.0f}%"
st.sidebar.caption(f"→ {_main_crop_name} max: {_corn_max/1e6:.2f}M ha ({_area_note})")

# ── Projection Range ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 5. Projection Range")
year_end = st.sidebar.slider("Simulate until", 2030, 2100, 2080, 5)
st.sidebar.caption(f"→ {year_end - 1971} years (1971–{year_end})")

# Convert tech_boost back for model (the slider was in %, model needs decimal)
tech_boost = tech_boost_val

# ============================================================
# RUN PIPELINE
# ============================================================

@st.cache_data
def run_pipeline(country_key, ssp, n_mult, p_mult, k_mult, tech, adapt,
                 conserv, irr_inv, area_c, y_end):
    results = {}

    # 1. Climate
    if country_key == "Canada":
        baseline = REGIONAL_BASELINES['canada_ontario']
        pop_df = load_csv('canada', 'canada_population.csv')
    else:
        baseline = REGIONAL_BASELINES['nigeria_north']
        pop_df = load_csv('nigeria', 'nigeria_population.csv')

    climate_mgr = ClimateScenarioManager(baseline)
    climate_df = climate_mgr.generate_scenario(ssp, 1971, y_end)
    results['climate'] = climate_df
    results['climate_summary'] = climate_mgr.get_scenario_summary()

    # 2. Crop models
    crop_configs = {
        'Canada': [
            ('maize_grain', 'canada_corn.json', 'corn_production.csv'),
            ('tomato', 'canada_tomato.json', 'tomato_production.csv'),
            ('leafy_greens', 'canada_leafy_greens.json', 'leafy_greens_production.csv'),
        ],
        'Nigeria': [
            ('maize_grain', 'nigeria_maize.json', 'maize_production.csv'),
            ('cassava', 'nigeria_cassava.json', 'cassava_production.csv'),
            ('rice_paddy', 'nigeria_rice.json', 'rice_production.csv'),
            ('sorghum', 'nigeria_sorghum.json', 'sorghum_production.csv'),
        ],
    }

    crop_results = {}
    for crop_key, config_file, data_file in crop_configs.get(country_key, []):
        params = load_config(config_file)
        if params is None:
            params = CropParameters(crop_name=crop_key, country=country_key.lower())
        params.year_end = y_end

        model = CropModel(params)

        # Historical data
        data_path = os.path.join(DATA_DIR, country_key.lower(), data_file)
        if os.path.exists(data_path):
            model.load_historical_data(data_path)

        # Climate & population
        model.climate_data = climate_df[['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']].copy()
        if pop_df is not None:
            model.population_data = pop_df.copy()

        model.set_scenario(
            nitrogen_multiplier=n_mult,
            phosphorus_multiplier=p_mult,
            potassium_multiplier=k_mult,
            technology_boost=tech,
            climate_adaptation=adapt,
            conservation_investment=conserv,
            irrigation_investment=irr_inv,
            area_constraint=area_c,
        )

        sim_df = model.run()
        crop_results[crop_key] = sim_df

        validation = model.validate()
        results[f'validation_{crop_key}'] = validation

    results['crops'] = crop_results

    # 3. Nutrient conversion
    converter = NutrientConverter()
    nutrient_timeseries = {}
    for crop_key, sim_df in crop_results.items():
        profile_key = crop_key
        if profile_key in CROP_NUTRIENT_PROFILES:
            nut_ts = converter.convert_production_timeseries(
                profile_key, sim_df['Production_tonnes'], sim_df['Year'])
            nutrient_timeseries[crop_key] = nut_ts

    results['nutrients'] = nutrient_timeseries

    # 4. Gap analysis
    profile_name = country_key.lower()
    if profile_name not in POPULATION_PROFILES:
        profile_name = 'global_average'

    final_year_supply = {}
    for crop_key, nut_df in nutrient_timeseries.items():
        last = nut_df.iloc[-1]
        for col in NutrientConverter.TRACKED_NUTRIENTS:
            if col in nut_df.columns:
                final_year_supply[col] = final_year_supply.get(col, 0) + last[col]

    pop_at_end = np.interp(y_end, pop_df['Year'], pop_df['Population']) if pop_df is not None else 40_000_000
    analyzer = NutritionalGapAnalyzer(population=pop_at_end, profile=profile_name)
    analyzer.set_nutrient_supply(final_year_supply)
    gap_df = analyzer.analyze()
    results['gap_analysis'] = gap_df
    results['interventions'] = analyzer.get_priority_interventions()
    results['population_at_end'] = pop_at_end

    # Gap timeseries
    gap_years = list(crop_results.values())[0]['Year'].values
    gap_ts_rows = []
    for yr in gap_years:
        yr_supply = {}
        for crop_key, nut_df in nutrient_timeseries.items():
            yr_row = nut_df[nut_df['Year'] == yr]
            if len(yr_row) > 0:
                for col in NutrientConverter.TRACKED_NUTRIENTS:
                    if col in nut_df.columns:
                        yr_supply[col] = yr_supply.get(col, 0) + yr_row[col].values[0]
        pop = np.interp(yr, pop_df['Year'], pop_df['Population']) if pop_df is not None else 40_000_000
        a = NutritionalGapAnalyzer(population=pop, profile=profile_name)
        a.set_nutrient_supply(yr_supply)
        g = a.analyze()
        row = {'Year': yr, 'Population': pop}
        for _, r in g.iterrows():
            row[r['Nutrient']] = r['Gap_Ratio']
        gap_ts_rows.append(row)

    results['gap_timeseries'] = pd.DataFrame(gap_ts_rows)
    return results


# Run
try:
    with st.spinner("Running climate-nutrition pipeline..."):
        results = run_pipeline(
            country, ssp_scenario, nitrogen_mult, phosphorus_mult, potassium_mult,
            tech_boost, climate_adapt, conservation, irrigation_inv, area_constraint,
            year_end
        )
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Overview",
    "Crop Production",
    "Water & Land",
    "Soil & Nutrients",
    "Climate Impact",
    "Nutrient Supply",
    "Nutritional Gaps",
    "Sensitivity Analysis",
    "Model Comparison",
    "Policy Interventions",
    "Data Tables",
])

# ============================================================
# TAB 1: OVERVIEW
# ============================================================
with tab1:
    st.header(f"Climate-Nutrition Analysis: {country}")
    st.markdown(f"**Scenario:** {ssp_scenario.upper()} | **Projection to:** {year_end}")

    col1, col2, col3, col4 = st.columns(4)

    total_production = sum(df.iloc[-1]['Production_tonnes'] for df in results['crops'].values())
    gap_df = results['gap_analysis']
    energy_ratio = gap_df[gap_df['Nutrient'] == 'Energy']['Gap_Ratio'].values
    protein_ratio = gap_df[gap_df['Nutrient'] == 'Protein']['Gap_Ratio'].values

    with col1:
        st.metric("Total Crop Production", fmt(total_production) + " t")
    with col2:
        st.metric("Population", fmt(results['population_at_end']))
    with col3:
        val = energy_ratio[0] if len(energy_ratio) > 0 else 0
        st.metric("Energy Adequacy", f"{val:.1%}",
                  delta="Sufficient" if val >= 1.0 else "Deficit",
                  delta_color="normal" if val >= 1.0 else "inverse")
    with col4:
        val = protein_ratio[0] if len(protein_ratio) > 0 else 0
        st.metric("Protein Adequacy", f"{val:.1%}",
                  delta="Sufficient" if val >= 1.0 else "Deficit",
                  delta_color="normal" if val >= 1.0 else "inverse")

    # Current policy settings summary
    st.subheader("Current Policy Settings")
    pc1, pc2, pc3, pc4 = st.columns(4)
    _level_labels = {0.0: "None", 0.2: "Minimal", 0.4: "Moderate", 0.6: "Strong", 0.8: "Intensive", 1.0: "Maximum"}
    with pc1:
        st.markdown("**Fertiliser**")
        st.caption(
            f"N: {_n_actual:.0f} kg/ha ({nitrogen_mult:.1f}x)  \n"
            f"P: {_p_actual:.0f} kg/ha ({phosphorus_mult:.1f}x)  \n"
            f"K: {_k_actual:.0f} kg/ha ({potassium_mult:.1f}x)  \n"
            f"Cost: ${_fert_cost_ha:.0f}/ha"
        )
    with pc2:
        st.markdown("**Technology**")
        st.caption(
            f"R&D boost: +{tech_boost*100:.1f}%/yr  \n"
            f"Total yield growth: {_total_tech*100:.1f}%/yr  \n"
            f"30-yr cumulative: +{((1+_total_tech)**30 - 1)*100:.0f}%"
        )
    with pc3:
        st.markdown("**Adaptation**")
        st.caption(
            f"Climate: {_level_labels.get(climate_adapt, f'{climate_adapt:.1f}')}  \n"
            f"Conservation: {_level_labels.get(conservation, f'{conservation:.1f}')}  \n"
            f"Stress reduction: ~{_adapt_effect:.0f}%"
        )
    with pc4:
        _irr_labels = {0.0: "None", 0.2: "Slow", 0.4: "Moderate", 0.6: "Fast", 0.8: "Rapid", 1.0: "Maximum"}
        st.markdown("**Land & Water**")
        st.caption(
            f"Irrigation: {_irr_labels.get(irrigation_inv, f'{irrigation_inv:.1f}')}  \n"
            f"Land constraint: {area_constraint:.1f}x  \n"
            f"Corn max: {_corn_max/1e6:.2f}M ha"
        )

    # ── Scenario Impact Summary ──
    # Show the user what their slider settings actually changed
    _any_change = (nitrogen_mult != 1.0 or phosphorus_mult != 1.0 or potassium_mult != 1.0 or
                   tech_boost != 0.0 or climate_adapt != 0.0 or conservation != 0.0 or
                   irrigation_inv != 0.0 or area_constraint != 1.0)
    if _any_change:
        st.subheader("Scenario Impact Preview")
        st.caption(
            "Comparing your scenario vs. default settings at the **final projection year**. "
            "Changes only appear in the projection period (after 2023)."
        )
        _corn_df = results['crops'].get('maize_grain')
        if _corn_df is not None:
            _last = _corn_df.iloc[-1]
            _mid_idx = len(_corn_df) // 2
            _mid = _corn_df.iloc[_mid_idx]
            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                st.metric(f"Corn Yield ({int(_last['Year'])})",
                          f"{_last['Yield_t_per_ha']:.1f} t/ha")
            with ic2:
                st.metric(f"Corn Production ({int(_last['Year'])})",
                          fmt(_last['Production_tonnes']) + " t")
            with ic3:
                st.metric(f"Soil Health ({int(_last['Year'])})",
                          f"{_last['Soil_Health']:.3f}")
            with ic4:
                st.metric(f"Irrigation ({int(_last['Year'])})",
                          f"{_last['Irrigation_Fraction']*100:.1f}%")

    st.subheader("Climate Scenarios Comparison")
    st.dataframe(results['climate_summary'], use_container_width=True, hide_index=True)


# ============================================================
# TAB 2: CROP PRODUCTION
# ============================================================
with tab2:
    st.header("Crop Production Results")
    st.info(
        "**What changes here:** All sidebar sliders affect the **projection period** (after 2023). "
        f"Historical data (1971–2023) comes from {'Statistics Canada' if country == 'Canada' else 'FAOSTAT/NBS Nigeria'} and never changes. "
        "Try: change Nitrogen to 2.0x or Tech R&D to +2%, then look at the right side of each chart.",
        icon="💡"
    )

    for crop_key, df in results['crops'].items():
        st.subheader(crop_key.replace('_', ' ').title())

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(df, x='Year', y='Production_tonnes', title='Production (tonnes)')
            fig.update_layout(template='plotly_white')
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(df, x='Year', y='Yield_t_per_ha', title='Yield (t/ha)')
            fig.update_layout(template='plotly_white')
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.line(df, x='Year', y='Area_Harvested_ha', title='Area Harvested (ha)')
            fig.update_layout(template='plotly_white')
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.line(df, x='Year', y='Price_per_tonne', title='Price per Tonne')
            fig.update_layout(template='plotly_white')
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)

        val_key = f'validation_{crop_key}'
        if val_key in results:
            with st.expander("Validation Metrics"):
                val = results[val_key]
                if isinstance(val, dict) and 'error' not in val:
                    st.dataframe(pd.DataFrame(val).T, use_container_width=True)
                else:
                    st.info(str(val))

# ============================================================
# TAB 3: WATER & LAND
# ============================================================
with tab3:
    st.header("Water Stress & Land Dynamics")
    st.info(
        "**Responds to:** Irrigation expansion, Climate scenario (SSP), Climate adaptation. "
        "Irrigation fraction changes even in the historical period when you increase the irrigation slider. "
        "Try: set Irrigation to 'Maximum' and watch the Irrigation Coverage chart change across the full timeline.",
        icon="💧"
    )

    st.subheader("Water Stress")
    water_fig = go.Figure()
    for crop_key, df in results['crops'].items():
        water_fig.add_trace(go.Scatter(
            x=df['Year'], y=df['Water_Stress_Factor'],
            name=crop_key.replace('_', ' ').title(), mode='lines'))
    water_fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="No Stress")
    water_fig.update_layout(title='Water Stress Factor', template='plotly_white', yaxis_range=[0, 1.3])
    add_projection_boundary(water_fig)
    st.plotly_chart(water_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        irr_fig = go.Figure()
        for crop_key, df in results['crops'].items():
            irr_fig.add_trace(go.Scatter(
                x=df['Year'], y=df['Irrigation_Fraction'] * 100,
                name=crop_key.replace('_', ' ').title(), mode='lines'))
        irr_fig.update_layout(title='Irrigation Coverage (%)', template='plotly_white')
        add_projection_boundary(irr_fig)
        st.plotly_chart(irr_fig, use_container_width=True)

    with col2:
        land_fig = go.Figure()
        for crop_key, df in results['crops'].items():
            land_fig.add_trace(go.Scatter(
                x=df['Year'], y=df['Arable_Land_Available_ha'],
                name=crop_key.replace('_', ' ').title(), mode='lines'))
        land_fig.update_layout(title='Arable Land Available (ha)', template='plotly_white')
        add_projection_boundary(land_fig)
        st.plotly_chart(land_fig, use_container_width=True)

# ============================================================
# TAB 4: SOIL & NUTRIENTS (N-P-K)
# ============================================================
with tab4:
    st.header("Soil Health & Nutrient Management")
    st.info(
        "**Responds to:** Conservation practices (strongest effect on soil health & erosion), "
        "Nitrogen multiplier (excess N degrades soil — try 2.0x to see soil health drop). "
        "Soil health and erosion change across the **full timeline** (including historical). "
        "N-P-K application rates change only in the projection period.",
        icon="🌱"
    )

    for crop_key, df in results['crops'].items():
        st.subheader(crop_key.replace('_', ' ').title())

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Soil_Health'], name='Soil Health', line=dict(color='brown')))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Soil_Organic_Matter_pct'] / 5,
                                     name='SOM % (scaled /5)', line=dict(color='green', dash='dot')))
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange", annotation_text="Warning")
            fig.update_layout(title='Soil Health & Organic Matter', template='plotly_white', yaxis_range=[0, 1.2])
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Erosion_Rate_t_ha_yr'], name='Erosion', line=dict(color='red')))
            fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Soil Formation Rate")
            fig.update_layout(title='Erosion Rate (t/ha/yr)', template='plotly_white')
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Nitrogen_kg_ha'], name='N (kg/ha)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Phosphorus_kg_ha'], name='P (kg/ha)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Potassium_kg_ha'], name='K (kg/ha)', line=dict(color='purple')))
            fig.update_layout(title='N-P-K Application (kg/ha)', template='plotly_white')
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Year'], y=df['NPK_Response'], name='NPK Response (Liebig)', line=dict(color='teal')))
            fig.add_trace(go.Scatter(x=df['Year'], y=df['Soil_Yield_Multiplier'], name='Soil Yield Mult.', line=dict(color='brown', dash='dot')))
            fig.update_layout(title='Nutrient Response & Soil Multiplier', template='plotly_white', yaxis_range=[0, 1.3])
            add_projection_boundary(fig)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5: CLIMATE IMPACT
# ============================================================
with tab5:
    st.header("Climate Projections & Impact")
    st.info(
        "**Responds to:** Climate scenario (SSP) dropdown only. "
        "Temperature, precipitation, and CO₂ are driven by the IPCC pathway — "
        "fertiliser and conservation sliders do **not** affect climate. "
        "Try: switch between SSP1-2.6 (green) and SSP5-8.5 (worst case) to see dramatic differences.",
        icon="🌡️"
    )
    climate_df = results['climate']

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(climate_df, x='Year', y='Temperature_C', title='Temperature (°C)',
                      color_discrete_sequence=[COLORS[ssp_scenario]])
        fig.update_layout(template='plotly_white')
        add_projection_boundary(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(climate_df, x='Year', y='Precipitation_mm', title='Precipitation (mm)',
                      color_discrete_sequence=[COLORS[ssp_scenario]])
        fig.update_layout(template='plotly_white')
        add_projection_boundary(fig)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(climate_df, x='Year', y='CO2_ppm', title='Atmospheric CO₂ (ppm)',
                      color_discrete_sequence=[COLORS[ssp_scenario]])
        fig.update_layout(template='plotly_white')
        add_projection_boundary(fig)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        fig = px.line(climate_df, x='Year', y='Climate_Stress_Factor',
                      title='Climate Stress Factor', color_discrete_sequence=[COLORS[ssp_scenario]])
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="No Stress")
        fig.update_layout(template='plotly_white')
        add_projection_boundary(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("CO₂ Effects on Crop Nutrition")
    co2_fig = go.Figure()
    for crop_key, df in results['crops'].items():
        co2_fig.add_trace(go.Scatter(
            x=df['Year'], y=df['CO2_Fert_Factor'],
            name=f'{crop_key} — Fertilisation', mode='lines'))
        co2_fig.add_trace(go.Scatter(
            x=df['Year'], y=df['CO2_Nutrient_Factor'],
            name=f'{crop_key} — Nutrient Quality', mode='lines', line=dict(dash='dot')))
    co2_fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    co2_fig.update_layout(title='CO₂: Yield Boost vs Nutrient Degradation',
                          template='plotly_white', yaxis_title='Factor (1.0 = baseline)')
    add_projection_boundary(co2_fig)
    st.plotly_chart(co2_fig, use_container_width=True)

# ============================================================
# TAB 6: NUTRIENT SUPPLY
# ============================================================
with tab6:
    st.header("Nutrient Supply from Crops")
    st.info(
        "**Responds to:** All production-affecting sliders (Nitrogen, Tech R&D, Conservation, Irrigation) — "
        "but only in the **projection period** (right of the dotted line). "
        "Nutrient supply = production × crop nutrient profile. "
        "Historical production comes from CSV data and never changes.",
        icon="🥗"
    )

    key_nuts = ['energy_kcal', 'protein_kg', 'iron_g', 'vitamin_a_mg_rae', 'zinc_g', 'vitamin_c_g']
    nut_labels = {
        'energy_kcal': 'Energy (kcal)', 'protein_kg': 'Protein (kg)',
        'iron_g': 'Iron (g)', 'vitamin_a_mg_rae': 'Vitamin A (mg RAE)',
        'zinc_g': 'Zinc (g)', 'vitamin_c_g': 'Vitamin C (g)',
    }

    for i in range(0, len(key_nuts), 2):
        col1, col2 = st.columns(2)
        for j, col in enumerate([col1, col2]):
            if i + j < len(key_nuts):
                nut = key_nuts[i + j]
                with col:
                    fig = go.Figure()
                    for crop_key, nut_df in results['nutrients'].items():
                        if nut in nut_df.columns:
                            fig.add_trace(go.Scatter(
                                x=nut_df['Year'], y=nut_df[nut],
                                name=crop_key.replace('_', ' ').title(),
                                stackgroup='one'))
                    fig.update_layout(title=nut_labels.get(nut, nut), template='plotly_white')
                    add_projection_boundary(fig)
                    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Crop Contribution Breakdown (Final Year)")
    converter = NutrientConverter()
    for crop_key, df in results['crops'].items():
        prod = df.iloc[-1]['Production_tonnes']
        if crop_key in CROP_NUTRIENT_PROFILES:
            converter.convert_crop(crop_key, prod)
    contrib_df = converter.get_contribution_table()
    if len(contrib_df) > 0:
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 7: NUTRITIONAL GAPS
# ============================================================
with tab7:
    st.header("Nutritional Gap Analysis")
    st.markdown(f"**Population at {year_end}:** {fmt(results['population_at_end'])}")

    gap_df = results['gap_analysis']
    st.subheader(f"Nutrient Adequacy in {year_end}")

    display_gap = gap_df[['Nutrient', 'Gap_Ratio', 'Status']].copy()
    display_gap['Adequacy %'] = (display_gap['Gap_Ratio'] * 100).round(1)
    st.dataframe(display_gap, use_container_width=True, hide_index=True)

    fig = px.bar(gap_df, x='Nutrient', y='Gap_Ratio', color='Status',
                 color_discrete_map={
                     'Surplus': COLORS['success'], 'Adequate': '#3498db',
                     'Mild Deficit': COLORS['warning'], 'Moderate Deficit': '#e67e22',
                     'Severe Deficit': COLORS['danger']},
                 title='Nutrient Adequacy Ratios (1.0 = 100% requirement met)')
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Adequate")
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Nutrient Adequacy Over Time")
    gap_ts = results['gap_timeseries']
    nutrient_cols = [c for c in gap_ts.columns if c not in ['Year', 'Population']]
    fig = go.Figure()
    for col in nutrient_cols:
        fig.add_trace(go.Scatter(x=gap_ts['Year'], y=gap_ts[col], name=col, mode='lines'))
    fig.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="Adequate")
    fig.update_layout(title='Nutrient Adequacy Over Time', yaxis_title='Ratio', template='plotly_white')
    add_projection_boundary(fig)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 8: SENSITIVITY ANALYSIS
# ============================================================
_SA_SUBSYSTEM_COLORS = {
    'Soil': '#8B4513', 'Water': '#1E90FF', 'Climate': '#FF6347',
    'Crop': '#32CD32', 'Land': '#DAA520', 'Economics': '#9370DB',
}

with tab8:
    st.header("Sensitivity Analysis")
    st.markdown(
        "Each parameter is varied while others are held constant. "
        "Identifies which parameters most influence model outputs."
    )

    _sa_crop_label = "corn" if country == "Canada" else "maize"
    sa_run = st.button(f"Run Sensitivity Analysis ({_sa_crop_label})", help="Takes ~30 seconds")
    if sa_run:
        with st.spinner("Running OAT sensitivity analysis..."):
            try:
                if country == "Canada":
                    _sa_config = 'canada_corn.json'
                    _sa_baseline_key = 'canada_ontario'
                    _sa_pop_file = ('canada', 'canada_population.csv')
                    _sa_hist_file = os.path.join(DATA_DIR, 'canada', 'corn_production.csv')
                else:
                    _sa_config = 'nigeria_maize.json'
                    _sa_baseline_key = 'nigeria_north'
                    _sa_pop_file = ('nigeria', 'nigeria_population.csv')
                    _sa_hist_file = os.path.join(DATA_DIR, 'nigeria', 'maize_production.csv')

                corn_params = load_config(_sa_config)
                if corn_params is None:
                    st.error(f"Cannot load {_sa_crop_label} configuration.")
                    st.stop()
                corn_params.year_end = year_end

                baseline_cl = REGIONAL_BASELINES[_sa_baseline_key]
                climate_mgr_sa = ClimateScenarioManager(baseline_cl)
                climate_sa = climate_mgr_sa.generate_scenario(ssp_scenario, 1971, year_end)
                pop_sa = load_csv(*_sa_pop_file)

                sa_obj = SensitivityAnalyzer(
                    base_params=corn_params,
                    climate_data=climate_sa[['Year', 'Temperature_C', 'Precipitation_mm', 'GDD', 'CO2_ppm']],
                    population_data=pop_sa,
                    historical_data_path=_sa_hist_file,
                    scenario_adjustments={
                        'nitrogen_multiplier': nitrogen_mult,
                        'phosphorus_multiplier': phosphorus_mult,
                        'potassium_multiplier': potassium_mult,
                        'technology_boost': tech_boost,
                        'climate_adaptation': climate_adapt,
                        'conservation_investment': conservation,
                        'irrigation_investment': irrigation_inv,
                        'area_constraint': area_constraint,
                    },
                )
                sa_obj.run_oat(n_levels=5)
                st.session_state['sa_analyzer'] = sa_obj
            except Exception as e:
                st.error(f"Sensitivity analysis failed: {e}")

    if 'sa_analyzer' in st.session_state:
        sa = st.session_state['sa_analyzer']

        for metric_key, chart_title, max_bars, chart_h in [
            ('Production_2080', 'Production in 2080', 15, 500),
            ('Yield_2080', 'Yield in 2080', 10, 400),
        ]:
            st.subheader(f"Tornado Diagram — {chart_title}")
            try:
                tornado = sa.get_tornado_data(metric_key)
                if len(tornado) > 0:
                    fig = go.Figure()
                    for _, row in tornado.head(max_bars).iterrows():
                        fig.add_trace(go.Bar(
                            y=[row['Parameter']],
                            x=[row['High_Pct'] - row['Low_Pct']],
                            base=[row['Low_Pct']],
                            orientation='h',
                            name=row['Parameter'],
                            text=f"{row['Low_Pct']:+.1f}% to {row['High_Pct']:+.1f}%",
                            textposition='outside',
                            marker_color=_SA_SUBSYSTEM_COLORS.get(row['Subsystem'], '#808080'),
                        ))
                    fig.update_layout(
                        title=f'Parameter Sensitivity — {chart_title} (% change)',
                        xaxis_title='% Change from Baseline',
                        showlegend=False, template='plotly_white',
                        height=chart_h, yaxis=dict(autorange='reversed'),
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="black")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate tornado for {metric_key}: {e}")

        st.subheader("Parameter Rankings (by overall influence)")
        try:
            rankings = sa.get_parameter_rankings()
            if len(rankings) > 0:
                st.dataframe(
                    rankings[['Rank', 'Parameter', 'Subsystem', 'Avg_Range', 'Max_Range']],
                    use_container_width=True, hide_index=True,
                )
        except Exception as e:
            st.warning(f"Could not compute rankings: {e}")
    else:
        st.info("Click 'Run Sensitivity Analysis' above to generate results.")

# ============================================================
# TAB 9: MODEL COMPARISON
# ============================================================
with tab9:
    st.header("Model Comparison & Benchmarking")
    st.markdown(
        "Compares SD model projections against published benchmarks from "
        "IPCC AR6, AgMIP ensemble, DSSAT-CERES-Maize, and FACE experiments."
    )

    # Run comparison
    comp = ModelComparison()
    for crop_key, df in results['crops'].items():
        comp.add_sd_results(crop_key, ssp_scenario, df)

    # Yield projections comparison
    st.subheader("Yield Projections vs Published Benchmarks")
    yield_comp = comp.compare_yield_projections()
    if len(yield_comp) > 0:
        # Color code Within_Range
        st.dataframe(yield_comp[
            ['Crop', 'Scenario', 'Period', 'SD_Yield_Change_Pct',
             'Benchmark', 'Benchmark_Median_Pct', 'Benchmark_Range_Lo',
             'Benchmark_Range_Hi', 'Within_Range', 'Source']
        ], use_container_width=True, hide_index=True)

        # Visual comparison
        fig = go.Figure()
        for _, row in yield_comp.iterrows():
            color = '#27ae60' if row['Within_Range'] else '#e74c3c'
            fig.add_trace(go.Bar(
                x=[f"{row['Benchmark']}<br>{row['Period'][:12]}"],
                y=[row['SD_Yield_Change_Pct']],
                name='SD Model',
                marker_color=color,
                showlegend=False,
            ))
            # Benchmark range
            fig.add_trace(go.Scatter(
                x=[f"{row['Benchmark']}<br>{row['Period'][:12]}",
                   f"{row['Benchmark']}<br>{row['Period'][:12]}"],
                y=[row['Benchmark_Range_Lo'], row['Benchmark_Range_Hi']],
                mode='markers+lines',
                marker=dict(size=10, symbol='line-ew-open', color='black'),
                line=dict(width=3, color='black'),
                showlegend=False,
            ))
        fig.update_layout(
            title='SD Model vs Published Benchmarks (yield % change)',
            yaxis_title='Yield Change (%)',
            template='plotly_white', height=400,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No benchmark comparison data available for this scenario.")

    # CO₂ nutrient effects
    st.subheader("CO₂ Nutrient Degradation vs FACE Experiments")
    co2_comp = comp.compare_co2_nutrient_effects()
    if len(co2_comp) > 0:
        st.dataframe(co2_comp, use_container_width=True, hide_index=True)

    # Scorecard
    st.subheader("Model Performance Scorecard")
    scorecard = comp.get_scorecard()
    if len(scorecard) > 0:
        st.dataframe(scorecard, use_container_width=True, hide_index=True)

    # Reference benchmarks table
    with st.expander("Full Benchmark Reference Table"):
        ref_table = ModelComparison.get_benchmark_table()
        st.dataframe(ref_table, use_container_width=True, hide_index=True)

    # Historical validation
    st.subheader("Historical Validation Statistics")
    _crop_file_map = {
        'maize_grain': 'corn_production.csv' if country == 'Canada' else 'maize_production.csv',
        'tomato': 'tomato_production.csv',
        'leafy_greens': 'leafy_greens_production.csv',
        'cassava': 'cassava_production.csv',
        'rice_paddy': 'rice_production.csv',
        'sorghum': 'sorghum_production.csv',
    }
    for crop_key, df in results['crops'].items():
        hist_path = os.path.join(DATA_DIR, country.lower(),
            _crop_file_map.get(crop_key, ''))
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            stats = comp.compute_validation_stats(df, hist_df)
            if 'error' not in stats:
                st.markdown(f"**{crop_key.replace('_', ' ').title()}**")
                stats_display = {k: v for k, v in stats.items()}
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R²", f"{stats['R2']:.4f}")
                with col2:
                    st.metric("MAPE", f"{stats['MAPE_pct']:.1f}%")
                with col3:
                    st.metric("NSE", f"{stats['NSE']:.4f}")
                with col4:
                    st.metric("Willmott d", f"{stats['Willmott_d']:.4f}")
                with st.expander(f"Full validation stats for {crop_key}"):
                    st.json(stats)

# ============================================================
# TAB 10: POLICY INTERVENTIONS
# ============================================================
with tab10:
    st.header("Recommended Policy Interventions")

    interventions = results['interventions']
    if len(interventions) == 0:
        st.success("All nutrients at adequate levels!")
    else:
        st.warning(f"**{len(interventions)} nutrient(s) below adequate levels**")
        for intv in interventions:
            icon = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(intv['priority'], '⚪')
            st.markdown(f"""
            ### {icon} {intv['nutrient']} — {intv['status']}
            - **Current adequacy:** {intv['current_ratio']:.1%}
            - **Priority:** {intv['priority']}
            - **Recommended crops:** {', '.join(intv['recommended_crops'])}
            """)

    st.markdown("---")
    st.subheader("Policy Lever Guide")
    st.markdown("""
    | Sidebar Control | Model Effect | Real-World Meaning | Best For |
    |----------------|-------------|-------------------|----------|
    | **Nitrogen (N)** 0.5x–2.0x | Multiplies baseline N (corn: 50 kg/ha in 1971) | 25–100 kg N/ha. Cost ~$1.10/kg | Short-term yield boost |
    | **Phosphorus (P)** 0.5x–2.0x | Multiplies baseline P (corn: 22 kg/ha) | 11–44 kg P/ha. Cost ~$2.80/kg | Prevents P limitation |
    | **Potassium (K)** 0.5x–2.0x | Multiplies baseline K (corn: 35 kg/ha) | 18–70 kg K/ha. Cost ~$0.75/kg | Prevents K limitation |
    | **R&D investment** 0–2% | Additional annual yield growth on top of base 0.8%/yr | Improved seeds, precision agriculture, biotech | Long-term productivity |
    | **Climate adaptation** None–Max | Reduces climate stress impact by 0–50% | Drought-tolerant varieties, adjusted planting dates, heat shelters | Maintaining yield under warming |
    | **Conservation** None–Max | Reduces erosion by 0–50%, boosts soil regeneration | No-till, cover crops, organic amendments, crop rotation | Soil health, long-term sustainability |
    | **Irrigation expansion** None–Max | Expands irrigated fraction by 0–0.5%/yr | New wells, drip systems, water storage infrastructure | Water-limited crops & regions |
    | **Land use policy** 0.5x–1.5x | Constrains or expands max farmland area | Protection of forests/wetlands vs opening new land | Land-use balance |
    """)

# ============================================================
# TAB 11: DATA TABLES
# ============================================================
with tab11:
    st.header("Raw Data Tables")

    selection = st.selectbox("Select Dataset",
        ["Crop Production", "Climate Projections", "Nutrient Supply",
         "Gap Analysis", "Population"])

    if selection == "Crop Production":
        for crop_key, df in results['crops'].items():
            st.subheader(crop_key.replace('_', ' ').title())
            st.dataframe(df, use_container_width=True, hide_index=True)
    elif selection == "Climate Projections":
        st.dataframe(results['climate'], use_container_width=True, hide_index=True)
    elif selection == "Nutrient Supply":
        for crop_key, df in results['nutrients'].items():
            st.subheader(crop_key.replace('_', ' ').title())
            st.dataframe(df, use_container_width=True, hide_index=True)
    elif selection == "Gap Analysis":
        st.dataframe(results['gap_analysis'], use_container_width=True, hide_index=True)
    elif selection == "Population":
        _pop_file = 'canada_population.csv' if country == 'Canada' else 'nigeria_population.csv'
        pop_df = load_csv(country.lower(), _pop_file)
        if pop_df is not None:
            st.dataframe(pop_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Export")
    for crop_key, df in results['crops'].items():
        csv = df.to_csv(index=False)
        st.download_button(
            f"Download {crop_key.replace('_', ' ').title()}",
            csv,
            f"{crop_key}_results.csv",
            "text/csv",
            key=f"dl_{crop_key}",
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Climate-Nutrition Modelling Framework v2.1 | "
    "7 Subsystems | Data: StatCan, FAOSTAT, IPCC AR6, WHO | "
    f"Scenario: {ssp_scenario.upper()} | Benchmarked: IPCC, AgMIP, DSSAT, FACE"
)
