"""
Generic Crop Production Model using System Dynamics — v2.0

Seven interconnected subsystems modelling crop production under climate change:
  1. LAND    — arable area, urbanisation, land development
  2. WATER   — crop water requirement, irrigation, water stress
  3. SOIL    — N-P-K dynamics, organic matter, erosion (USLE-inspired)
  4. CLIMATE — temperature, precipitation, GDD, CO₂ fertilisation & nutrient degradation
  5. CROP    — yield determined by Liebig's minimum across all stresses
  6. ECONOMICS — input costs (N/P/K), revenue, profit, farmer behaviour
  7. DEMAND  — domestic (food, feed, industrial), export, population-driven

Theoretical foundations:
  • Forrester (1961) / Sterman (2000) — System Dynamics methodology
  • Mitscherlich response curves — diminishing fertiliser returns
  • Liebig's Law of the Minimum — yield limited by scarcest factor
  • USLE-inspired erosion — simplified Universal Soil Loss Equation
  • IPCC AR6 WGII Ch5 — CO₂-induced nutrient dilution in crops
  • Zhu et al. (2025) Nature Climate Change — climate-crop-nutrient modelling
  • Ogunleye et al. (2025) JHPN — SD analysis of agriculture & food security
  • AgMIP protocols — climate-crop model coupling

Author: Climate-Nutrition Modelling Project
License: MIT
"""

import numpy as np
import pandas as pd
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ================================================================
# CropParameters — complete configuration for one crop × one region
# ================================================================

@dataclass
class CropParameters:
    """
    Full parameter set for a crop model instance.

    Parameters are grouped by subsystem.  Every field has a sensible
    default so that a bare ``CropParameters()`` still produces a valid
    (if generic) simulation.
    """

    # ---- Identity ------------------------------------------------
    crop_name: str = "generic_crop"
    country: str = "generic"
    unit: str = "tonnes"

    # ---- Time ----------------------------------------------------
    year_start: int = 1971
    year_end: int = 2100
    dt: float = 1.0  # years

    # ---- Initial conditions (year_start values) ------------------
    initial_area_ha: float = 1_000_000
    initial_yield_t_per_ha: float = 3.0
    initial_price_per_tonne: float = 200.0
    initial_soil_health: float = 0.85
    initial_machinery_capital: float = 500.0  # millions local currency

    # ---- LAND SUBSYSTEM ------------------------------------------
    total_arable_land_ha: float = 5_000_000      # national/regional total
    urban_encroachment_rate: float = 0.001        # fraction lost / year to urbanisation
    land_development_rate: float = 0.002          # fraction developed / year (clearing, irrigation)
    land_development_ceiling: float = 0.95        # max fraction of arable that can be cultivated
    area_price_elasticity: float = 0.3
    area_max_ha: float = 5_000_000
    area_min_ha: float = 100_000
    area_adjustment_delay: int = 2

    # ---- WATER SUBSYSTEM -----------------------------------------
    crop_water_requirement_mm: float = 500.0      # seasonal water need
    irrigation_fraction: float = 0.15             # fraction of area irrigated
    irrigation_efficiency: float = 0.60           # how much irrigation water reaches root
    rainfed_water_capture: float = 0.70           # fraction of rainfall usable
    water_stress_threshold: float = 0.60          # below this ratio → stress
    water_stress_sensitivity: float = 1.5         # exponent of water-stress curve
    max_irrigation_expansion_rate: float = 0.005  # fraction / year

    # ---- SOIL SUBSYSTEM ------------------------------------------
    # Nitrogen
    initial_nitrogen_kg_ha: float = 80.0
    nitrogen_response_coeff: float = 0.015
    nitrogen_diminishing_point: float = 150.0
    nitrogen_loss_rate: float = 0.30              # leaching + volatilisation fraction
    nitrogen_cost_per_kg: float = 1.2             # USD / kg

    # Phosphorus
    initial_phosphorus_kg_ha: float = 25.0
    phosphorus_response_coeff: float = 0.020
    phosphorus_diminishing_point: float = 60.0
    phosphorus_soil_stock_kg_ha: float = 500.0    # total soil P stock
    phosphorus_loss_rate: float = 0.05            # erosion-driven loss fraction
    phosphorus_cost_per_kg: float = 2.5

    # Potassium
    initial_potassium_kg_ha: float = 40.0
    potassium_response_coeff: float = 0.012
    potassium_diminishing_point: float = 80.0
    potassium_loss_rate: float = 0.15
    potassium_cost_per_kg: float = 0.8

    # Soil organic matter & health
    initial_som_pct: float = 3.5                  # soil organic matter %
    som_decomposition_rate: float = 0.02          # fraction / year
    som_input_rate: float = 0.015                 # crop residue return
    som_yield_multiplier_min: float = 0.50
    som_yield_multiplier_max: float = 1.10

    # Erosion (USLE-inspired)
    erosion_base_rate_t_ha_yr: float = 5.0        # baseline annual erosion t/ha
    erosion_rainfall_factor: float = 0.001         # erosion increase per mm rainfall
    erosion_slope_factor: float = 1.0             # topography multiplier
    erosion_cover_factor: float = 0.40            # crop cover management (lower = better)
    erosion_practice_factor: float = 1.0          # conservation practice (lower = better)
    soil_formation_rate_t_ha_yr: float = 1.0      # natural soil formation

    soil_degradation_rate: float = 0.005
    soil_regeneration_rate: float = 0.01
    soil_nitrogen_sensitivity: float = 0.0001

    # ---- CLIMATE SUBSYSTEM ---------------------------------------
    optimal_temperature: float = 25.0
    temperature_sensitivity: float = 0.05
    optimal_precipitation: float = 600.0
    precipitation_sensitivity: float = 0.001
    gdd_base: float = 10.0
    gdd_optimal: float = 2500.0
    heat_stress_threshold: float = 35.0

    # CO₂ effects
    co2_fertilisation_beta: float = 0.0008        # yield gain per ppm above 400
    co2_protein_degradation: float = -0.00015     # protein loss per ppm above 400
    co2_iron_degradation: float = -0.00015        # iron loss per ppm above 400
    co2_zinc_degradation: float = -0.00010        # zinc loss per ppm above 400
    co2_reference_ppm: float = 400.0

    # ---- CROP SUBSYSTEM ------------------------------------------
    max_attainable_yield: float = 12.0
    technology_growth_rate: float = 0.005
    harvest_index: float = 0.50                   # grain fraction of total biomass
    residue_return_fraction: float = 0.50         # crop residues returned to soil

    # ---- ECONOMICS -----------------------------------------------
    input_cost_growth_rate: float = 0.02
    price_floor: float = 50.0
    price_ceiling: float = 1000.0
    demand_price_elasticity: float = -0.3
    supply_demand_price_adjustment: float = 0.1
    machinery_depreciation_rate: float = 0.08
    machinery_investment_rate: float = 0.12
    machinery_yield_effect: float = 0.0005

    # ---- DEMAND --------------------------------------------------
    base_domestic_demand: float = 5_000_000
    feed_demand_fraction: float = 0.50
    food_demand_fraction: float = 0.15
    industrial_demand_fraction: float = 0.10
    export_demand_fraction: float = 0.25
    demand_population_elasticity: float = 0.8
    demand_income_elasticity: float = 0.3

    # ---- I/O helpers ---------------------------------------------
    @classmethod
    def from_json(cls, filepath: str) -> 'CropParameters':
        """Load parameters from a JSON config file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_json(self, filepath: str):
        from dataclasses import asdict
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ================================================================
# CropModel — 7-subsystem system-dynamics simulation
# ================================================================

class CropModel:
    """
    Generic system-dynamics crop production model with seven subsystems.

    Feedback loops (all closed and endogenous)
    ──────────────────────────────────────────
    B1. Price → Area → Supply → Price  (balancing: market equilibrium)
    B2. Intensive N → Soil degradation → Yield ↓ → Profit ↓  (balancing: soil)
    R1. Profit → NPK investment → Yield ↑ → Profit ↑  (reinforcing: input)
    R2. Revenue → Machinery → Yield ↑ → Revenue ↑  (reinforcing: capital)
    B3. Water stress → Irrigation expansion → Water supply ↑  (balancing: water)
    B4. Erosion → P soil-stock depletion → P limitation → Yield ↓  (balancing: erosion-nutrient)
    ---  Exogenous forcings (one-directional, not feedback loops)  ---
    F1. Climate change → Temperature / water stress → Yield  (forcing)
    F2. CO₂ rise → Fertilisation ↑ but nutrient dilution ↓  (forcing, trade-off)

    Usage::

        params = CropParameters.from_json('config/canada_corn.json')
        model  = CropModel(params)
        model.load_historical_data('data/canada/corn_production.csv')
        model.climate_data = climate_df
        model.population_data = pop_df
        model.set_scenario(nitrogen_multiplier=1.2)
        results = model.run()
    """

    def __init__(self, params: CropParameters):
        self.params = params
        self.n_steps = int((params.year_end - params.year_start) / params.dt) + 1
        self.time = np.arange(params.year_start, params.year_end + params.dt, params.dt)

        # External data holders
        self.historical_data: Optional[pd.DataFrame] = None
        self.climate_data: Optional[pd.DataFrame] = None
        self.population_data: Optional[pd.DataFrame] = None

        # Scenario adjustments
        self.scenario_adjustments: Dict[str, float] = {}
        self._has_run = False

        # Calibration: running calibration factor is computed each historical
        # step as the ratio of observed yield to model yield.  This creates
        # a smooth time-varying correction rather than a single jump at the
        # transition.  The factor is exponentially smoothed (alpha=0.3) to
        # filter noise while tracking structural shifts.
        self._yield_calibration_factor: float = 1.0
        self._transition_k: Optional[int] = None  # index of first projection step
        self._shadow_yield: Optional[np.ndarray] = None  # model yield during history

        self._init_state_variables()

    # ────────────────────────────────────────────────────────────
    # State initialisation
    # ────────────────────────────────────────────────────────────

    def _init_state_variables(self):
        n = self.n_steps
        p = self.params

        # LAND
        self.area_harvested = np.zeros(n)
        self.arable_land_available = np.zeros(n)

        # WATER
        self.water_supply_mm = np.zeros(n)
        self.water_demand_mm = np.zeros(n)
        self.water_stress_factor = np.zeros(n)
        self.irrigation_fraction = np.zeros(n)

        # SOIL
        self.nitrogen_use = np.zeros(n)
        self.phosphorus_use = np.zeros(n)
        self.potassium_use = np.zeros(n)
        self.soil_organic_matter = np.zeros(n)
        self.soil_health = np.zeros(n)
        self.erosion_rate = np.zeros(n)
        self.soil_depth_equiv = np.zeros(n)  # cumulative erosion budget (t/ha)
        self.p_soil_stock = np.zeros(n)

        # CLIMATE
        self.climate_stress = np.zeros(n)
        self.co2_ppm = np.zeros(n)
        self.co2_fert_factor = np.zeros(n)
        self.co2_nutrient_factor = np.zeros(n)  # degradation multiplier on protein/Fe/Zn

        # CROP
        self.crop_yield = np.zeros(n)
        self.production = np.zeros(n)
        self.technology_factor = np.zeros(n)
        self.npk_response = np.zeros(n)
        self.soil_yield_multiplier = np.zeros(n)

        # ECONOMICS
        self.price = np.zeros(n)
        self.revenue = np.zeros(n)
        self.input_cost = np.zeros(n)
        self.profit_margin = np.zeros(n)
        self.machinery_capital = np.zeros(n)

        # DEMAND
        self.total_demand = np.zeros(n)
        self.feed_demand = np.zeros(n)
        self.food_demand = np.zeros(n)
        self.industrial_demand = np.zeros(n)
        self.export_demand = np.zeros(n)
        self.demand_satisfaction = np.zeros(n)

        # ── Initial conditions ──
        self.area_harvested[0] = p.initial_area_ha
        self.arable_land_available[0] = p.total_arable_land_ha
        self.crop_yield[0] = p.initial_yield_t_per_ha
        self.production[0] = p.initial_area_ha * p.initial_yield_t_per_ha
        self.price[0] = p.initial_price_per_tonne
        self.nitrogen_use[0] = p.initial_nitrogen_kg_ha
        self.phosphorus_use[0] = p.initial_phosphorus_kg_ha
        self.potassium_use[0] = p.initial_potassium_kg_ha
        self.soil_organic_matter[0] = p.initial_som_pct
        self.soil_health[0] = p.initial_soil_health
        self.machinery_capital[0] = p.initial_machinery_capital
        self.technology_factor[0] = 1.0
        self.climate_stress[0] = 1.0
        self.water_stress_factor[0] = 1.0
        self.irrigation_fraction[0] = p.irrigation_fraction
        self.p_soil_stock[0] = p.phosphorus_soil_stock_kg_ha
        self.soil_depth_equiv[0] = 0.0
        self.co2_ppm[0] = 330.0
        self.co2_fert_factor[0] = 1.0
        self.co2_nutrient_factor[0] = 1.0
        self.npk_response[0] = 1.0
        self.soil_yield_multiplier[0] = self._calc_soil_yield_mult(p.initial_soil_health)

    # ────────────────────────────────────────────────────────────
    # Data loaders
    # ────────────────────────────────────────────────────────────

    def load_historical_data(self, filepath: str):
        """Load historical crop data (CSV with Year, Area, Yield, Price, …)."""
        if not os.path.exists(filepath):
            print(f"Warning: Historical data not found: {filepath}")
            return
        self.historical_data = pd.read_csv(filepath)
        first = self.historical_data.iloc[0]
        p = self.params
        if 'Area_Harvested_ha' in self.historical_data.columns:
            p.initial_area_ha = first['Area_Harvested_ha']
            self.area_harvested[0] = p.initial_area_ha
        if 'Yield_tonne_per_ha' in self.historical_data.columns:
            p.initial_yield_t_per_ha = first['Yield_tonne_per_ha']
            self.crop_yield[0] = p.initial_yield_t_per_ha
        if 'Price_per_tonne' in self.historical_data.columns:
            p.initial_price_per_tonne = first['Price_per_tonne']
            self.price[0] = p.initial_price_per_tonne
        self.production[0] = self.area_harvested[0] * self.crop_yield[0]
        print(f"  Loaded historical data: {len(self.historical_data)} years")

    def load_climate_scenarios(self, filepath: str):
        if filepath is None or not os.path.exists(filepath):
            if filepath is not None:
                print(f"Warning: Climate file not found: {filepath}")
            return
        self.climate_data = pd.read_csv(filepath)

    def load_population_projection(self, filepath: str):
        if filepath is None or not os.path.exists(filepath):
            if filepath is not None:
                print(f"Warning: Population file not found: {filepath}")
            return
        self.population_data = pd.read_csv(filepath)

    def set_scenario(self, **kwargs):
        """
        Supported keys:
            nitrogen_multiplier, phosphorus_multiplier, potassium_multiplier,
            technology_boost, climate_adaptation, conservation_investment,
            area_constraint, price_support, irrigation_investment
        """
        self.scenario_adjustments = kwargs

    # ────────────────────────────────────────────────────────────
    # Subsystem calculations
    # ────────────────────────────────────────────────────────────

    def _get_climate_at_year(self, year: float) -> Tuple[float, float, float, float]:
        """Return (temp, precip, gdd, co2) for *year*."""
        p = self.params
        if self.climate_data is not None and year in self.climate_data['Year'].values:
            row = self.climate_data[self.climate_data['Year'] == year].iloc[0]
            temp = row.get('Temperature_C', p.optimal_temperature)
            precip = row.get('Precipitation_mm', p.optimal_precipitation)
            gdd = row.get('GDD', p.gdd_optimal)
            co2 = row.get('CO2_ppm', 400.0)
            return temp, precip, gdd, co2
        # Fallback: gradual warming
        yr_offset = year - p.year_start
        temp = p.optimal_temperature + 0.02 * yr_offset
        precip = p.optimal_precipitation * (1 - 0.001 * yr_offset)
        gdd = p.gdd_optimal + 5 * yr_offset
        co2 = 330 + 1.8 * yr_offset
        return temp, precip, gdd, co2

    def _get_population_at_year(self, year: float) -> float:
        if self.population_data is not None:
            if year in self.population_data['Year'].values:
                return self.population_data[self.population_data['Year'] == year].iloc[0]['Population']
            return float(np.interp(year,
                                   self.population_data['Year'],
                                   self.population_data['Population']))
        return 30_000_000 * (1.01 ** (year - self.params.year_start))

    # --- 1. WATER STRESS ---
    def _calc_water_stress(self, precip: float, k: int) -> float:
        """Water stress factor ∈ [0.2, 1.1].

        FEEDBACK LOOP (closed):
            Water stress → irrigation expansion → improved water supply → reduced stress.
            Irrigation investment responds *endogenously* to water stress severity,
            plus any exogenous scenario boost.  This closes the water-scarcity loop
            that was previously open (irrigation only grew from scenario parameters).
        """
        p = self.params
        irr_inv = self.scenario_adjustments.get('irrigation_investment', 0.0)

        # ── Endogenous irrigation response to water stress ──
        # When crops are water-stressed, farmers invest in irrigation.
        # The expansion rate is proportional to the severity of stress
        # experienced in the previous period (information delay of 1 year).
        if k > 0:
            prev_stress = self.water_stress_factor[k - 1]
            # Stress-driven expansion: more stress → faster irrigation growth
            # Only triggers below stress threshold of 0.9 (mild stress starts response)
            stress_driven_expansion = 0.0
            if prev_stress < 0.9:
                stress_severity = max(0.0, 0.9 - prev_stress)  # 0 to ~0.7
                stress_driven_expansion = p.max_irrigation_expansion_rate * stress_severity * 2.0
            # Exogenous policy boost (scenario parameter) adds on top
            policy_expansion = p.max_irrigation_expansion_rate * irr_inv
            self.irrigation_fraction[k] = min(
                0.95,
                self.irrigation_fraction[k - 1] + stress_driven_expansion + policy_expansion
            )
        else:
            self.irrigation_fraction[k] = p.irrigation_fraction

        rainfed_supply = precip * p.rainfed_water_capture * (1 - self.irrigation_fraction[k])
        irrigated_supply = p.crop_water_requirement_mm * self.irrigation_fraction[k] * p.irrigation_efficiency
        total_supply = rainfed_supply + irrigated_supply
        demand = p.crop_water_requirement_mm

        self.water_supply_mm[k] = total_supply
        self.water_demand_mm[k] = demand

        ratio = total_supply / max(demand, 1.0)
        if ratio >= 1.0:
            return min(1.1, 1.0 + 0.05 * (ratio - 1.0))  # slight benefit from excess
        elif ratio >= p.water_stress_threshold:
            return 1.0 - (1.0 - ratio) ** p.water_stress_sensitivity
        else:
            return max(0.2, ratio / p.water_stress_threshold * 0.6)

    # --- 2. SOIL EROSION (USLE-inspired) ---
    def _calc_soil_erosion(self, precip: float, k: int) -> float:
        """Annual erosion rate (t / ha / yr)."""
        p = self.params
        conserv = self.scenario_adjustments.get('conservation_investment', 0.0)

        R = p.erosion_base_rate_t_ha_yr + p.erosion_rainfall_factor * max(0, precip - 400)
        practice = p.erosion_practice_factor * max(0.3, 1.0 - 0.5 * conserv)
        cover = p.erosion_cover_factor
        erosion = R * p.erosion_slope_factor * cover * practice
        return max(0.0, erosion)

    # --- 3. SOIL ORGANIC MATTER ---
    def _calc_som(self, k: int, production: float) -> float:
        """Soil organic matter % after one year."""
        p = self.params
        prev_som = self.soil_organic_matter[k - 1] if k > 0 else p.initial_som_pct
        residue_input = p.som_input_rate * (production / max(self.area_harvested[k], 1)) * p.residue_return_fraction
        decomp = p.som_decomposition_rate * prev_som
        conserv = self.scenario_adjustments.get('conservation_investment', 0.0)
        new_som = prev_som - decomp + residue_input + 0.005 * conserv
        return np.clip(new_som, 0.5, 8.0)

    # --- 4. N-P-K NUTRIENT DYNAMICS ---
    def _calc_npk_step(self, k: int) -> Tuple[float, float, float]:
        """Update fertiliser application based on *smoothed* profitability.

        FEEDBACK LOOP (reinforcing / balancing):
            Profit ↑ → more fertiliser → higher yield → more production →
            higher revenue → higher profit (reinforcing).
            BUT: excess N → soil degradation → yield decline (balancing).

        Uses a 3-year exponential moving average of profit margin (first-order
        information delay) so that farmers respond to trends, not noise.
        This is a standard System Dynamics pattern (Sterman 2000, Ch. 11).

        Returns (N, P, K) in kg/ha.
        """
        p = self.params
        n_mult = self.scenario_adjustments.get('nitrogen_multiplier', 1.0)
        p_mult = self.scenario_adjustments.get('phosphorus_multiplier', 1.0)
        k_mult = self.scenario_adjustments.get('potassium_multiplier', 1.0)

        if k == 0:
            return (p.initial_nitrogen_kg_ha * n_mult,
                    p.initial_phosphorus_kg_ha * p_mult,
                    p.initial_potassium_kg_ha * k_mult)

        # ── Smoothed profit signal (3-year first-order delay) ──
        # Farmers don't react to a single bad/good year — they respond to
        # the perceived trend over ~3 years (information + decision delay).
        delay_years = 3
        lookback = max(0, k - delay_years)
        smoothed_profit = np.mean(self.profit_margin[lookback:k])

        # Adjustment rate: proportional to smoothed profit, with asymmetry
        # (farmers increase inputs faster than they decrease — loss aversion)
        if smoothed_profit > 0:
            change = 0.015 * smoothed_profit  # cautious increase
        else:
            change = 0.008 * smoothed_profit  # slow decrease (loss aversion)

        n_app = np.clip(self.nitrogen_use[k - 1] * (1 + change) * n_mult, 0, 300)
        p_app = np.clip(self.phosphorus_use[k - 1] * (1 + change * 0.5) * p_mult, 0, 100)
        k_app = np.clip(self.potassium_use[k - 1] * (1 + change * 0.5) * k_mult, 0, 150)
        return n_app, p_app, k_app

    def _mitscherlich(self, application: float, coeff: float, diminishing: float) -> float:
        """Mitscherlich response curve with diminishing returns."""
        if application <= 0:
            return 0.5
        response = 1.0 - 0.5 * np.exp(-coeff * application)
        if application > diminishing:
            excess = application - diminishing
            response -= 0.0005 * excess
        return np.clip(response, 0.3, 1.2)

    def _calc_npk_response(self, n: float, p_app: float, k_app: float,
                           step: int = 0) -> float:
        """Liebig's Law: yield limited by most deficient nutrient.

        FEEDBACK LOOP (closed):
            Erosion → phosphorus soil-stock depletion → reduced P availability
            → lower NPK response → lower yield → lower production → lower
            residue return → less SOM → more erosion.

            The P soil stock acts as a *buffer*: even if a farmer applies
            phosphorus fertiliser, a depleted soil stock reduces the
            effective availability (sorption, fixation).  This closes the
            erosion–nutrient loop that was previously tracked but not used.
        """
        pr = self.params
        n_resp = self._mitscherlich(n, pr.nitrogen_response_coeff, pr.nitrogen_diminishing_point)
        p_resp = self._mitscherlich(p_app, pr.phosphorus_response_coeff, pr.phosphorus_diminishing_point)
        k_resp = self._mitscherlich(k_app, pr.potassium_response_coeff, pr.potassium_diminishing_point)

        # ── Erosion–P feedback: soil stock depletion reduces P effectiveness ──
        # When erosion depletes the soil phosphorus stock, even applied P is
        # less effective (poor sorption capacity in degraded soils).
        if step > 0 and self.p_soil_stock[step] > 0:
            p_stock_ratio = self.p_soil_stock[step] / pr.phosphorus_soil_stock_kg_ha
            # Clamp to [0.4, 1.0] — below 40% of original stock, P is severely limited
            p_stock_factor = np.clip(p_stock_ratio, 0.4, 1.0)
            p_resp *= p_stock_factor

        return min(n_resp, p_resp, k_resp)  # Liebig's minimum

    # --- 5. SOIL HEALTH COMPOSITE ---
    def _calc_soil_yield_mult(self, soil_health: float) -> float:
        p = self.params
        return p.som_yield_multiplier_min + (
            (p.som_yield_multiplier_max - p.som_yield_multiplier_min) * soil_health
        )

    def _calc_soil_health(self, k: int, erosion: float, som: float) -> float:
        """Composite soil health index ∈ [0, 1]."""
        p = self.params
        if k == 0:
            return p.initial_soil_health

        prev = self.soil_health[k - 1]

        # Erosion degrades soil
        erosion_impact = -0.001 * max(0, erosion - p.soil_formation_rate_t_ha_yr)

        # SOM supports health
        som_effect = 0.01 * (som - 2.0) / 3.0  # normalised around typical SOM

        # N over-use degrades
        n_degradation = -p.soil_nitrogen_sensitivity * self.nitrogen_use[k]

        # Conservation boost
        conserv = self.scenario_adjustments.get('conservation_investment', 0.0)
        conserv_boost = p.soil_regeneration_rate * conserv * 0.5

        new_health = prev + erosion_impact + som_effect + n_degradation + conserv_boost
        return np.clip(new_health, 0, 1)

    # --- 6. CLIMATE STRESS ---
    def _calc_climate_stress(self, temp: float, precip: float, gdd: float) -> float:
        p = self.params
        adaptation = self.scenario_adjustments.get('climate_adaptation', 0.0)

        temp_dev = abs(temp - p.optimal_temperature)
        temp_stress = max(0, 1.0 - p.temperature_sensitivity * temp_dev ** 1.5)

        precip_dev = abs(precip - p.optimal_precipitation)
        precip_stress = max(0, 1.0 - p.precipitation_sensitivity * precip_dev)

        gdd_ratio = min(gdd / p.gdd_optimal, 1.3) if p.gdd_optimal > 0 else 1.0
        gdd_effect = min(1.1, gdd_ratio)

        heat_penalty = 1.0
        if temp > p.heat_stress_threshold:
            excess = temp - p.heat_stress_threshold
            heat_penalty = max(0.5, 1.0 - 0.05 * excess)

        raw = temp_stress * precip_stress * gdd_effect * heat_penalty
        adapted = raw + adaptation * (1.0 - raw) * 0.5
        return np.clip(adapted, 0.3, 1.15)

    # --- 7. CO₂ EFFECTS ---
    def _calc_co2_effects(self, co2: float) -> Tuple[float, float]:
        """Return (fertilisation_factor, nutrient_degradation_factor)."""
        p = self.params
        delta = max(0, co2 - p.co2_reference_ppm)
        fert = 1.0 + p.co2_fertilisation_beta * delta
        # Nutrient degradation: protein −6%, iron −6%, zinc −4% per doubling
        # We store a single multiplier (average effect)
        avg_degrad = 1.0 + ((p.co2_protein_degradation + p.co2_iron_degradation + p.co2_zinc_degradation) / 3.0) * delta
        return np.clip(fert, 1.0, 1.30), np.clip(avg_degrad, 0.70, 1.0)

    # --- 8. LAND DYNAMICS ---
    def _calc_land_available(self, k: int) -> float:
        p = self.params
        if k == 0:
            return p.total_arable_land_ha
        prev = self.arable_land_available[k - 1]
        urbanisation_loss = prev * p.urban_encroachment_rate
        development_gain = (p.total_arable_land_ha - prev) * p.land_development_rate
        new_land = prev - urbanisation_loss + development_gain
        return max(p.area_min_ha, min(new_land, p.total_arable_land_ha))

    def _calc_farmer_area_response(self, profit_margin: float, k: int) -> float:
        p = self.params
        delay = min(k, p.area_adjustment_delay)
        if delay > 0 and k >= delay:
            avg_profit = np.mean(self.profit_margin[max(0, k - delay):k + 1])
        else:
            avg_profit = profit_margin

        area_change_rate = p.area_price_elasticity * np.tanh(avg_profit * 2)
        constraint = self.scenario_adjustments.get('area_constraint', 1.0)

        current = self.area_harvested[k - 1] if k > 0 else p.initial_area_ha
        new_area = current * (1 + area_change_rate)
        max_possible = min(p.area_max_ha * constraint,
                           self.arable_land_available[k] * p.land_development_ceiling)
        return np.clip(new_area, p.area_min_ha, max_possible)

    # --- 9. DEMAND ---
    def _calc_demand(self, year: float, k: int) -> Dict[str, float]:
        p = self.params
        pop = self._get_population_at_year(year)
        pop_ratio = pop / self._get_population_at_year(p.year_start)
        years_elapsed = year - p.year_start
        income_growth = (1 + 0.02) ** years_elapsed

        feed = p.base_domestic_demand * p.feed_demand_fraction * (pop_ratio ** p.demand_population_elasticity)
        food = p.base_domestic_demand * p.food_demand_fraction * (pop_ratio ** p.demand_population_elasticity)
        industrial = (p.base_domestic_demand * p.industrial_demand_fraction *
                      (pop_ratio ** p.demand_population_elasticity) *
                      (income_growth ** p.demand_income_elasticity))
        export = p.base_domestic_demand * p.export_demand_fraction * (1 + 0.01 * years_elapsed)

        if k > 0 and self.price[0] > 0:
            price_ratio = max(self.price[k], 0.01) / self.price[0]
            demand_adj = price_ratio ** p.demand_price_elasticity
        else:
            demand_adj = 1.0

        total = (feed + food + industrial + export) * demand_adj
        return {
            'feed': feed * demand_adj,
            'food': food * demand_adj,
            'industrial': industrial * demand_adj,
            'export': export * demand_adj,
            'total': total,
        }

    # ────────────────────────────────────────────────────────────
    # Main simulation step
    # ────────────────────────────────────────────────────────────

    def _step(self, k: int):
        p = self.params
        year = self.time[k]

        # ── Check for historical data ──
        use_historical = False
        if self.historical_data is not None:
            hist_match = self.historical_data[self.historical_data['Year'] == int(year)]
            if len(hist_match) > 0:
                use_historical = True
                row = hist_match.iloc[0]

        # ── Climate (always computed) ──
        temp, precip, gdd, co2 = self._get_climate_at_year(year)
        self.co2_ppm[k] = co2

        if use_historical and k > 0:
            # Use actual data for historical period
            if 'Area_Harvested_ha' in row.index:
                self.area_harvested[k] = row['Area_Harvested_ha']
            if 'Yield_tonne_per_ha' in row.index:
                self.crop_yield[k] = row['Yield_tonne_per_ha']
            if 'Price_per_tonne' in row.index:
                self.price[k] = row['Price_per_tonne']
            if 'Nitrogen_kg_per_ha' in row.index:
                self.nitrogen_use[k] = row['Nitrogen_kg_per_ha']
            else:
                self.nitrogen_use[k] = self.nitrogen_use[k - 1]

            self.production[k] = self.area_harvested[k] * self.crop_yield[k]

            # Compute derived for consistency
            self.climate_stress[k] = self._calc_climate_stress(temp, precip, gdd)
            self.water_stress_factor[k] = self._calc_water_stress(precip, k)
            self.co2_fert_factor[k], self.co2_nutrient_factor[k] = self._calc_co2_effects(co2)
            self.arable_land_available[k] = self._calc_land_available(k)

            self.erosion_rate[k] = self._calc_soil_erosion(precip, k)
            self.soil_depth_equiv[k] = self.soil_depth_equiv[k - 1] + (self.erosion_rate[k] - p.soil_formation_rate_t_ha_yr)
            self.soil_organic_matter[k] = self._calc_som(k, self.production[k])

            self.phosphorus_use[k] = self.phosphorus_use[k - 1] if k > 0 else p.initial_phosphorus_kg_ha
            self.potassium_use[k] = self.potassium_use[k - 1] if k > 0 else p.initial_potassium_kg_ha
            self.p_soil_stock[k] = self.p_soil_stock[k - 1] - p.phosphorus_loss_rate * self.erosion_rate[k] * 0.1

            self.npk_response[k] = self._calc_npk_response(
                self.nitrogen_use[k], self.phosphorus_use[k], self.potassium_use[k],
                step=k)
            self.soil_health[k] = self._calc_soil_health(k, self.erosion_rate[k], self.soil_organic_matter[k])
            self.soil_yield_multiplier[k] = self._calc_soil_yield_mult(self.soil_health[k])
            self.technology_factor[k] = self.technology_factor[k - 1] * (1 + p.technology_growth_rate)
            self.machinery_capital[k] = (self.machinery_capital[k - 1] * (1 - p.machinery_depreciation_rate) +
                                         p.machinery_investment_rate * self.machinery_capital[k - 1])

            # ── Shadow yield: run the model yield equation in parallel ──
            # During the historical period we use observed data, but we also
            # compute what the model *would* produce.  This (a) validates the
            # model structure against history and (b) maintains a running
            # calibration factor for a smooth transition to projections.
            if self._shadow_yield is not None:
                base_yield = p.initial_yield_t_per_ha
                machinery_effect = min(1.2, 1.0 + p.machinery_yield_effect * self.machinery_capital[k])
                shadow = (
                    base_yield
                    * self.npk_response[k]
                    * self.soil_yield_multiplier[k]
                    * self.climate_stress[k]
                    * self.water_stress_factor[k]
                    * self.technology_factor[k]
                    * self.co2_fert_factor[k]
                    * machinery_effect
                )
                self._shadow_yield[k] = shadow
                # Update running calibration factor with exponential smoothing
                if shadow > 0 and self.crop_yield[k] > 0:
                    instant_factor = self.crop_yield[k] / shadow
                    alpha = 0.3  # smoothing parameter
                    self._yield_calibration_factor = (
                        alpha * instant_factor +
                        (1 - alpha) * self._yield_calibration_factor
                    )

        elif k > 0:
            # ── PROJECTION ──

            # On the first projection step, compute calibration factor
            # so projected yield is continuous with the last observed yield.
            if self._transition_k is None:
                self._transition_k = k  # mark this as the transition step

            # 1. Land availability
            self.arable_land_available[k] = self._calc_land_available(k)

            # 2. Water stress
            self.water_stress_factor[k] = self._calc_water_stress(precip, k)

            # 3. Climate stress
            self.climate_stress[k] = self._calc_climate_stress(temp, precip, gdd)

            # 4. CO₂ effects
            self.co2_fert_factor[k], self.co2_nutrient_factor[k] = self._calc_co2_effects(co2)

            # 5. Technology factor
            tech_boost = self.scenario_adjustments.get('technology_boost', 0.0)
            self.technology_factor[k] = self.technology_factor[k - 1] * (1 + p.technology_growth_rate + tech_boost)

            # 6. Erosion
            self.erosion_rate[k] = self._calc_soil_erosion(precip, k)
            self.soil_depth_equiv[k] = self.soil_depth_equiv[k - 1] + (self.erosion_rate[k] - p.soil_formation_rate_t_ha_yr)

            # 7. N-P-K application
            n_app, p_app, k_app = self._calc_npk_step(k)
            self.nitrogen_use[k] = n_app
            self.phosphorus_use[k] = p_app
            self.potassium_use[k] = k_app

            # 8. Phosphorus soil stock (erosion-driven loss)
            self.p_soil_stock[k] = max(0, self.p_soil_stock[k - 1] -
                                       p.phosphorus_loss_rate * self.erosion_rate[k] * 0.1 +
                                       p_app * 0.3)  # some P replenished

            # 9. NPK response (Liebig's minimum, with erosion-P feedback)
            self.npk_response[k] = self._calc_npk_response(n_app, p_app, k_app, step=k)

            # 10. SOM & soil health
            # Use previous production as proxy for residue input
            self.soil_organic_matter[k] = self._calc_som(k, self.production[k - 1])
            self.soil_health[k] = self._calc_soil_health(k, self.erosion_rate[k], self.soil_organic_matter[k])
            self.soil_yield_multiplier[k] = self._calc_soil_yield_mult(self.soil_health[k])

            # 11. Machinery
            self.machinery_capital[k] = (
                self.machinery_capital[k - 1] * (1 - p.machinery_depreciation_rate) +
                p.machinery_investment_rate * max(0, self.revenue[k - 1]) / 1e6
            )

            # 12. YIELD (multiplicative, limited by Liebig's minimum)
            base_yield = p.initial_yield_t_per_ha
            machinery_effect = min(1.2, 1.0 + p.machinery_yield_effect * self.machinery_capital[k])

            raw_yield = (
                base_yield
                * self.npk_response[k]
                * self.soil_yield_multiplier[k]
                * self.climate_stress[k]
                * self.water_stress_factor[k]
                * self.technology_factor[k]
                * self.co2_fert_factor[k]
                * machinery_effect
            )

            # The calibration factor was continuously updated during the
            # historical period via shadow yield tracking (exponential
            # smoothing).  No sudden recalculation needed at the transition
            # — the factor is already close to the right value.
            # We do a final refinement to ensure exact continuity:
            if k == self._transition_k and self.crop_yield[k - 1] > 0 and raw_yield > 0:
                transition_factor = self.crop_yield[k - 1] / raw_yield
                # Blend with the running estimate (80% running, 20% exact)
                self._yield_calibration_factor = (
                    0.8 * self._yield_calibration_factor +
                    0.2 * transition_factor
                )

            self.crop_yield[k] = np.clip(
                raw_yield * self._yield_calibration_factor,
                0.1, p.max_attainable_yield,
            )

            # 13. Area (farmer behaviour)
            self.area_harvested[k] = self._calc_farmer_area_response(self.profit_margin[k - 1], k)

            # 14. Production
            self.production[k] = self.area_harvested[k] * self.crop_yield[k]

            # 15. Demand
            demand_info = self._calc_demand(year, k)
            self.total_demand[k] = demand_info['total']
            self.feed_demand[k] = demand_info['feed']
            self.food_demand[k] = demand_info['food']
            self.industrial_demand[k] = demand_info['industrial']
            self.export_demand[k] = demand_info['export']

            supply_demand_ratio = self.production[k] / max(1, self.total_demand[k])
            self.demand_satisfaction[k] = min(supply_demand_ratio, 2.0)

            # 16. Price (with smoothed supply-demand signal)
            # Price adjusts based on smoothed supply-demand imbalance
            # (market information delay — prices don't jump instantly)
            if k >= 2:
                avg_sd_ratio = np.mean([
                    self.production[j] / max(1, self.total_demand[j])
                    for j in range(max(0, k - 2), k + 1)
                    if self.total_demand[j] > 0
                ]) if any(self.total_demand[max(0,k-2):k+1] > 0) else supply_demand_ratio
            else:
                avg_sd_ratio = supply_demand_ratio
            price_pressure = (1 - avg_sd_ratio) * p.supply_demand_price_adjustment
            price_support = self.scenario_adjustments.get('price_support', 0.0)
            self.price[k] = np.clip(
                self.price[k - 1] * (1 + price_pressure),
                max(p.price_floor, price_support),
                p.price_ceiling,
            )

            # 17. Economics
            self.revenue[k] = self.production[k] * self.price[k]
            years_elapsed = year - p.year_start
            npk_cost = (
                self.nitrogen_use[k] * p.nitrogen_cost_per_kg +
                self.phosphorus_use[k] * p.phosphorus_cost_per_kg +
                self.potassium_use[k] * p.potassium_cost_per_kg
            )
            self.input_cost[k] = (
                self.area_harvested[k] * (50 + npk_cost) *
                (1 + p.input_cost_growth_rate) ** years_elapsed
            )
            self.profit_margin[k] = (self.revenue[k] - self.input_cost[k]) / max(1, self.revenue[k])

        # ── Step 0 special ──
        if k == 0:
            temp, precip, gdd, co2 = self._get_climate_at_year(self.time[0])
            self.co2_ppm[0] = co2
            self.co2_fert_factor[0], self.co2_nutrient_factor[0] = self._calc_co2_effects(co2)
            self.water_stress_factor[0] = self._calc_water_stress(precip, 0)
            self.erosion_rate[0] = self._calc_soil_erosion(precip, 0)

            demand_info = self._calc_demand(self.time[0], 0)
            self.total_demand[0] = demand_info['total']
            self.feed_demand[0] = demand_info['feed']
            self.food_demand[0] = demand_info['food']
            self.industrial_demand[0] = demand_info['industrial']
            self.export_demand[0] = demand_info['export']
            self.revenue[0] = self.production[0] * self.price[0]
            self.input_cost[0] = self.area_harvested[0] * (
                50 + p.initial_nitrogen_kg_ha * p.nitrogen_cost_per_kg +
                p.initial_phosphorus_kg_ha * p.phosphorus_cost_per_kg +
                p.initial_potassium_kg_ha * p.potassium_cost_per_kg
            )
            self.profit_margin[0] = (self.revenue[0] - self.input_cost[0]) / max(1, self.revenue[0])
            self.demand_satisfaction[0] = self.production[0] / max(1, self.total_demand[0])

    # ────────────────────────────────────────────────────────────
    # Run & output
    # ────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        print(f"Running {self.params.crop_name} model for {self.params.country}...")
        print(f"  Period: {self.params.year_start} – {self.params.year_end}")
        self._init_state_variables()
        # Reset calibration state for a fresh run
        self._yield_calibration_factor = 1.0
        self._transition_k = None
        self._shadow_yield = np.zeros(self.n_steps)

        self._step(0)
        for k in range(1, self.n_steps):
            self._step(k)

        self._has_run = True
        print(f"  Simulation complete. {self.n_steps} time steps.")
        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        n = self.n_steps
        return pd.DataFrame({
            # Core outputs
            'Year': self.time[:n],
            'Area_Harvested_ha': self.area_harvested[:n],
            'Yield_t_per_ha': self.crop_yield[:n],
            'Production_tonnes': self.production[:n],
            'Price_per_tonne': self.price[:n],

            # Land
            'Arable_Land_Available_ha': self.arable_land_available[:n],

            # Water
            'Water_Supply_mm': self.water_supply_mm[:n],
            'Water_Demand_mm': self.water_demand_mm[:n],
            'Water_Stress_Factor': self.water_stress_factor[:n],
            'Irrigation_Fraction': self.irrigation_fraction[:n],

            # Soil — nutrients
            'Nitrogen_kg_ha': self.nitrogen_use[:n],
            'Phosphorus_kg_ha': self.phosphorus_use[:n],
            'Potassium_kg_ha': self.potassium_use[:n],
            'NPK_Response': self.npk_response[:n],
            'P_Soil_Stock_kg_ha': self.p_soil_stock[:n],

            # Soil — health
            'Soil_Health': self.soil_health[:n],
            'Soil_Organic_Matter_pct': self.soil_organic_matter[:n],
            'Erosion_Rate_t_ha_yr': self.erosion_rate[:n],
            'Soil_Depth_Equiv': self.soil_depth_equiv[:n],
            'Soil_Yield_Multiplier': self.soil_yield_multiplier[:n],

            # Climate
            'Climate_Stress': self.climate_stress[:n],
            'CO2_ppm': self.co2_ppm[:n],
            'CO2_Fert_Factor': self.co2_fert_factor[:n],
            'CO2_Nutrient_Factor': self.co2_nutrient_factor[:n],

            # Technology & machinery
            'Technology_Factor': self.technology_factor[:n],
            'Machinery_Capital': self.machinery_capital[:n],

            # Demand
            'Total_Demand': self.total_demand[:n],
            'Feed_Demand': self.feed_demand[:n],
            'Food_Demand': self.food_demand[:n],
            'Industrial_Demand': self.industrial_demand[:n],
            'Export_Demand': self.export_demand[:n],
            'Demand_Satisfaction': self.demand_satisfaction[:n],

            # Economics
            'Revenue': self.revenue[:n],
            'Input_Cost': self.input_cost[:n],
            'Profit_Margin': self.profit_margin[:n],

            # Shadow yield (model yield during historical period for validation)
            'Shadow_Yield': self._shadow_yield[:n] if self._shadow_yield is not None else np.zeros(n),
        })

    # ────────────────────────────────────────────────────────────
    # Validation
    # ────────────────────────────────────────────────────────────

    def validate(self) -> Dict[str, any]:
        """Validate simulation against historical data.  Returns MAPE, R², bias."""
        if self.historical_data is None:
            return {"error": "No historical data loaded for validation"}
        if not self._has_run:
            return {"error": "Run simulation first"}

        results = self.get_results()
        hist = self.historical_data
        metrics = {}

        var_map = {
            'Production_tonnes': 'Production_tonnes',
            'Yield_t_per_ha': 'Yield_tonne_per_ha',
            'Area_Harvested_ha': 'Area_Harvested_ha',
        }

        for sim_col, hist_col in var_map.items():
            if hist_col not in hist.columns:
                continue

            sim_sub = results[['Year', sim_col]].rename(columns={sim_col: 'sim'})
            hist_sub = hist[['Year', hist_col]].rename(columns={hist_col: 'hist'})
            merged = pd.merge(sim_sub, hist_sub, on='Year', how='inner')
            if len(merged) == 0:
                continue

            actual = merged['hist'].values
            predicted = merged['sim'].values

            nonzero = actual != 0
            mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100 if nonzero.sum() > 0 else np.nan
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            bias = np.mean(predicted - actual)

            metrics[sim_col] = {
                'MAPE_%': round(mape, 2),
                'R2': round(r2, 4),
                'Bias': round(bias, 2),
                'N_observations': len(merged),
            }

        return metrics
