"""
Sensitivity Analysis Module — v1.0

Provides systematic parameter sensitivity analysis for the crop model:
  1. One-at-a-time (OAT) local sensitivity
  2. Morris method (elementary effects) for global screening
  3. Output: tornado diagrams, sensitivity indices, parameter rankings

References:
  • Saltelli et al. (2008) "Global Sensitivity Analysis: The Primer"
  • Morris (1991) "Factorial Sampling Plans for Preliminary Computational Experiments"
  • Pianosi et al. (2016) "Sensitivity analysis of environmental models" Environ. Model. Softw.

Author: Climate-Nutrition Modelling Project
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict
import copy
import warnings

from climate_nutrition_modelling.models.crop_model import CropModel, CropParameters


# ================================================================
# Sensitivity Parameter Definitions
# ================================================================

# Parameters to analyse, grouped by subsystem
SENSITIVITY_PARAMETERS = {
    # SOIL / NUTRIENTS
    'initial_nitrogen_kg_ha': {
        'label': 'N application (kg/ha)',
        'subsystem': 'Soil',
        'range': (0.5, 2.0),  # multiplier range
        'description': 'Baseline nitrogen fertiliser application rate',
    },
    'initial_phosphorus_kg_ha': {
        'label': 'P application (kg/ha)',
        'subsystem': 'Soil',
        'range': (0.5, 2.0),
        'description': 'Baseline phosphorus fertiliser application rate',
    },
    'initial_potassium_kg_ha': {
        'label': 'K application (kg/ha)',
        'subsystem': 'Soil',
        'range': (0.5, 2.0),
        'description': 'Baseline potassium fertiliser application rate',
    },
    'nitrogen_response_coeff': {
        'label': 'N response coefficient',
        'subsystem': 'Soil',
        'range': (0.5, 2.0),
        'description': 'Mitscherlich response curve steepness for N',
    },
    'erosion_base_rate_t_ha_yr': {
        'label': 'Base erosion rate (t/ha/yr)',
        'subsystem': 'Soil',
        'range': (0.5, 2.0),
        'description': 'Baseline soil erosion rate',
    },
    'initial_som_pct': {
        'label': 'Initial SOM (%)',
        'subsystem': 'Soil',
        'range': (0.5, 1.5),
        'description': 'Initial soil organic matter percentage',
    },

    # WATER
    'crop_water_requirement_mm': {
        'label': 'Crop water requirement (mm)',
        'subsystem': 'Water',
        'range': (0.7, 1.3),
        'description': 'Seasonal crop water demand',
    },
    'irrigation_fraction': {
        'label': 'Irrigated fraction',
        'subsystem': 'Water',
        'range': (0.0, 5.0),  # absolute multiplier since base can be very small
        'description': 'Fraction of cropland under irrigation',
    },
    'water_stress_sensitivity': {
        'label': 'Water stress sensitivity',
        'subsystem': 'Water',
        'range': (0.5, 2.0),
        'description': 'Exponent of water-stress response curve',
    },

    # CLIMATE
    'temperature_sensitivity': {
        'label': 'Temperature sensitivity',
        'subsystem': 'Climate',
        'range': (0.5, 2.0),
        'description': 'Yield response to temperature deviation',
    },
    'co2_fertilisation_beta': {
        'label': 'CO₂ fertilisation effect',
        'subsystem': 'Climate',
        'range': (0.5, 2.0),
        'description': 'Yield benefit per ppm CO₂ above reference',
    },
    'co2_protein_degradation': {
        'label': 'CO₂ nutrient degradation',
        'subsystem': 'Climate',
        'range': (0.5, 2.0),
        'description': 'Protein/micronutrient loss per ppm CO₂',
    },
    'heat_stress_threshold': {
        'label': 'Heat stress threshold (°C)',
        'subsystem': 'Climate',
        'range': (0.9, 1.1),  # small range — physical parameter
        'description': 'Temperature above which heat stress begins',
    },

    # CROP / TECHNOLOGY
    'technology_growth_rate': {
        'label': 'Technology growth rate (%/yr)',
        'subsystem': 'Crop',
        'range': (0.0, 3.0),  # absolute multiplier
        'description': 'Annual rate of yield improvement from technology',
    },
    'max_attainable_yield': {
        'label': 'Max attainable yield (t/ha)',
        'subsystem': 'Crop',
        'range': (0.7, 1.3),
        'description': 'Theoretical maximum yield ceiling',
    },

    # LAND
    'urban_encroachment_rate': {
        'label': 'Urban encroachment rate',
        'subsystem': 'Land',
        'range': (0.0, 3.0),
        'description': 'Annual rate of farmland loss to urbanisation',
    },

    # ECONOMICS
    'area_price_elasticity': {
        'label': 'Area-price elasticity',
        'subsystem': 'Economics',
        'range': (0.5, 2.0),
        'description': 'Farmer area response to price changes',
    },
}

# Output metrics to track
OUTPUT_METRICS = {
    'Production_2050': 'Production (tonnes) in 2050',
    'Production_2080': 'Production (tonnes) in 2080',
    'Yield_2050': 'Yield (t/ha) in 2050',
    'Yield_2080': 'Yield (t/ha) in 2080',
    'Soil_Health_2080': 'Soil health index in 2080',
    'CO2_Nutrient_Factor_2080': 'CO₂ nutrient degradation factor in 2080',
    'Water_Stress_2080': 'Water stress factor in 2080',
}


# ================================================================
# SensitivityAnalyzer
# ================================================================

class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on the crop model.

    Methods
    -------
    run_oat(params, n_levels=5)
        One-at-a-time local sensitivity analysis.
    run_morris(params, n_trajectories=10, n_levels=4)
        Morris method for global screening.
    get_tornado_data(metric='Production_2080')
        Returns data for tornado diagram.
    get_parameter_rankings()
        Returns parameters ranked by influence.

    Usage::

        analyzer = SensitivityAnalyzer(
            base_params=CropParameters.from_json('config/canada_corn.json'),
            climate_data=climate_df,
            population_data=pop_df,
            historical_data_path='data/canada/corn_production.csv'
        )
        results = analyzer.run_oat()
        tornado = analyzer.get_tornado_data('Production_2080')
    """

    def __init__(
        self,
        base_params: CropParameters,
        climate_data: Optional[pd.DataFrame] = None,
        population_data: Optional[pd.DataFrame] = None,
        historical_data_path: Optional[str] = None,
        scenario_adjustments: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
    ):
        self.base_params = base_params
        self.climate_data = climate_data
        self.population_data = population_data
        self.historical_data_path = historical_data_path
        self.scenario_adjustments = scenario_adjustments or {}
        self.parameters = parameters or SENSITIVITY_PARAMETERS
        self.results: Optional[pd.DataFrame] = None
        self._baseline_outputs: Optional[Dict] = None

    def _run_model(self, params: CropParameters) -> pd.DataFrame:
        """Run a single model instance and return results."""
        model = CropModel(params)
        if self.historical_data_path:
            model.load_historical_data(self.historical_data_path)
        if self.climate_data is not None:
            model.climate_data = self.climate_data.copy()
        if self.population_data is not None:
            model.population_data = self.population_data.copy()
        model.set_scenario(**self.scenario_adjustments)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.run()

    def _extract_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract key output metrics from simulation results."""
        metrics = {}

        for target_year in [2050, 2080]:
            yr_data = df[df['Year'] == target_year]
            if len(yr_data) == 0:
                # Use closest available year
                yr_data = df.iloc[[-1]]

            row = yr_data.iloc[0]

            if target_year == 2050:
                metrics['Production_2050'] = row['Production_tonnes']
                metrics['Yield_2050'] = row['Yield_t_per_ha']
            else:
                metrics['Production_2080'] = row['Production_tonnes']
                metrics['Yield_2080'] = row['Yield_t_per_ha']
                metrics['Soil_Health_2080'] = row['Soil_Health']
                metrics['CO2_Nutrient_Factor_2080'] = row['CO2_Nutrient_Factor']
                metrics['Water_Stress_2080'] = row['Water_Stress_Factor']

        return metrics

    def _perturb_param(self, param_name: str, multiplier: float) -> CropParameters:
        """Create a copy of base params with one parameter perturbed."""
        new_params = copy.deepcopy(self.base_params)
        current_val = getattr(new_params, param_name)

        # Handle zero or near-zero base values
        if abs(current_val) < 1e-10:
            new_val = multiplier * 0.01  # small absolute perturbation
        else:
            new_val = current_val * multiplier

        setattr(new_params, param_name, new_val)
        return new_params

    # ────────────────────────────────────────────────────────────
    # One-at-a-Time (OAT) Sensitivity Analysis
    # ────────────────────────────────────────────────────────────

    def run_oat(self, n_levels: int = 5) -> pd.DataFrame:
        """
        Run one-at-a-time sensitivity analysis.

        Each parameter is varied across `n_levels` multiplier levels
        while all others are held at baseline. Returns a DataFrame
        with columns: Parameter, Multiplier, and all output metrics.

        Parameters
        ----------
        n_levels : int
            Number of multiplier levels to test per parameter.

        Returns
        -------
        pd.DataFrame
            Full OAT results.
        """
        print(f"Running OAT sensitivity analysis ({len(self.parameters)} parameters × {n_levels} levels)...")

        # 1. Baseline run
        baseline_df = self._run_model(self.base_params)
        self._baseline_outputs = self._extract_metrics(baseline_df)
        print(f"  Baseline: Production_2080 = {self._baseline_outputs.get('Production_2080', 0):,.0f} t")

        rows = []

        # Baseline row
        for metric, value in self._baseline_outputs.items():
            rows.append({
                'Parameter': 'BASELINE',
                'Subsystem': '-',
                'Multiplier': 1.0,
                'Metric': metric,
                'Value': value,
                'Pct_Change': 0.0,
            })

        # 2. Perturb each parameter
        for param_name, info in self.parameters.items():
            lo, hi = info['range']
            multipliers = np.linspace(lo, hi, n_levels)
            print(f"  {info['label']} ({param_name}): {lo:.1f}x – {hi:.1f}x")

            for mult in multipliers:
                perturbed = self._perturb_param(param_name, mult)
                try:
                    sim_df = self._run_model(perturbed)
                    metrics = self._extract_metrics(sim_df)
                except Exception as e:
                    print(f"    Warning: {param_name} @ {mult:.2f}x failed: {e}")
                    continue

                for metric_name, value in metrics.items():
                    baseline_val = self._baseline_outputs.get(metric_name, 0)
                    pct_change = ((value - baseline_val) / max(abs(baseline_val), 1e-10)) * 100

                    rows.append({
                        'Parameter': info['label'],
                        'Subsystem': info['subsystem'],
                        'Multiplier': round(mult, 3),
                        'Metric': metric_name,
                        'Value': value,
                        'Pct_Change': round(pct_change, 2),
                    })

        self.results = pd.DataFrame(rows)
        print(f"  Complete: {len(rows)} data points.")
        return self.results

    # ────────────────────────────────────────────────────────────
    # Morris Method (Elementary Effects)
    # ────────────────────────────────────────────────────────────

    def run_morris(self, n_trajectories: int = 10, n_levels: int = 4) -> pd.DataFrame:
        """
        Morris method for global sensitivity screening.

        Computes μ* (mean absolute elementary effect) and σ
        (standard deviation of elementary effects) for each parameter.
        High μ* = influential parameter. High σ = non-linear or interactive.

        Parameters
        ----------
        n_trajectories : int
            Number of Morris trajectories.
        n_levels : int
            Number of grid levels for parameter space.

        Returns
        -------
        pd.DataFrame
            Morris indices: μ*, σ, μ*/σ ratio for each parameter × metric.
        """
        param_names = list(self.parameters.keys())
        k = len(param_names)
        delta = 1.0 / (n_levels - 1)

        print(f"Running Morris method ({k} parameters, {n_trajectories} trajectories)...")

        # Baseline
        baseline_df = self._run_model(self.base_params)
        self._baseline_outputs = self._extract_metrics(baseline_df)

        # Storage for elementary effects
        ee = {p: {m: [] for m in OUTPUT_METRICS} for p in param_names}

        for traj in range(n_trajectories):
            # Random starting point in [0, 1]^k
            x_base = np.random.uniform(0.1, 0.9, k)

            # Run baseline at this starting point
            params_base = self._morris_point_to_params(x_base, param_names)
            try:
                df_base = self._run_model(params_base)
                metrics_base = self._extract_metrics(df_base)
            except Exception:
                continue

            # Elementary effects: perturb each parameter
            for i, pname in enumerate(param_names):
                x_perturbed = x_base.copy()
                x_perturbed[i] = min(1.0, x_perturbed[i] + delta)

                params_pert = self._morris_point_to_params(x_perturbed, param_names)
                try:
                    df_pert = self._run_model(params_pert)
                    metrics_pert = self._extract_metrics(df_pert)
                except Exception:
                    continue

                for metric in OUTPUT_METRICS:
                    if metric in metrics_base and metric in metrics_pert:
                        effect = (metrics_pert[metric] - metrics_base[metric]) / delta
                        ee[pname][metric].append(effect)

        # Compute Morris indices
        rows = []
        for pname in param_names:
            info = self.parameters[pname]
            for metric in OUTPUT_METRICS:
                effects = ee[pname][metric]
                if len(effects) == 0:
                    continue
                mu_star = np.mean(np.abs(effects))
                sigma = np.std(effects)
                rows.append({
                    'Parameter': info['label'],
                    'Param_Key': pname,
                    'Subsystem': info['subsystem'],
                    'Metric': metric,
                    'mu_star': round(mu_star, 4),
                    'sigma': round(sigma, 4),
                    'mu_star_sigma_ratio': round(mu_star / max(sigma, 1e-10), 2),
                    'n_effects': len(effects),
                })

        morris_df = pd.DataFrame(rows)
        print(f"  Complete: {len(rows)} indices computed.")
        return morris_df

    def _morris_point_to_params(self, x: np.ndarray, param_names: List[str]) -> CropParameters:
        """Convert a [0,1]^k point to actual parameter values."""
        new_params = copy.deepcopy(self.base_params)
        for i, pname in enumerate(param_names):
            info = self.parameters[pname]
            lo, hi = info['range']
            multiplier = lo + x[i] * (hi - lo)
            current_val = getattr(new_params, pname)
            if abs(current_val) < 1e-10:
                setattr(new_params, pname, multiplier * 0.01)
            else:
                setattr(new_params, pname, current_val * multiplier)
        return new_params

    # ────────────────────────────────────────────────────────────
    # Analysis outputs
    # ────────────────────────────────────────────────────────────

    def get_tornado_data(self, metric: str = 'Production_2080') -> pd.DataFrame:
        """
        Get data formatted for a tornado diagram.

        Returns DataFrame with columns:
            Parameter, Low_Value, High_Value, Low_Pct, High_Pct, Range, Subsystem

        Parameters
        ----------
        metric : str
            Which output metric to compute tornado for.

        Returns
        -------
        pd.DataFrame
            Tornado data sorted by range (most influential first).
        """
        if self.results is None:
            raise RuntimeError("Run run_oat() first.")

        metric_data = self.results[
            (self.results['Metric'] == metric) &
            (self.results['Parameter'] != 'BASELINE')
        ]

        tornado_rows = []
        for param_label in metric_data['Parameter'].unique():
            param_df = metric_data[metric_data['Parameter'] == param_label]

            lo_row = param_df.loc[param_df['Multiplier'].idxmin()]
            hi_row = param_df.loc[param_df['Multiplier'].idxmax()]

            tornado_rows.append({
                'Parameter': param_label,
                'Subsystem': lo_row['Subsystem'],
                'Low_Value': lo_row['Value'],
                'High_Value': hi_row['Value'],
                'Low_Pct': lo_row['Pct_Change'],
                'High_Pct': hi_row['Pct_Change'],
                'Range': abs(hi_row['Pct_Change'] - lo_row['Pct_Change']),
            })

        tornado_df = pd.DataFrame(tornado_rows).sort_values('Range', ascending=False)
        return tornado_df.reset_index(drop=True)

    def get_parameter_rankings(self) -> pd.DataFrame:
        """
        Rank parameters by their overall influence across all metrics.

        Returns DataFrame with: Parameter, Subsystem, Avg_Range, Max_Range, Rank
        """
        if self.results is None:
            raise RuntimeError("Run run_oat() first.")

        rankings = []
        for metric in OUTPUT_METRICS:
            try:
                tornado = self.get_tornado_data(metric)
                for _, row in tornado.iterrows():
                    rankings.append({
                        'Parameter': row['Parameter'],
                        'Subsystem': row['Subsystem'],
                        'Metric': metric,
                        'Range': row['Range'],
                    })
            except Exception:
                continue

        if not rankings:
            return pd.DataFrame()

        rank_df = pd.DataFrame(rankings)
        summary = rank_df.groupby(['Parameter', 'Subsystem']).agg(
            Avg_Range=('Range', 'mean'),
            Max_Range=('Range', 'max'),
            N_Metrics=('Metric', 'count'),
        ).reset_index().sort_values('Avg_Range', ascending=False)
        summary['Rank'] = range(1, len(summary) + 1)
        return summary.reset_index(drop=True)

    def get_baseline_outputs(self) -> Dict[str, float]:
        """Return baseline metric values."""
        if self._baseline_outputs is None:
            df = self._run_model(self.base_params)
            self._baseline_outputs = self._extract_metrics(df)
        return self._baseline_outputs
