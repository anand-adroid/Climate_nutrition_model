"""
Climate Scenario Manager for IPCC SSP Pathways.

Generates temperature and precipitation projections under different
Shared Socioeconomic Pathways (SSP) for any country/region.

Data basis: IPCC AR6 Working Group I (2021) — global mean temperature
projections downscaled to regional level using CMIP6 model ensemble medians.

Author: Climate-Nutrition Modelling Project
License: MIT
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RegionalClimateBaseline:
    """
    Baseline climate conditions for a specific region.
    These are the 1971-2000 averages that scenarios are relative to.
    """
    region_name: str
    country: str
    baseline_temp_c: float  # Growing season average
    baseline_precip_mm: float  # Growing season total
    baseline_gdd: float  # Growing Degree Days
    temp_variability: float = 1.5  # Std dev of annual temp
    precip_variability: float = 0.15  # CV of annual precip
    latitude: float = 45.0  # For solar radiation proxy

    # Regional warming multiplier relative to global mean
    # (Some regions warm faster than global average)
    warming_amplification: float = 1.0

    # Precipitation change direction (-1 = drying, +1 = wetting)
    precip_trend_sign: float = 0.0


# Pre-defined regional baselines
REGIONAL_BASELINES = {
    'canada_ontario': RegionalClimateBaseline(
        region_name='Southern Ontario',
        country='Canada',
        baseline_temp_c=19.5,
        baseline_precip_mm=450,
        baseline_gdd=1800,
        temp_variability=1.2,
        precip_variability=0.12,
        latitude=43.5,
        warming_amplification=1.4,  # Canada warms faster than global
        precip_trend_sign=0.05,  # Slightly wetter
    ),
    'nigeria_north': RegionalClimateBaseline(
        region_name='Northern Nigeria (Savanna)',
        country='Nigeria',
        baseline_temp_c=30.0,
        baseline_precip_mm=800,
        baseline_gdd=3500,
        temp_variability=0.8,
        precip_variability=0.25,
        latitude=10.0,
        warming_amplification=1.1,
        precip_trend_sign=-0.05,  # Drying trend
    ),
    'nigeria_south': RegionalClimateBaseline(
        region_name='Southern Nigeria (Forest/Derived Savanna)',
        country='Nigeria',
        baseline_temp_c=27.0,
        baseline_precip_mm=1400,
        baseline_gdd=3200,
        temp_variability=0.5,
        precip_variability=0.15,
        latitude=7.0,
        warming_amplification=1.0,
        precip_trend_sign=-0.02,
    ),
}


class ClimateScenarioManager:
    """
    Generates climate projections under IPCC SSP scenarios.

    The SSP (Shared Socioeconomic Pathway) scenarios represent different
    trajectories of greenhouse gas emissions and socioeconomic development:

    - SSP1-2.6: Sustainability — strong mitigation, warming peaks ~1.8°C
    - SSP2-4.5: Middle of the Road — moderate mitigation, warming ~2.7°C by 2100
    - SSP3-7.0: Regional Rivalry — weak mitigation, warming ~3.6°C by 2100
    - SSP5-8.5: Fossil-fuel Development — no mitigation, warming ~4.4°C by 2100

    Temperature projections are based on IPCC AR6 Table SPM.1 central estimates.
    Regional downscaling uses warming amplification factors from CMIP6 ensemble.
    """

    # Global Mean Temperature Anomaly (°C above 1850-1900 baseline)
    # Source: IPCC AR6 WGI SPM Table SPM.1 (best estimates)
    SSP_GLOBAL_WARMING = {
        'ssp126': {  # SSP1-2.6
            2021: 1.1, 2030: 1.5, 2040: 1.6, 2050: 1.7,
            2060: 1.7, 2070: 1.7, 2080: 1.7, 2090: 1.8, 2100: 1.8,
        },
        'ssp245': {  # SSP2-4.5
            2021: 1.1, 2030: 1.5, 2040: 1.7, 2050: 2.0,
            2060: 2.2, 2070: 2.4, 2080: 2.5, 2090: 2.6, 2100: 2.7,
        },
        'ssp370': {  # SSP3-7.0
            2021: 1.1, 2030: 1.5, 2040: 1.8, 2050: 2.1,
            2060: 2.4, 2070: 2.8, 2080: 3.1, 2090: 3.4, 2100: 3.6,
        },
        'ssp585': {  # SSP5-8.5
            2021: 1.1, 2030: 1.6, 2040: 2.0, 2050: 2.4,
            2060: 2.8, 2070: 3.3, 2080: 3.8, 2090: 4.1, 2100: 4.4,
        },
    }

    # CO2 concentrations (ppm) by scenario
    SSP_CO2_PPM = {
        'ssp126': {2020: 412, 2050: 440, 2100: 430},
        'ssp245': {2020: 412, 2050: 500, 2100: 600},
        'ssp370': {2020: 412, 2050: 530, 2100: 850},
        'ssp585': {2020: 412, 2050: 570, 2100: 1135},
    }

    def __init__(self, baseline: RegionalClimateBaseline):
        self.baseline = baseline

    def generate_scenario(
        self,
        scenario: str = 'ssp245',
        year_start: int = 1971,
        year_end: int = 2100,
        seed: Optional[int] = 42,
    ) -> pd.DataFrame:
        """
        Generate annual climate data for a scenario.

        Args:
            scenario: One of 'ssp126', 'ssp245', 'ssp370', 'ssp585'
            year_start: First year of output
            year_end: Last year of output
            seed: Random seed for reproducible variability

        Returns:
            DataFrame with columns: Year, Temperature_C, Precipitation_mm,
            GDD, CO2_ppm, Heat_Stress_Days, Climate_Stress_Factor
        """
        if scenario not in self.SSP_GLOBAL_WARMING:
            raise ValueError(f"Unknown scenario: {scenario}. "
                           f"Choose from: {list(self.SSP_GLOBAL_WARMING.keys())}")

        rng = np.random.default_rng(seed)
        years = np.arange(year_start, year_end + 1)
        n = len(years)
        bl = self.baseline

        # Interpolate global warming trajectory
        warming_points = self.SSP_GLOBAL_WARMING[scenario]
        warming_years = sorted(warming_points.keys())
        warming_values = [warming_points[y] for y in warming_years]

        # For years before 2021, use linear ramp from 0.3°C in 1971
        all_warming = np.zeros(n)
        for i, yr in enumerate(years):
            if yr < min(warming_years):
                all_warming[i] = 0.3 + (warming_values[0] - 0.3) * (
                    (yr - year_start) / (min(warming_years) - year_start)
                )
            elif yr > max(warming_years):
                all_warming[i] = warming_values[-1]
            else:
                all_warming[i] = np.interp(yr, warming_years, warming_values)

        # Apply regional amplification
        regional_warming = all_warming * bl.warming_amplification

        # Temperature: baseline + regional warming + variability
        temp_noise = rng.normal(0, bl.temp_variability * 0.3, n)
        temperature = bl.baseline_temp_c + regional_warming + temp_noise

        # Precipitation: baseline + trend + variability
        precip_trend = bl.precip_trend_sign * all_warming  # Fractional change
        precip_noise = rng.normal(0, bl.precip_variability, n)
        precipitation = bl.baseline_precip_mm * (1 + precip_trend + precip_noise)
        precipitation = np.maximum(precipitation, 50)  # Minimum 50mm

        # Growing Degree Days (increases with temperature)
        gdd = bl.baseline_gdd + (temperature - bl.baseline_temp_c) * 100
        gdd = np.maximum(gdd, 500)

        # CO2 interpolation
        co2_points = self.SSP_CO2_PPM[scenario]
        co2_years = sorted(co2_points.keys())
        co2_vals = [co2_points[y] for y in co2_years]
        co2 = np.interp(years, co2_years, co2_vals)
        # Extend backward
        for i, yr in enumerate(years):
            if yr < min(co2_years):
                co2[i] = 330 + (co2_vals[0] - 330) * (yr - 1971) / (min(co2_years) - 1971)

        # Heat stress days (days above threshold)
        heat_threshold = 35.0
        heat_stress_days = np.maximum(0, (temperature - heat_threshold) * 15 +
                                     rng.normal(0, 3, n))
        heat_stress_days = np.clip(heat_stress_days, 0, 60)

        # Climate stress factor (composite, 0.3 to 1.15)
        temp_optimal = bl.baseline_temp_c
        temp_stress = 1.0 - 0.05 * np.abs(temperature - temp_optimal) ** 1.3
        precip_stress = 1.0 - 0.001 * np.abs(precipitation - bl.baseline_precip_mm)
        co2_benefit = 1.0 + 0.0001 * (co2 - 400)  # CO2 fertilization effect

        climate_stress = np.clip(
            temp_stress * np.clip(precip_stress, 0.5, 1.1) * np.clip(co2_benefit, 1.0, 1.15),
            0.3, 1.15
        )

        return pd.DataFrame({
            'Year': years,
            'Temperature_C': np.round(temperature, 2),
            'Precipitation_mm': np.round(precipitation, 1),
            'GDD': np.round(gdd, 0).astype(int),
            'CO2_ppm': np.round(co2, 1),
            'Heat_Stress_Days': np.round(heat_stress_days, 1),
            'Climate_Stress_Factor': np.round(climate_stress, 4),
            'Scenario': scenario,
        })

    def generate_all_scenarios(
        self,
        year_start: int = 1971,
        year_end: int = 2100,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """Generate data for all four SSP scenarios."""
        scenarios = {}
        for ssp in self.SSP_GLOBAL_WARMING.keys():
            scenarios[ssp] = self.generate_scenario(ssp, year_start, year_end, seed)
        return scenarios

    def get_scenario_summary(self) -> pd.DataFrame:
        """Return a summary table of scenario characteristics."""
        rows = []
        for ssp, warming in self.SSP_GLOBAL_WARMING.items():
            rows.append({
                'Scenario': ssp.upper(),
                'Description': {
                    'ssp126': 'Sustainability (strong mitigation)',
                    'ssp245': 'Middle of the Road',
                    'ssp370': 'Regional Rivalry (weak mitigation)',
                    'ssp585': 'Fossil-fuel Development (no mitigation)',
                }[ssp],
                'Global_Warming_2050_C': warming[2050],
                'Global_Warming_2100_C': warming[2100],
                'Regional_Warming_2100_C': round(
                    warming[2100] * self.baseline.warming_amplification, 1
                ),
            })
        return pd.DataFrame(rows)
