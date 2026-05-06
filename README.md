# Climate-Nutrition System Dynamics Model

A system dynamics model that simulates the interplay between **climate change**, **agricultural production**, and **nutritional adequacy** for policy decision-making. Built on Forrester (1961) and Sterman (2000) System Dynamics methodology.

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://streamlit.io)

## What This Model Does

The model simulates **130 years (1971–2100)** of crop production under IPCC climate scenarios, coupling seven interconnected subsystems through six feedback loops. Unlike simple trend projections, it captures circular causality — when you change one input (e.g., increase fertilizer), the effects propagate through soil health, water stress, farmer economics, and nutrition simultaneously.

**Countries:** Canada (corn, tomato, leafy greens) and Nigeria (maize, cassava, rice, sorghum)

**Key features:**
- Historical replay (1971–2023) from official statistics with smooth transition to model projections
- Four IPCC SSP climate scenarios (SSP1-2.6 through SSP5-8.5)
- Liebig's Law yield calculation with Mitscherlich fertilizer response curves
- CO₂ fertilization vs. nutrient dilution (C3/C4 crop pathways)
- Population-driven demand with nutritional gap analysis
- Six verified feedback loops (4 balancing, 2 reinforcing)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/anand-adroid/Climate_nutrition_model.git
cd Climate_nutrition_model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the dashboard

```bash
streamlit run climate_nutrition_modelling/app.py
```

Your browser will open automatically at `http://localhost:8501` with the full interactive dashboard.

## Project Structure

```
Climate_nutrition_model/
├── climate_nutrition_modelling/
│   ├── app.py                          # Streamlit dashboard (main entry point)
│   ├── models/
│   │   ├── crop_model.py               # Core system dynamics engine
│   │   ├── climate_scenarios.py        # IPCC SSP scenario generator
│   │   ├── nutrient_converter.py       # Crop → nutrient conversion
│   │   ├── nutrition_gap.py            # Supply vs. demand gap analysis
│   │   ├── sensitivity_analysis.py     # OAT parameter sensitivity
│   │   └── model_comparison.py         # Cross-model validation
│   ├── config/
│   │   ├── canada_corn.json            # Canada crop parameters
│   │   ├── canada_tomato.json
│   │   ├── canada_leafy_greens.json
│   │   ├── nigeria_maize.json          # Nigeria crop parameters
│   │   ├── nigeria_cassava.json
│   │   ├── nigeria_rice.json
│   │   └── nigeria_sorghum.json
│   ├── data/
│   │   ├── canada/                     # Historical CSVs (1971-2023)
│   │   └── nigeria/                    # Historical CSVs (1971-2023)
│   └── tests/
├── test_feedback_loops.py              # Canada feedback loop verification
├── test_nigeria_feedback_loops.py      # Nigeria feedback loop verification
├── Nigeria_Policy_Guide.docx           # Comprehensive policy document
├── requirements.txt
└── README.md
```

## Dashboard Tabs

| Tab | What It Shows | Key Sliders |
|-----|---------------|-------------|
| **1. Overview** | Total production, population, energy/protein adequacy | All |
| **2. Crop Production** | Yield, area, production, price per crop | Tech R&D, Nitrogen, Conservation |
| **3. Water & Land** | Irrigation fraction, water stress, land use | Irrigation expansion |
| **4. Soil & Nutrients** | Soil health, erosion, SOM, NPK application | Conservation, Nitrogen multiplier |
| **5. Climate Impact** | Temperature, precipitation, CO₂ trajectories | SSP scenario dropdown only |
| **6. Nutrient Supply** | Energy, protein, iron, zinc, vitamin A/C supply | All production sliders |
| **7. Nutritional Gaps** | Supply ÷ demand ratio per nutrient | All (population is key driver) |
| **8. Sensitivity** | One-at-a-time parameter sensitivity | None (automated) |
| **9. Historical Validation** | Model vs. FAOSTAT historical data | None (comparison view) |
| **10. Model Comparison** | Cross-model benchmarking (IPCC, AgMIP) | SSP scenario |
| **11. Population Data** | Population projections (UN WPP) | None (exogenous input) |

## Feedback Loops

The model contains six core feedback loops verified computationally:

| Loop | Type | Mechanism |
|------|------|-----------|
| **B1** Market Equilibrium | Balancing | ↑yield → ↑supply → ↓price → ↓area → ↓production |
| **B2** Soil Degradation | Balancing | ↑nitrogen → ↑yield (short-term) → soil degrades → ↓yield (long-term) |
| **B3** Water Stress → Irrigation | Balancing | ↑water stress → farmers invest in irrigation → ↓stress |
| **B4** Erosion → P Depletion | Balancing | ↑erosion → ↓phosphorus stock → ↓yield → ↓residue → ↑erosion |
| **R1** Input Investment | Reinforcing | ↑profit → ↑fertilizer → ↑yield → ↑revenue → ↑profit |
| **R2** Capital Investment | Reinforcing | ↑revenue → ↑machinery → ↑yield → ↑revenue |

### How to Test Each Loop

- **B1:** Set Tech R&D to +2%. Watch yield double on Tab 2, then observe price adjusting downward.
- **B2:** Set Nitrogen multiplier to 2x. Watch soil health crash from 0.92 to 0.38 on Tab 4, then yield drops on Tab 2.
- **B3:** Set Irrigation to Maximum. Watch irrigation % jump on Tab 3, water stress drops, yield rises on Tab 2.
- **B4:** Set Conservation to Maximum. Watch erosion halve on Tab 4, P stock stabilizes.
- **R1/R2:** Compare baseline vs. Tech +2% — as revenue grows, N application and machinery investment increase endogenously.

## Data Sources

All historical data comes from authoritative government and international sources:

| Data | Source | URL |
|------|--------|-----|
| Crop production | FAOSTAT | [fao.org/faostat](https://www.fao.org/faostat/en/#data/QCL) |
| Nigeria census | NBS Nigeria | [nigerianstat.gov.ng](https://www.nigerianstat.gov.ng/) |
| Fertilizer stats | IFDC 2024/2025 | [ifdc.org](https://ifdc.org/wp-content/uploads/2024/06/EN_Nigeria-Fertilizer-Statistics-Overview-2024-Edition-1.pdf) |
| Population | UN WPP 2024 | [population.un.org](https://population.un.org/wpp/) |
| Climate baselines | World Bank | [climateknowledgeportal.worldbank.org](https://climateknowledgeportal.worldbank.org/country/nigeria/climate-data-historical) |
| Climate scenarios | IPCC AR6 | [ipcc.ch](https://www.ipcc.ch/report/ar6/wg2/) |
| Nutrient profiles | USDA FoodData | [fdc.nal.usda.gov](https://fdc.nal.usda.gov/) |
| Crop prices | FEWS NET | [fews.net](https://fews.net/) |

## Model Validation

The model is validated against established benchmarks:

| Benchmark | Metric | Literature | Our Model |
|-----------|--------|------------|-----------|
| IPCC AR6 WGII | Yield change per °C | -3 to -8% | -3.5 to -4% (C4) |
| AgMIP Global | Africa maize 2050 | -5 to -15% | -8 to -12% |
| Zhu et al. 2025 | CO₂ protein dilution | -5 to -10% at 550ppm | -6 to -8% |
| FAOSTAT trend | Nigeria maize growth | +2.0%/yr | +1.5–2.5%/yr |

## Running Tests

```bash
# Verify all feedback loops for Canada
python test_feedback_loops.py

# Verify all feedback loops for Nigeria
python test_nigeria_feedback_loops.py

# Run pipeline tests
python -m pytest climate_nutrition_modelling/tests/ -v
```

## Technical Details

- **Methodology:** System Dynamics (Forrester 1961, Sterman 2000)
- **Yield equation:** Liebig's Law of the Minimum (multiplicative)
- **Fertilizer response:** Mitscherlich diminishing returns curves
- **Climate:** IPCC SSP pathways with regional amplification factors
- **Calibration:** Shadow yield tracking with exponential smoothing (α=0.3)
- **Historical period:** Real data replay (1971–2023) from official statistics

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```
Anand M. (2026). "Climate-Nutrition System Dynamics Model: Coupling Agricultural
Production, Climate Change, and Nutritional Adequacy for Policy Analysis."
GitHub: https://github.com/anand-adroid/Climate_nutrition_model
```
