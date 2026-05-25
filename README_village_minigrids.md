<!--
SPDX-FileCopyrightText: 2026 Mohamed Amine Chebaane

SPDX-License-Identifier: CC-BY-4.0
-->
# README – Village-Level Mini-Grid Optimization (village_minigrids.py)

**Bachelor Thesis: Offgrid Integration in PyPSA-Earth**
OTH Regensburg | Supervisors: Anton Achhammer, Prof. Michael Sterner

---

## Overview

`village_minigrids.py` extends the county-level approach of `build_offgrid.py` by going one
step deeper: instead of one large mini-grid per county, it creates **many small
mini-grids per village or settlement cluster**.

| Script | Granularity | Method |
|--------|-------------|--------|
| `build_offgrid.py`  | County level | One mini-grid per county |
| `village_minigrids.py` | Village level | One mini-grid per village cluster |

The key addition is **WorldPop raster data** combined with **DBSCAN clustering**
to identify individual unelectrified settlements within each offgrid county.

---

## How It Works – Step by Step

### Step 1 – Load Data
Loads the baseline PyPSA-Earth network, ERA5 weather data, GADM shapes,
electrification data, and the WorldPop raster file.

### Step 2 – Geo-Logic (Layer 1): Identify Offgrid Counties
Same criteria as `build_offgrid.py` – identifies which counties are offgrid candidates:

| Criterion | Description | Default |
|-----------|-------------|---------|
| C1 | Grid distance > 50 km | active |
| C2 | Electrification rate < 50% | active |
| C3 | Population density > 10 P/km² | active |
| C4 | Baseline load < 20 MW | inactive |

### Step 3 – WorldPop Clustering: Identify Villages
For each offgrid county:
1. **WorldPop raster** is read and cropped to the county boundary
2. Cells with population > 5 persons and distance > 50 km from the grid are kept
3. **DBSCAN clustering** groups nearby cells into village clusters
4. Clusters are filtered by population (1,000–50,000 persons)

```
WorldPop Raster (1km resolution)
    → Filter: pop > 5, dist > 50 km
    → DBSCAN clustering (radius = 3 km)
    → Village clusters (1,000–50,000 persons)
```

### Step 4 – Tech-Logic (Layer 2): Optimize Mini-Grid per Village
For each village cluster a standalone PyPSA network is built and optimized:

```
Isolated mini-grid (400V)
    ├── Solar PV      (ERA5 profile, p_nom_extendable)
    ├── Battery       (6h storage, p_nom_extendable)
    ├── Diesel        (backup, p_nom_extendable)
    └── Load          (60 kWh/yr × population)
```

### Step 5 – Results & Visualization
Results are saved to CSV and a map is generated showing all village mini-grids.

---

## PyPSA Units

| Quantity | Unit | Note |
|----------|------|------|
| `p_set`, `p_nom` | MW | Load and capacities |
| `capital_cost` | EUR/MW/a | Annualized |
| `marginal_cost` | EUR/MWh | Operating cost |
| `lts.sum()` | MWh/yr | Total annual load |

**Unit conversions in the code:**
```python
# Load: kW → MW
avg_load_mw = population * kwh_per_person_yr / 8760 / 1000

# CAPEX: EUR/kW → EUR/MW/a
capital_cost = solar_capex * 1000 * annuity_factor

# LCOE: EUR/yr / MWh/yr / 1000 = EUR/kWh
lcoe = n.objective / total_load_mwh / 1000
```

---

## Configuration (`CONFIG` dictionary)

All parameters are set in the `CONFIG` dictionary at the top of the script.

### File Paths
```python
"shapes_file": "resources/shapes/gadm_shapes.geojson"
"network_pattern": "results/networks/*.nc"
"cutout_file": "cutouts/cutout-2013-era5.nc"
"elec_data_dir": "data/elec_rates"
"worldpop_file": "data/WorldPop/ken_ppp_2020_UNadj_constrained.tif"
```

### Geo-Logic Parameters
```python
"max_distance_km": 50  # C1: minimum grid distance [km]
"max_elec_rate": 0.50  # C2: maximum electrification rate
"min_pop_density": 10  # C3: minimum population density [P/km²]
"use_c1_distance": True  # activate/deactivate criteria
"use_c2_elec_rate": True
"use_c3_pop_density": True
"use_c4_low_load": False
```

### WorldPop Clustering Parameters
```python
"min_pop_per_cell": 5  # minimum population per WorldPop cell
"min_village_pop": 1000  # minimum village cluster population
"max_village_pop": 50000  # maximum village cluster population
"cluster_radius_km": 3.0  # DBSCAN neighborhood radius [km]
"cluster_min_cells": 2  # DBSCAN minimum cells per cluster
```

### Technology CAPEX
```python
"solar_capex": 800  # [EUR/kW]  – PyPSA technology-data
"battery_capex_kwh": 250  # [EUR/kWh] – PyPSA technology-data
"battery_max_hours": 6  # [h]
"diesel_capex": 400  # [EUR/kW]  – PyPSA technology-data
"diesel_marginal": 300  # [EUR/MWh]
"diesel_co2": 0.27  # [t/MWh]   – IPCC Guidelines 2006
"shedding_cost": 5000  # [EUR/MWh]
```

### Annuity
```python
"discount_rate": 0.08  # 8% – ESMAP Mini Grid Design Manual (2019)
"asset_lifetime": 20  # years
```

### Target Counties
```python
"target_counties": ["KE.37_1"]  # [] = all offgrid counties
# ["KE.37_1"] = only this county (for testing)
```

---

## Running the Script

```bash
# Run directly:
python village_minigrids.py

# To test with a single county first:
# Set in CONFIG: "target_counties": ["KE.37_1"]

# To run all offgrid counties:
# Set in CONFIG: "target_counties": []
```

### Required Files
```
data/WorldPop/ken_ppp_2020_UNadj_constrained.tif   ← WorldPop raster (Kenya)
data/elec_rates/KE_electricity_access.csv           ← (optional) electrification data
cutouts/cutout-2013-era5.nc                         ← ERA5 weather data
resources/shapes/gadm_shapes.geojson               ← GADM shapes
results/networks/*.nc                               ← solved baseline network
```

---

## DBSCAN Clustering Explained

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
groups nearby WorldPop cells into village clusters without requiring
a predefined number of clusters.

```
Parameters:
  eps = cluster_radius_km / 111  → neighborhood radius in degrees (≈ 3 km)
  min_samples = 2                → minimum cells per cluster

Output:
  label = -1   → noise (isolated cell, not part of any cluster)
  label >= 0   → cluster ID
```

**Why DBSCAN?**
- Finds clusters of arbitrary shape (villages are not circular)
- Automatically determines the number of clusters
- Robust to noise and outliers

*Source: Ester et al. (1996), density-based clustering*

---

## Key Formulas

### Annuity Factor
```
annuity_factor = r × (1+r)^n / ((1+r)^n - 1)

With r = 0.08 (8%), n = 20 years:
annuity_factor ≈ 0.1019

Source: ESMAP Mini Grid Design Manual (2019)
```

### LCOE (Levelized Cost of Energy)
```
LCOE [EUR/kWh] = n.objective [EUR/yr] / total_load [MWh/yr] / 1000

Where n.objective = annualized CAPEX + OPEX + diesel fuel costs
```

### Solar Autarky
```
autarky [%] = solar_gen_mwh / (solar_gen_mwh + diesel_gen_mwh) × 100

Note: solar_gen / total_load would exceed 100% due to battery losses.
      This formula gives the correct solar share of actual generation.
```

### Grid Connection Cost Comparison
```
grid_capex    = distance_km × 15,000 EUR/km + 35,000 EUR (substation)
grid_total_yr = grid_capex × annuity_factor + grid_capex × 0.03 (OPEX)

offgrid_cheaper = total_cost_yr < grid_total_yr
```

---

## Output Files

| File | Description |
|------|-------------|
| `village_minigrid_results.csv` | Results per village cluster |
| `village_minigrids_map.png` | Map of all village mini-grids |

### Results CSV Columns

| Column | Unit | Description |
|--------|------|-------------|
| `village_id` | – | Unique ID (e.g. `KE.37_1_v001`) |
| `gadm_id` | – | Parent county GADM ID |
| `population` | persons | Village cluster population |
| `dist_km` | km | Average distance to grid |
| `n_cells` | – | Number of WorldPop cells in cluster |
| `elec_rate` | 0–1 | County electrification rate |
| `solar_src` | – | ERA5 / Fallback |
| `solar_kw` | kW | Optimal solar capacity |
| `battery_kw` | kW | Optimal battery power |
| `battery_kwh` | kWh | Optimal battery energy |
| `diesel_kw` | kW | Optimal diesel capacity |
| `lcoe_eur_kwh` | EUR/kWh | Levelized cost of energy |
| `autarky_pct` | % | Solar share of total generation |
| `co2_t_yr` | t/yr | CO₂ from diesel backup |
| `capex_total_eur` | EUR | Total investment costs |
| `total_cost_eur_yr` | EUR/yr | Annual total costs |
| `grid_capex_eur` | EUR | Grid connection CAPEX |
| `grid_total_eur_yr` | EUR/yr | Annual grid connection costs |
| `offgrid_cheaper` | bool | True = offgrid cheaper |
| `centroid_x` | degrees | Village longitude |
| `centroid_y` | degrees | Village latitude |

---

## Expected Results

Based on results for **Kenya county KE.37_1**:

| KPI | Expected Range |
|-----|---------------|
| LCOE | 0.09 – 0.12 EUR/kWh |
| Solar autarky | 95 – 99% |
| Solar capacity | ~50–800 kW per village |
| Battery storage | ~6h × solar capacity |
| Diesel backup | Small fraction of solar |
| CO₂ emissions | Low (diesel rarely used) |

**Typical village (1,000–5,000 persons):**
```
Population:   1,445 persons
Solar:        51 kW
Battery:      24 kW / 143 kWh
Diesel:       3 kW (backup only)
LCOE:         0.099 EUR/kWh
Autarky:      97.8%
CO₂:          0.54 t/yr
```

**offgrid_cheaper = True** for most villages when grid distance > 50 km,
because the grid connection cost (15,000 EUR/km) far exceeds mini-grid costs.

---

## Difference Between build_offgrid.py and village_minigrids.py

| Feature | build_offgrid.py | village_minigrids.py |
|---------|-------|-------|
| Granularity | County level | Village level |
| Population input | County total | WorldPop raster |
| Clustering | None | DBSCAN |
| Mini-grids per county | 1 | 5–50+ |
| Spatial precision | County centroid | Village centroid |
| Data requirement | GADM only | GADM + WorldPop |
| Use case | Quick assessment | Detailed planning |

---

## References

| Reference | Usage |
|-----------|-------|
| WorldPop (2020), DOI: 10.5258/SOTON/WP00645 | Population raster |
| Ester et al. (1996) | DBSCAN clustering algorithm |
| Osiolo et al. (2019) | 60 kWh/yr energy demand rural Kenya |
| PyPSA technology-data (TU Berlin/DEA) | Solar/battery/diesel CAPEX |
| ERA5 – Hersbach et al. (2020), doi:10.1002/qj.3803 | Solar weather profile |
| ESMAP Mini Grid Design Manual (2019) | Annuity r=8%, n=20yr |
| IEA Africa Energy Outlook (2022) | Grid connection costs 15,000 EUR/km |
| IPCC Guidelines 2006 | Diesel CO₂ factor 0.27 t/MWh |
| OnSSET methodology (KTH) | C3 threshold 10 P/km² |
| KOSAP planning data | C1 threshold 50 km |
