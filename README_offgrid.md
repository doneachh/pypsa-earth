<!--
SPDX-FileCopyrightText: 2026 Mohamed Amine Chebaane

SPDX-License-Identifier: CC-BY-4.0
-->
# README – Offgrid-Integration in PyPSA-Earth

**Bachelorarbeit: Offgrid-Integration in PyPSA-Earth**
OTH Regensburg | Betreuer: Anton Achhammer, Prof. Michael Sterner

---

## Übersicht

Dieses Modul erweitert PyPSA-Earth um isolierte Mini-Grid Busse für Regionen
die noch nicht ans Stromnetz angeschlossen sind. Es besteht aus drei Teilen:

| Datei | Beschreibung |
|-------|-------------|
| `scripts/build_offgrid.py` | Hauptskript – Geo-Logik + Mini-Grid Integration |
| `config.yaml` | Konfiguration aller Parameter unter `offgrid:` |
| `Snakefile` | Snakemake-Regel `build_offgrid` nach `solve_network` |

---

## Schnellstart

```bash
# 1. Offgrid aktivieren (in config.yaml line 882):
#    offgrid:
#      enable: true

# 2. Workflow ausführen:
snakemake solve_all_offgrid -j4

# 3. Oder direkt testen (ohne Snakemake):
python scripts/build_offgrid.py
```

---

## Voraussetzungen

- PyPSA-Earth bereits installiert und konfiguriert
- Gelöstes Baseline-Netzwerk: `results/networks/elec_s_{simpl}_{clusters}_ec_l{ll}_{opts}.nc`
- ERA5 Cutout: `cutouts/cutout-2013-era5.nc`
- GADM Shapes: `resources/shapes/gadm_shapes.geojson`
- (Optional) Elektrifizierungsdaten: `data/elec_rates/{COUNTRY}_electricity_access.csv`

---

## Konfiguration (`config.yaml`)

Alle Parameter werden unter dem Schlüssel `offgrid:` in `config.yaml` gesetzt.

```yaml
offgrid:

  # Feature an/aus
  enable: true    # false = Snakemake-Regel wird komplett übersprungen

  # ── Geo-Logik Kriterien ─────────────────────────────────────────
  # true  = Kriterium wird geprüft
  # false = Kriterium wird ignoriert (gilt als erfüllt)

  use_c1_distance:    true   # C1: Netzabstand > max_distance_km
  use_c2_elec_rate:   true   # C2: Elektrifizierungsrate < max_elec_rate
  use_c3_pop_density: true   # C3: Bevölkerungsdichte > min_pop_density
  use_c4_low_load:    false  # C4: Baseline-Last < c4_max_load_mw (optional)
  use_c5_congestion:  false  # C5: Leitungsauslastung > c5_line_loading (Szenario C)

  # ── Schwellenwerte ───────────────────────────────────────────────
  max_distance_km:   50     # C1 [km]   – Quelle: KOSAP-Planungsdaten
  max_elec_rate:     0.50   # C2 [0-1]  – Elektrifizierungsrate
  min_pop_density:   10     # C3 [P/km²]– Quelle: OnSSET-Methodik (KTH)
  c4_max_load_mw:    20.0   # C4 [MW]   – Baseline-Last Schwellenwert
  c5_line_loading:   0.50   # C5 [0-1]  – Leitungsauslastung

  # ── Nachfrage ────────────────────────────────────────────────────
  kwh_per_person_yr: 60     # [kWh/yr]  – Quelle: Osiolo et al. (2019)

  # ── Technologie CAPEX ────────────────────────────────────────────
  # Overnight-Kosten (werden intern auf EUR/MW/a annualisiert)
  solar_capex:       800    # [EUR/kW]  – PyPSA technology-data (TU Berlin/DEA)
  battery_capex_kwh: 250    # [EUR/kWh] – PyPSA technology-data
  battery_max_hours: 6      # [h]       – Maximale Speicherdauer
  diesel_capex:      400    # [EUR/kW]  – PyPSA technology-data
  diesel_marginal:   300    # [EUR/MWh] – Betriebskosten
  diesel_co2:        0.27   # [t/MWh]  – Quelle: IPCC Guidelines 2006
  shedding_cost:     5000   # [EUR/MWh] – Lastabwurf-Strafkosten

  # ── Annuität ─────────────────────────────────────────────────────
  discount_rate:     0.071   # 7,1%  – Quelle: ESMAP Mini Grid Design Manual (2019)
  asset_lifetime:    20     # [Jahre]

  # ── Solver ───────────────────────────────────────────────────────
  solver: gurobi            # gurobi, highs, cplex
```

---
# README – Off-grid Integration in PyPSA-Earth

**Bachelor's Thesis: Off-grid Integration in PyPSA-Earth**
OTH Regensburg | Supervisors: Anton Achhammer, Prof. Michael Sterner

---

## Overview

This module extends PyPSA-Earth with isolated mini-grid buses for regions
that are not yet connected to the power grid. It consists of three parts:

| File | Description |
|------|-------------|
| `scripts/build_offgrid.py` | Main script – geo logic + mini-grid integration |
| `config.yaml` | Configuration of all parameters under `offgrid:` |
| `Snakefile` | Snakemake rule `build_offgrid` after `solve_network` |

---

## Quick Start

```bash
# 1. Enable off-grid (in config.yaml line 882):
#    offgrid:
#      enable: true

# 2. Run the workflow:
snakemake solve_all_offgrid -j4

# 3. Or test it directly (without Snakemake):
python scripts/build_offgrid.py
```

---

## Prerequisites

- PyPSA-Earth already installed and configured
- Solved baseline network: `results/networks/elec_s_{simpl}_{clusters}_ec_l{ll}_{opts}.nc`
- ERA5 cutout: `cutouts/cutout-2013-era5.nc`
- GADM shapes: `resources/shapes/gadm_shapes.geojson`
- (Optional) Electrification data: `data/elec_rates/{COUNTRY}_electricity_access.csv`

---

## Configuration (`config.yaml`)

All parameters are set under the `offgrid:` key in `config.yaml`.

```yaml
offgrid:

  # Feature on/off
  enable: true    # false = Snakemake rule is skipped entirely

  # ── Geo-logic criteria ──────────────────────────────────────────
  # true  = criterion is checked
  # false = criterion is ignored (treated as satisfied)

  use_c1_distance:    true   # C1: grid distance > max_distance_km
  use_c2_elec_rate:   true   # C2: electrification rate < max_elec_rate
  use_c3_pop_density: true   # C3: population density > min_pop_density
  use_c4_low_load:    false  # C4: baseline load < c4_max_load_mw (optional)
  use_c5_congestion:  false  # C5: line loading > c5_line_loading (Scenario C)

  # ── Threshold values ─────────────────────────────────────────────
  max_distance_km:   50     # C1 [km]   – Source: KOSAP planning data
  max_elec_rate:     0.50   # C2 [0-1]  – electrification rate
  min_pop_density:   10     # C3 [P/km²]– Source: OnSSET methodology (KTH)
  c4_max_load_mw:    20.0   # C4 [MW]   – baseline load threshold
  c5_line_loading:   0.50   # C5 [0-1]  – line loading

  # ── Demand ───────────────────────────────────────────────────────
  kwh_per_person_yr: 60     # [kWh/yr]  – Source: Osiolo et al. (2019)

  # ── Technology CAPEX ─────────────────────────────────────────────
  # Overnight costs (annualized internally to EUR/MW/yr)
  solar_capex:       800    # [EUR/kW]  – PyPSA technology-data (TU Berlin/DEA)
  battery_capex_kwh: 250    # [EUR/kWh] – PyPSA technology-data
  battery_max_hours: 6      # [h]       – maximum storage duration
  diesel_capex:      400    # [EUR/kW]  – PyPSA technology-data
  diesel_marginal:   300    # [EUR/MWh] – operating costs
  diesel_co2:        0.27   # [t/MWh]  – Source: IPCC Guidelines 2006
  shedding_cost:     5000   # [EUR/MWh] – load-shedding penalty cost

  # ── Annuity ──────────────────────────────────────────────────────
  discount_rate:     0.071   # 7.1%  – Source: ESMAP Mini Grid Design Manual (2019)
  asset_lifetime:    20     # [years]

  # ── Solver ───────────────────────────────────────────────────────
  solver: gurobi            # gurobi, highs, cplex
```

---

## Geo-logic Criteria (C1–C5)

The following criteria are checked for each region in `gadm_shapes.geojson`.
Only regions that satisfy **all active criteria** are identified as off-grid candidates.

| Criterion | Description | Default | Source |
|-----------|-------------|---------|--------|
| **C1** | Grid distance > 50 km | active | KOSAP planning data |
| **C2** | Electrification rate < 50% | active | Kenya Census 2019 / heuristic |
| **C3** | Population density > 10 P/km² | active | OnSSET methodology (KTH) |
| **C4** | Baseline load < 20 MW | inactive | optional |
| **C5** | Line loading > 50% | inactive | Scenario C (off-grid push) |

### Electrification Rate – Priorities

```
1. Census data from CSV    → data/elec_rates/{COUNTRY}_electricity_access.csv
2. Heuristic               → calculated from grid distance + population density
                             (used when no CSV is available OR
                              the region is not contained in the CSV)
```

**CSV format** (`KE_electricity_access.csv`):
```
GADM_ID,                Access to electricity
National Average,       38.0
KE.1_1,                 9.6
KE.7_1,                 11.6
```

---

## Mini-Grid Setup

For each identified off-grid region, an **isolated bus** is inserted into the network:

```
offgrid_{GADM_ID}  (400V low voltage, no link to the main grid)
    ├── Solar PV      (ERA5 weather profile, p_nom_extendable)
    ├── Battery       (6h storage, p_nom_extendable)
    ├── Diesel        (backup generator, p_nom_extendable)
    └── Load          (60 kWh/yr × population, daily profile)
```

> **Important:** The mini-grid buses have **no link to the main grid**.
> They are fully isolated in energy terms but part of the same PyPSA network object.

### PyPSA Units

| Quantity | Unit |
|----------|------|
| Power (`p_set`, `p_nom`) | MW |
| Energy | MWh |
| `capital_cost` | EUR/MW/yr (annualized) |
| `marginal_cost` | EUR/MWh |

The CAPEX in EUR/kW is converted internally:
```
capital_cost [EUR/MW/yr] = CAPEX [EUR/kW] × 1000 × annuity_factor
```

---

## Workflow

```
config.yaml          →  Set parameters (enable: true)
      ↓
solve_network        →  Solve baseline network (Gurobi)
      ↓
build_offgrid        →  Geo logic + mini-grid integration + optimization
      ↓
elec_..._offgrid.nc  →  Extended network
offgrid_results.csv  →  Results per region
```

### Why after `solve_network`?

The geo-logic criteria **C4** and **C5** require the optimization results
of the baseline network (`p_nom_opt`, `lines_t.p0`). Since the mini-grids are fully
isolated, their integration does not affect the main grid – a second
solve of the main grid is not necessary.

---

## Snakemake Commands

```bash

# Build all off-grid networks:
snakemake solve_all_offgrid -j4

# Test directly (without Snakemake):
python scripts/build_offgrid.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/networks/elec_s_{...}_offgrid.nc` | Extended PyPSA network with mini-grid buses |
| `results/offgrid_results_s_{...}.csv` | Results per off-grid region |

### Result CSV Columns

| Column | Unit | Description |
|--------|------|-------------|
| `region` | – | GADM_ID of the region |
| `population` | people | population |
| `distance_km` | km | distance to the main grid |
| `elec_rate` | 0–1 | electrification rate |
| `elec_source` | – | Census / heuristic |
| `solar_kw` | kW | installed solar capacity |
| `battery_kw` | kW | installed battery power |
| `battery_kwh` | kWh | installed battery energy |
| `diesel_kw` | kW | installed diesel capacity |
| `lcoe_eur_kwh` | EUR/kWh | levelized cost of energy |
| `autarky_pct` | % | solar share of total generation |
| `co2_t_yr` | t/yr | CO₂ emissions from diesel |
| `capex_total_keur` | kEUR | total investment costs |
| `total_cost_keur_yr` | kEUR/yr | annual total costs |
| `grid_capex_keur` | kEUR | grid connection CAPEX (for comparison) |
| `grid_total_keur_yr` | kEUR/yr | annual grid connection costs |
| `offgrid_cheaper` | bool | True = off-grid cheaper than grid connection |

---

## Analyzing the Results

The Jupyter notebook `offgrid_analyse.ipynb` provides a complete analysis:

```bash
conda activate pypsa-earth-kenya
jupyter notebook offgrid_analyse.ipynb
```

Contents:
- Load network and overview
- Mini-grid capacities
- KPI comparison (LCOE, autarky, CO₂)
- Time series (solar, load, battery)
- Cost comparison: off-grid vs. grid connection
- Map of the off-grid regions

---

## Using It for Other Countries

The script is fully generic:

```yaml
# config.yaml – change the country:
countries: ["NG"]   # Nigeria instead of Kenya

# Provide electrification data:
# data/elec_rates/NG_electricity_access.csv
# (if not available → heuristic is used automatically)
```

---

## Sources

| Source | Use |
|--------|-----|
| Osiolo et al. (2019) | 60 kWh/yr energy demand, rural Kenya |
| PyPSA technology-data (TU Berlin/DEA) | Solar/battery/diesel CAPEX |
| ERA5 – Hersbach et al. (2020), doi:10.1002/qj.3803 | Solar weather profile |
| ESMAP Mini Grid Design Manual (2019) | Annuity r=7.1%, n=20yr |
| IEA Africa Energy Outlook (2022) | Grid connection cost 15,000 EUR/km |
| Kenya Census 2019 | Electrification rates per county |
| World Bank (2022) | Electrification rate heuristic |
| KOSAP planning data | C1 threshold 50 km |
| OnSSET methodology (KTH) | C3 threshold 10 P/km² |
| IPCC Guidelines 2006 | Diesel CO₂ factor 0.27 t/MWh |
## Geo-Logik Kriterien (C1–C5)

Für jede Region in `gadm_shapes.geojson` werden folgende Kriterien geprüft.
Nur Regionen die **alle aktiven Kriterien** erfüllen werden als Offgrid-Kandidaten identifiziert.

| Kriterium | Beschreibung | Standard | Quelle |
|-----------|-------------|---------|--------|
| **C1** | Netzabstand > 50 km | aktiv | KOSAP-Planungsdaten |
| **C2** | Elektrifizierungsrate < 50% | aktiv | Kenya Census 2019 / Heuristik |
| **C3** | Bevölkerungsdichte > 10 P/km² | aktiv | OnSSET-Methodik (KTH) |
| **C4** | Baseline-Last < 20 MW | inaktiv | Optional |
| **C5** | Leitungsauslastung > 50% | inaktiv | Szenario C (Offgrid-Push) |

### Elektrifizierungsrate – Prioritäten

```
1. Census-Daten aus CSV    → data/elec_rates/{COUNTRY}_electricity_access.csv
2. Heuristik               → berechnet aus Netzabstand + Bevölkerungsdichte
                             (wird genutzt wenn keine CSV vorhanden ODER
                              Region nicht in CSV enthalten)
```

**CSV Format** (`KE_electricity_access.csv`):
```
GADM_ID,                Access to electricity
National Average,       38.0
KE.1_1,                 9.6
KE.7_1,                 11.6
```

---

## Mini-Grid Aufbau

Für jede identifizierte Offgrid-Region wird ein **isolierter Bus** ins Netzwerk eingefügt:

```
offgrid_{GADM_ID}  (400V Niederspannung, kein Link zum Hauptnetz)
    ├── Solar PV      (ERA5 Wetterprofil, p_nom_extendable)
    ├── Batterie      (6h Speicher, p_nom_extendable)
    ├── Diesel        (Backup Generator, p_nom_extendable)
    └── Last          (60 kWh/yr × Bevölkerung, Tagesprofil)
```

> **Wichtig:** Die Mini-Grid Busse haben **keinen Link zum Hauptnetz**.
> Sie sind energetisch vollständig isoliert aber Teil desselben PyPSA-Netzwerk-Objekts.

### PyPSA Einheiten

| Größe | Einheit |
|-------|---------|
| Leistung (`p_set`, `p_nom`) | MW |
| Energie | MWh |
| `capital_cost` | EUR/MW/a (annualisiert) |
| `marginal_cost` | EUR/MWh |

Der CAPEX in EUR/kW wird intern umgerechnet:
```
capital_cost [EUR/MW/a] = CAPEX [EUR/kW] × 1000 × annuity_factor
```

---

## Workflow

```
config.yaml          →  Parameter setzen (enable: true)
      ↓
solve_network        →  Baseline-Netzwerk lösen (Gurobi)
      ↓
build_offgrid        →  Geo-Logik + Mini-Grid Integration + Optimierung
      ↓
elec_..._offgrid.nc  →  Erweitertes Netzwerk
offgrid_results.csv  →  Ergebnisse pro Region
```

### Warum nach `solve_network`?

Die Geo-Logik-Kriterien **C4** und **C5** benötigen die Optimierungsergebnisse
des Baseline-Netzwerks (`p_nom_opt`, `lines_t.p0`). Da die Mini-Grids vollständig
isoliert sind, beeinflusst ihre Integration das Hauptnetz nicht – ein zweites
Solve des Hauptnetzes ist nicht nötig.

---

## Snakemake Befehle

```bash

# Alle Offgrid-Netzwerke bauen:
snakemake solve_all_offgrid -j4

# Direkt testen (ohne Snakemake):
python scripts/build_offgrid.py
```

---

## Output Dateien

| Datei | Beschreibung |
|-------|-------------|
| `results/networks/elec_s_{...}_offgrid.nc` | Erweitertes PyPSA-Netzwerk mit Mini-Grid Bussen |
| `results/offgrid_results_s_{...}.csv` | Ergebnisse pro Offgrid-Region |

### Ergebnis-CSV Spalten

| Spalte | Einheit | Beschreibung |
|--------|---------|-------------|
| `region` | – | GADM_ID der Region |
| `population` | Personen | Bevölkerung |
| `distance_km` | km | Abstand zum Hauptnetz |
| `elec_rate` | 0–1 | Elektrifizierungsrate |
| `elec_source` | – | Census / Heuristik |
| `solar_kw` | kW | Installierte Solar-Kapazität |
| `battery_kw` | kW | Installierte Batterie-Leistung |
| `battery_kwh` | kWh | Installierte Batterie-Energie |
| `diesel_kw` | kW | Installierte Diesel-Kapazität |
| `lcoe_eur_kwh` | EUR/kWh | Levelized Cost of Energy |
| `autarky_pct` | % | Solar-Anteil an Gesamterzeugung |
| `co2_t_yr` | t/yr | CO₂-Emissionen durch Diesel |
| `capex_total_keur` | kEUR | Gesamte Investitionskosten |
| `total_cost_keur_yr` | kEUR/yr | Jährliche Gesamtkosten |
| `grid_capex_keur` | kEUR | Netzanschluss CAPEX (zum Vergleich) |
| `grid_total_keur_yr` | kEUR/yr | Jährliche Netzanschlusskosten |
| `offgrid_cheaper` | bool | True = Offgrid günstiger als Netzanschluss |

---

## Ergebnisse analysieren

Das Jupyter Notebook `offgrid_analyse.ipynb` bietet eine vollständige Analyse:

```bash
conda activate pypsa-earth-kenya
jupyter notebook offgrid_analyse.ipynb
```

Inhalte:
- Netzwerk laden und Überblick
- Kapazitäten der Mini-Grids
- KPI Vergleich (LCOE, Autarkie, CO₂)
- Zeitreihen (Solar, Last, Batterie)
- Kostenvergleich Offgrid vs. Netzanschluss
- Karte der Offgrid-Regionen

---

## Für andere Länder nutzen

Das Skript ist vollständig generisch:

```yaml
# config.yaml – Land ändern:
countries: ["NG"]   # Nigeria statt Kenya

# Elektrifizierungsdaten bereitstellen:
# data/elec_rates/NG_electricity_access.csv
# (falls nicht vorhanden → automatisch Heuristik)
```

---

## Quellen

| Quelle | Verwendung |
|--------|-----------|
| Osiolo et al. (2019) | 60 kWh/yr Energiebedarf ländliches Kenia |
| PyPSA technology-data (TU Berlin/DEA) | Solar/Batterie/Diesel CAPEX |
| ERA5 – Hersbach et al. (2020), doi:10.1002/qj.3803 | Solar Wetterprofil |
| ESMAP Mini Grid Design Manual (2019) | Annuität r=7,1%, n=20yr |
| IEA Africa Energy Outlook (2022) | Netzanschlusskosten 15.000 EUR/km |
| Kenya Census 2019 | Elektrifizierungsraten pro County |
| Weltbank (2022) | Heuristik Elektrifizierungsrate |
| KOSAP-Planungsdaten | C1 Schwellenwert 50 km |
| OnSSET-Methodik (KTH) | C3 Schwellenwert 10 P/km² |
| IPCC Guidelines 2006 | Diesel CO₂-Faktor 0.27 t/MWh |
