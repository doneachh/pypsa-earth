# SPDX-FileCopyrightText: 2026 Mohamed Amine Chebaane

# SPDX-License-Identifier: CC-BY-4.0

"""
village_minigrids.py
========================
Bachelorarbeit: Offgrid-Integration in PyPSA-Earth
OTH Regensburg | Betreuer: Anton Achhammer, Prof. Michael Sterner

Erweiterung von build_offgrid.py (geo_logic_multi_minigrid.py):
Statt ein grosses Mini-Grid pro County → viele kleine Mini-Grids pro Dorf/Siedlung

Ablauf:
    1. Geo-Logik (Layer 1): Offgrid-Counties identifizieren
    2. WorldPop Raster: Unelektrifizierte Siedlungszellen finden
    3. DBSCAN Clustering: Nahe Zellen zu Dorf-Clustern gruppieren
    4. Tech-Logik (Layer 2): Mini-Grid pro Dorf optimieren

EINHEITEN in PyPSA (Quelle: PyPSA Dokumentation, docs.pypsa.org):
    - Leistung p_set, p_nom:  MW
    - Energie:                MWh
    - capital_cost:           EUR/MW/a  (annualisiert)
    - marginal_cost:          EUR/MWh
    Umrechnungen im Code:
    - Last:         kW / 1000 = MW
    - capital_cost: EUR/kW × 1000 × annuity_factor = EUR/MW/a
    - LCOE:         n.objective [EUR/a] / lts.sum() [MWh/a] / 1000 = EUR/kWh

Quellen:
    - WorldPop (2020): ken_ppp_2020_UNadj_constrained.tif
      University of Southampton, DOI: 10.5258/SOTON/WP00645
    - DBSCAN: Ester et al. (1996), density-based clustering
    - Osiolo et al. (2019): 60 kWh/yr rural Kenya
    - PyPSA technology-data (TU Berlin/DEA): Solar/Batterie CAPEX
    - ERA5: Hersbach et al. (2020), doi:10.1002/qj.3803
    - IEA Africa Energy Outlook (2022): Netzanschlusskosten
    - ESMAP Mini Grid Design Manual (2019): Annuität r=8%, n=20yr
"""

import glob
import os
import warnings

import atlite
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import rasterio
import rasterio.mask
from rasterio.transform import rowcol
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════
CONFIG = {
    # Dateipfade
    "shapes_file": "resources/shapes/gadm_shapes.geojson",
    "network_pattern": "results/networks/*.nc",
    "cutout_file": "cutouts/cutout-2013-era5.nc",
    "elec_data_dir": "data/elec_rates",
    "worldpop_file": "data/WorldPop/ken_ppp_2020_UNadj_constrained.tif",
    # Geo-Logik Schwellenwerte (Layer 1) – County-Ebene
    "max_distance_km": 50,  # C1: Quelle: KOSAP-Planungsdaten
    "max_elec_rate": 0.50,  # C2: Schwellenwert Elektrifizierungsrate
    "min_pop_density": 10,  # C3: Quelle: OnSSET-Methodik (KTH)
    "use_c1_distance": True,
    "use_c2_elec_rate": True,
    "use_c3_pop_density": True,
    "use_c4_low_load": False,
    "c4_max_load_mw": 20.0,
    # Siedlungs-Clustering (WorldPop → Dörfer)
    "min_pop_per_cell": 5,  # Mindest-Bevoelkerung pro WorldPop-Zelle
    "min_village_pop": 1000,  # Mindest-Bevoelkerung pro Dorf-Cluster
    "max_village_pop": 50000,  # Max-Bevoelkerung pro Dorf-Cluster
    "cluster_radius_km": 3.0,  # DBSCAN: Radius fuer Nachbarschaft [km]
    "cluster_min_cells": 2,  # DBSCAN: Mindestanzahl Zellen pro Cluster
    # Nachfrage
    "kwh_per_person_yr": 60,  # Quelle: Osiolo et al. (2019)
    # Technologie CAPEX (Overnight-Kosten in EUR/kW bzw. EUR/kWh)
    # Werden in build_minigrid × 1000 auf EUR/MW/a umgerechnet (PyPSA-Einheit)
    "solar_capex": 800,  # EUR/kW  – PyPSA technology-data
    "battery_capex_kwh": 250,  # EUR/kWh – PyPSA technology-data
    "battery_max_hours": 6,
    "diesel_capex": 400,  # EUR/kW  – PyPSA technology-data
    "diesel_marginal": 300,  # EUR/MWh – PyPSA marginal_cost Einheit
    "diesel_co2": 0.27,  # t CO2/MWh_el – IPCC Guidelines 2006
    "shedding_cost": 5000,  # EUR/MWh – PyPSA marginal_cost Einheit
    "solver": "gurobi",
    # Kapazitaetsbegrenzungen
    "solar_nom_max_factor": None,
    "battery_nom_max_factor": None,
    # Netzanschlusskosten – Quelle: IEA Africa Energy Outlook (2022)
    "grid_line_cost_per_km": 15000,  # EUR/km
    "grid_substation_cost": 35000,  # EUR pauschal – ESMAP (2019)
    "grid_opex_rate": 0.03,  # 3%/Jahr – Weltbank (2022)
    # Annuität – Quelle: ESMAP Mini Grid Design Manual (2019)
    "discount_rate": 0.071,  # 7,1%
    "asset_lifetime": 20,  # Jahre
    # Ziel-Counties fuer Optimierung
    "target_counties": ["KE.37_1"],
}


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Elektrifizierungsdaten
# ══════════════════════════════════════════════════════


def load_electrification_data(country_code, elec_data_dir):
    path = os.path.join(elec_data_dir, f"{country_code}_electricity_access.csv")
    if not os.path.exists(path):
        print(f"      ⚠️  Keine echten Daten fuer {country_code} → Heuristik")
        return None, 0.23
    df = pd.read_csv(path)
    nat_row = df[df["GADM_ID"] == "National Average"]["Access to electricity"].values
    nat_avg = float(nat_row[0]) / 100 if len(nat_row) > 0 else 0.23
    df_regions = df[df["GADM_ID"] != "National Average"].dropna(subset=["GADM_ID"])
    elec_dict = dict(
        zip(
            df_regions["GADM_ID"].str.strip(),
            df_regions["Access to electricity"].astype(float) / 100,
        )
    )
    print(f"      → Elektrifizierungsdaten geladen: {len(elec_dict)} Regionen ✅")
    print(f"      → Nationaler Durchschnitt: {nat_avg*100:.1f}%")
    return elec_dict, nat_avg


def get_electrification_rate(
    gadm_id, distance_km, pop_density, elec_data=None, national_avg=0.23
):
    if elec_data is not None:
        if gadm_id in elec_data:
            return elec_data[gadm_id], "Census"
        return national_avg, "National Avg"
    if distance_km > 100:
        base_rate = 0.15
    elif distance_km > 50:
        base_rate = 0.30
    elif distance_km > 25:
        base_rate = 0.55
    else:
        base_rate = 0.76
    if pop_density > 500:
        density_factor = 1.2
    elif pop_density > 100:
        density_factor = 1.0
    elif pop_density > 25:
        density_factor = 0.8
    else:
        density_factor = 0.6
    return round(min(base_rate * density_factor, 1.0), 4), "Heuristik"


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – WorldPop Clustering
# ══════════════════════════════════════════════════════


def extract_unelectrified_cells(worldpop_file, county_geometry, bus_geom, config):
    """
    Liest WorldPop Rasterzellen fuer ein County und filtert:
    - Zellen mit genuegend Bevoelkerung
    - Zellen weiter als max_distance_km vom Netz
    Quelle: WorldPop (2020), DOI: 10.5258/SOTON/WP00645
    """
    with rasterio.open(worldpop_file) as src:
        try:
            out_image, out_transform = rasterio.mask.mask(
                src, [county_geometry], crop=True, nodata=-99999
            )
            data = out_image[0]
        except Exception as e:
            return gpd.GeoDataFrame()

    rows, cols = np.where(data > config["min_pop_per_cell"])
    if len(rows) == 0:
        return gpd.GeoDataFrame()

    from rasterio.transform import xy

    xs, ys = xy(out_transform, rows, cols)
    pops = data[rows, cols]

    from shapely.geometry import Point

    cells = gpd.GeoDataFrame(
        {"pop": pops, "geometry": [Point(x, y) for x, y in zip(xs, ys)]},
        crs="EPSG:4326",
    )

    cells["dist_km"] = cells.geometry.apply(
        lambda g: (bus_geom.geometry.distance(g) * 111).min()
    )
    unelec = cells[cells["dist_km"] > config["max_distance_km"]].copy()
    return unelec


def cluster_villages(unelec_cells, config):
    """
    Gruppiert unelektrifizierte Zellen zu Dorf-Clustern via DBSCAN.
    Quelle: Ester et al. (1996), density-based clustering
    """
    if len(unelec_cells) == 0:
        return gpd.GeoDataFrame()

    coords = np.array([[g.x, g.y] for g in unelec_cells.geometry])
    eps_deg = config["cluster_radius_km"] / 111.0
    db = DBSCAN(eps=eps_deg, min_samples=config["cluster_min_cells"]).fit(coords)
    unelec_cells = unelec_cells.copy()
    unelec_cells["cluster"] = db.labels_

    valid = unelec_cells[unelec_cells["cluster"] >= 0]
    if len(valid) == 0:
        return gpd.GeoDataFrame()

    villages = []
    for cid, group in valid.groupby("cluster"):
        pop = group["pop"].sum()
        centroid_x = group.geometry.x.mean()
        centroid_y = group.geometry.y.mean()
        avg_dist = group["dist_km"].mean()
        n_cells = len(group)

        if pop < config["min_village_pop"]:
            continue
        if pop > config["max_village_pop"]:
            continue

        villages.append(
            {
                "cluster_id": cid,
                "population": int(pop),
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "dist_km": round(avg_dist, 1),
                "n_cells": n_cells,
            }
        )

    if not villages:
        return gpd.GeoDataFrame()

    from shapely.geometry import Point

    vdf = pd.DataFrame(villages)
    vdf["geometry"] = [Point(r.centroid_x, r.centroid_y) for _, r in vdf.iterrows()]
    vdf = gpd.GeoDataFrame(vdf, geometry="geometry", crs="EPSG:4326")
    return vdf


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – ERA5 Solar + Mini-Grid
# ══════════════════════════════════════════════════════


def get_solar_profile_from_era5(centroid_x, centroid_y, cutout):
    cutout_point = cutout.sel(
        x=slice(centroid_x - 0.3, centroid_x + 0.3),
        y=slice(centroid_y - 0.3, centroid_y + 0.3),
    )
    influx = cutout_point.data["influx_direct"] + cutout_point.data["influx_diffuse"]
    influx_ts = influx.mean(dim=["x", "y"]).values
    max_val = influx_ts.max()
    if max_val > 0:
        return influx_ts / max_val
    return solar_profile_fallback()


def load_profile(n_hours=8760):
    hours = np.arange(n_hours)
    hod = hours % 24
    p = np.ones(n_hours) * 0.6
    p[hod >= 6] = 0.8
    p[hod >= 9] = 0.6
    p[hod >= 17] = 1.0
    p[hod >= 21] = 0.7
    p[hod >= 23] = 0.4
    return p / p.mean()


def solar_profile_fallback(n_hours=8760):
    hours = np.arange(n_hours)
    hod = hours % 24
    s = np.zeros(n_hours)
    s[hod >= 6] = 0.3
    s[hod >= 9] = 0.7
    s[hod >= 11] = 0.9
    s[hod >= 13] = 0.8
    s[hod >= 16] = 0.4
    s[hod >= 18] = 0.0
    return s


def build_minigrid(
    village_id,
    population,
    centroid_x,
    centroid_y,
    config,
    cutout=None,
    annuity_factor=0.1019,
):
    """
    Baut und optimiert ein isoliertes Mini-Grid fuer ein Dorf.

    PyPSA arbeitet in MW/MWh (Quelle: docs.pypsa.org/stable/design.html):
      - p_set / p_nom:  MW
      - capital_cost:   EUR/MW/a  (annualisiert)
      - marginal_cost:  EUR/MWh

    Umrechnungen:
      - Last:          kW / 1000 = MW
      - capital_cost:  EUR/kW × 1000 × annuity_factor = EUR/MW/a
      - LCOE:          n.objective [EUR/a] / lts.sum() [MWh/a] / 1000 = EUR/kWh
    """
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2013-01-01", periods=8760, freq="h"))
    for c in ["AC", "solar", "battery", "diesel", "load_shedding"]:
        n.add("Carrier", c)

    bus = f"bus_{village_id}"
    n.add("Bus", bus, carrier="AC", x=centroid_x, y=centroid_y)

    # Last in MW: population × kWh/yr / 8760h / 1000 = MW
    avg_load_mw = population * config["kwh_per_person_yr"] / 8760 / 1000
    lp = load_profile() * avg_load_mw  # MW (stündlich)
    lts = pd.Series(lp, index=n.snapshots)  # MW
    n.add("Load", f"load_{village_id}", bus=bus, p_set=lts)

    if cutout is not None:
        try:
            sp = get_solar_profile_from_era5(centroid_x, centroid_y, cutout)
            solar_src = "ERA5"
        except:
            sp = solar_profile_fallback()
            solar_src = "Fallback"
    else:
        sp = solar_profile_fallback()
        solar_src = "Fallback"

    sts = pd.Series(sp, index=n.snapshots)

    solar_nom_max = (
        avg_load_mw * config["solar_nom_max_factor"]
        if config["solar_nom_max_factor"] is not None
        else float("inf")
    )

    # Solar capital_cost: EUR/kW × 1000 × annuity_factor = EUR/MW/a
    n.add(
        "Generator",
        f"solar_{village_id}",
        bus=bus,
        carrier="solar",
        p_nom_extendable=True,
        p_nom_max=solar_nom_max,
        p_max_pu=sts,
        capital_cost=config["solar_capex"] * 1000 * annuity_factor,
        marginal_cost=0.01,
    )  # EUR/MWh

    battery_nom_max = (
        avg_load_mw * config["battery_nom_max_factor"]
        if config["battery_nom_max_factor"] is not None
        else float("inf")
    )

    # Batterie capital_cost: EUR/kWh × 1000 × max_hours × annuity_factor = EUR/MW/a
    n.add(
        "StorageUnit",
        f"battery_{village_id}",
        bus=bus,
        carrier="battery",
        p_nom_extendable=True,
        p_nom_max=battery_nom_max,
        max_hours=config["battery_max_hours"],
        capital_cost=config["battery_capex_kwh"]
        * 1000
        * config["battery_max_hours"]
        * annuity_factor,
        marginal_cost=1,  # EUR/MWh
        efficiency_store=0.95,
        efficiency_dispatch=0.95,
        cyclic_state_of_charge=True,
    )

    # Diesel capital_cost: EUR/kW × 1000 × annuity_factor = EUR/MW/a
    n.add(
        "Generator",
        f"diesel_{village_id}",
        bus=bus,
        carrier="diesel",
        p_nom_extendable=True,
        capital_cost=config["diesel_capex"] * 1000 * annuity_factor,
        marginal_cost=config["diesel_marginal"],
    )  # EUR/MWh

    # Lastabwurf: kein CAPEX, hoher marginaler Preis
    n.add(
        "Generator",
        f"shedding_{village_id}",
        bus=bus,
        carrier="load_shedding",
        p_nom=1e6,
        p_nom_extendable=False,
        marginal_cost=config["shedding_cost"],
    )  # EUR/MWh

    n.optimize(solver_name=config["solver"])
    return n, lts, solar_src


# ══════════════════════════════════════════════════════
# SCHRITT 1: DATEN LADEN
# ══════════════════════════════════════════════════════
print("=" * 70)
print("C4 – VILLAGE MINI-GRIDS (WorldPop Clustering)")
print("=" * 70)

print("\n[1/5] Lade Shapes...")
shapes = gpd.read_file(CONFIG["shapes_file"])
country_code = shapes["country"].iloc[0] if "country" in shapes.columns else "KE"
print(f"      → {len(shapes)} Regionen | Land: {country_code}")

print("[1/5] Lade Baseline-Netzwerk...")
network_files = glob.glob(CONFIG["network_pattern"])
network_path = sorted(network_files)[-1]
print(f"      → {os.path.basename(network_path)}")
baseline = pypsa.Network(network_path)
buses = baseline.buses[["x", "y"]].copy()
from shapely.geometry import Point

bus_geom = gpd.GeoDataFrame(
    buses, geometry=gpd.points_from_xy(buses.x, buses.y), crs="EPSG:4326"
)
print(f"      → {len(buses)} Netzknoten")

bus_load_mw = {}
if len(baseline.loads_t.p_set.columns) > 0:
    avg_load = baseline.loads_t.p_set.mean()
    for load_name in avg_load.index:
        bus_name = baseline.loads.loc[load_name, "bus"]
        bus_load_mw[bus_name] = bus_load_mw.get(bus_name, 0) + avg_load[load_name]

print("[1/5] Lade ERA5 Wetterdaten...")
cutout = None
if os.path.exists(CONFIG["cutout_file"]):
    try:
        cutout = atlite.Cutout(CONFIG["cutout_file"])
        print(f"      → ERA5 geladen ✅")
    except Exception as e:
        print(f"      ⚠️  ERA5 Fehler: {e}")

print(f"[1/5] Lade Elektrifizierungsdaten ({country_code})...")
os.makedirs(CONFIG["elec_data_dir"], exist_ok=True)
elec_data, national_avg = load_electrification_data(
    country_code, CONFIG["elec_data_dir"]
)
elec_source_global = "Census" if elec_data else "Heuristik"

print(f"[1/5] WorldPop Raster: {CONFIG['worldpop_file']}")
if not os.path.exists(CONFIG["worldpop_file"]):
    raise FileNotFoundError(f"WorldPop nicht gefunden: {CONFIG['worldpop_file']}")
print(f"      → Gefunden ✅")


# ══════════════════════════════════════════════════════
# SCHRITT 2: GEO-LOGIK – OFFGRID COUNTIES
# ══════════════════════════════════════════════════════
print(f"\n[2/5] Geo-Logik: Offgrid-Counties identifizieren...")

offgrid_regions = []
skipped = []

for _, region in shapes.iterrows():
    gadm_id = region["GADM_ID"]
    centroid = region.geometry.centroid
    dist_km = (bus_geom.geometry.distance(centroid) * 111).min()
    area_km2 = region.geometry.area * (111**2)
    pop_density = region["pop"] / area_km2 if area_km2 > 0 else 0

    elec_rate, elec_src = get_electrification_rate(
        gadm_id, dist_km, pop_density, elec_data, national_avg
    )

    c1 = (dist_km > CONFIG["max_distance_km"]) if CONFIG["use_c1_distance"] else True
    c2 = (elec_rate < CONFIG["max_elec_rate"]) if CONFIG["use_c2_elec_rate"] else True
    c3 = (
        (pop_density > CONFIG["min_pop_density"])
        if CONFIG["use_c3_pop_density"]
        else True
    )
    nearest_bus = bus_geom.geometry.distance(centroid).idxmin()
    nearest_load = bus_load_mw.get(nearest_bus, 0.0)
    c4 = (
        (nearest_load < CONFIG["c4_max_load_mw"]) if CONFIG["use_c4_low_load"] else True
    )

    entry = {
        "gadm_id": gadm_id,
        "population": int(region["pop"]),
        "elec_rate": elec_rate,
        "elec_source": elec_src,
        "distance_km": round(dist_km, 1),
        "pop_density": round(pop_density, 1),
        "centroid_x": centroid.x,
        "centroid_y": centroid.y,
        "geometry": region.geometry,
    }
    if c1 and c2 and c3 and c4:
        offgrid_regions.append(entry)
    else:
        skipped.append(entry)

offgrid_df = pd.DataFrame(offgrid_regions)
print(f"      → {len(offgrid_df)} Offgrid-Counties | {len(skipped)} On-grid")

if CONFIG["target_counties"]:
    offgrid_df = offgrid_df[
        offgrid_df["gadm_id"].isin(CONFIG["target_counties"])
    ].reset_index(drop=True)
    print(f"      → Gefiltert auf: {CONFIG['target_counties']}")
    print(f"      → {len(offgrid_df)} Counties nach Filter")

if len(offgrid_df) > 0:
    print(
        offgrid_df[["gadm_id", "population", "distance_km", "elec_rate"]].to_string(
            index=False
        )
    )

if len(offgrid_df) == 0:
    print("⚠️  Keine Offgrid-Counties! Schwellenwerte anpassen.")
    exit()


# ══════════════════════════════════════════════════════
# SCHRITT 3: WORLDPOP CLUSTERING → DÖRFER
# ══════════════════════════════════════════════════════
print(f"\n[3/5] WorldPop Clustering: Dörfer identifizieren...")
print(f"      Radius:       {CONFIG['cluster_radius_km']} km")
print(f"      Min Pop/Dorf: {CONFIG['min_village_pop']}")
print(f"      Max Pop/Dorf: {CONFIG['max_village_pop']}")

all_villages = []

for idx, county in offgrid_df.iterrows():
    gadm_id = county["gadm_id"]
    print(f"\n      {gadm_id} ({county['population']:,} Einwohner):")

    unelec = extract_unelectrified_cells(
        CONFIG["worldpop_file"], county["geometry"], bus_geom, CONFIG
    )
    print(
        f"      → {len(unelec):,} unelektrifizierte Zellen (> {CONFIG['min_pop_per_cell']} P)"
    )

    if len(unelec) == 0:
        print(f"      ⚠️  Keine Zellen gefunden – ueberspringe")
        continue

    villages = cluster_villages(unelec, CONFIG)
    print(f"      → {len(villages)} Dorf-Cluster identifiziert")

    if len(villages) == 0:
        print(f"      ⚠️  Keine Cluster – ueberspringe")
        continue

    villages["gadm_id"] = gadm_id
    villages["elec_rate"] = county["elec_rate"]
    villages["elec_source"] = county["elec_source"]
    villages["county_pop"] = county["population"]

    print(
        f"      → Pop pro Dorf: min={villages['population'].min():,} "
        f"max={villages['population'].max():,} "
        f"mean={villages['population'].mean():.0f}"
    )

    all_villages.append(villages)

if not all_villages:
    print("⚠️  Keine Dörfer gefunden! Parameter anpassen.")
    exit()

villages_df = pd.concat(all_villages, ignore_index=True)
villages_df["village_id"] = [
    f"{row.gadm_id}_v{int(row.cluster_id):03d}" for _, row in villages_df.iterrows()
]

print(f"\n      GESAMT: {len(villages_df)} Dorf-Cluster in {len(offgrid_df)} Counties")
print(f"      Pop gesamt:   {villages_df['population'].sum():,}")
print(
    f"      Pop min/max:  {villages_df['population'].min():,} / {villages_df['population'].max():,}"
)


# ══════════════════════════════════════════════════════
# SCHRITT 4: TECH-LOGIK – MINI-GRID PRO DORF
# ══════════════════════════════════════════════════════
print(f"\n[4/5] Tech-Logik: {len(villages_df)} Mini-Grids optimieren...")

test_villages = villages_df
results = []

# Annuitätsfaktor – Quelle: ESMAP Mini Grid Design Manual (2019)
_r = CONFIG["discount_rate"]
_n = CONFIG["asset_lifetime"]
annuity_factor = _r * (1 + _r) ** _n / ((1 + _r) ** _n - 1)  # ≈ 0.1019
print(
    f"      Annuitätsfaktor: {annuity_factor:.4f} "
    f"(r={CONFIG['discount_rate']*100:.0f}%, n={CONFIG['asset_lifetime']} Jahre)"
)

for i, row in test_villages.iterrows():
    vid = row["village_id"]
    idx = list(test_villages.index).index(i) + 1
    print(
        f"\n      [{idx}/{len(test_villages)}] {vid}  "
        f"Pop: {row['population']:,}  Dist: {row['dist_km']} km"
    )

    try:
        n, load_ts, solar_src = build_minigrid(
            village_id=vid,
            population=row["population"],
            centroid_x=row["centroid_x"],
            centroid_y=row["centroid_y"],
            config=CONFIG,
            cutout=cutout,
            annuity_factor=annuity_factor,
        )

        # Kapazitäten aus PyPSA (MW) → kW für CSV
        solar_cap_mw = n.generators.loc[f"solar_{vid}", "p_nom_opt"]  # MW
        battery_cap_mw = n.storage_units.loc[f"battery_{vid}", "p_nom_opt"]  # MW
        diesel_cap_mw = n.generators.loc[f"diesel_{vid}", "p_nom_opt"]  # MW
        solar_cap_kw = solar_cap_mw * 1000  # kW
        battery_cap_kw = battery_cap_mw * 1000  # kW
        diesel_cap_kw = diesel_cap_mw * 1000  # kW

        # Erzeugung in MWh/yr
        solar_gen_mwh = n.generators_t.p[f"solar_{vid}"].sum()  # MWh/yr
        diesel_gen_mwh = n.generators_t.p[f"diesel_{vid}"].sum()  # MWh/yr

        # load_ts in MW → .sum() ergibt MWh/yr
        total_load_mwh = load_ts.sum()  # MWh/yr

        # LCOE: n.objective [EUR/yr] / total_load [MWh/yr] = EUR/MWh → /1000 = EUR/kWh
        lcoe = n.objective / total_load_mwh / 1000  # EUR/kWh
        total_gen_mwh = solar_gen_mwh + diesel_gen_mwh
        autarky = (solar_gen_mwh / total_gen_mwh * 100) if total_gen_mwh > 0 else 0.0
        co2_t = diesel_gen_mwh * CONFIG["diesel_co2"]  # t CO2/yr

        # Manuelle CAPEX-Berechnung (kW-Basis) zur Verifikation mit Netzkosten
        capex_solar = solar_cap_kw * CONFIG["solar_capex"]
        capex_battery = (
            battery_cap_kw * CONFIG["battery_max_hours"] * CONFIG["battery_capex_kwh"]
        )
        capex_diesel = diesel_cap_kw * CONFIG["diesel_capex"]
        capex_total = capex_solar + capex_battery + capex_diesel
        opex_total = (capex_solar + capex_battery) * 0.01  # 1% OPEX/yr
        capex_ann = capex_total * annuity_factor
        total_cost_yr = capex_ann + opex_total

        # Netzanschluss Kosten
        grid_line = row["dist_km"] * CONFIG["grid_line_cost_per_km"]
        grid_sub = CONFIG["grid_substation_cost"]
        grid_capex = grid_line + grid_sub
        grid_opex_yr = grid_capex * CONFIG["grid_opex_rate"]
        grid_ann_yr = grid_capex * annuity_factor
        grid_total_yr = grid_ann_yr + grid_opex_yr
        offgrid_cheaper = total_cost_yr < grid_total_yr

        results.append(
            {
                "village_id": vid,
                "gadm_id": row["gadm_id"],
                "population": row["population"],
                "dist_km": row["dist_km"],
                "n_cells": row["n_cells"],
                "elec_rate": row["elec_rate"],
                "solar_src": solar_src,
                "solar_kw": round(solar_cap_kw, 2),
                "battery_kw": round(battery_cap_kw, 2),
                "battery_kwh": round(battery_cap_kw * CONFIG["battery_max_hours"], 2),
                "diesel_kw": round(diesel_cap_kw, 2),
                "lcoe_eur_kwh": round(lcoe, 3),
                "autarky_pct": round(autarky, 1),
                "co2_t_yr": round(co2_t, 3),
                "capex_total_eur": round(capex_total, 0),
                "total_cost_eur_yr": round(total_cost_yr, 0),
                "grid_capex_eur": round(grid_capex, 0),
                "grid_total_eur_yr": round(grid_total_yr, 0),
                "offgrid_cheaper": offgrid_cheaper,
                "centroid_x": row["centroid_x"],
                "centroid_y": row["centroid_y"],
            }
        )
        cheaper_str = (
            "✅ Offgrid guenstiger" if offgrid_cheaper else "⚠️  Netz guenstiger"
        )
        print(
            f"         ✅ Solar: {solar_cap_kw:.1f} kW  "
            f"Bat: {battery_cap_kw:.1f} kW  "
            f"LCOE: {lcoe:.3f} EUR/kWh  "
            f"CAPEX: {capex_total/1000:.1f} kEUR  "
            f"{cheaper_str}"
        )

    except Exception as e:
        print(f"         ❌ Fehler: {e}")

results_df = pd.DataFrame(results)


# ══════════════════════════════════════════════════════
# SCHRITT 5: ERGEBNISSE
# ══════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ERGEBNISSE – VILLAGE MINI-GRIDS")
print("=" * 70)

if len(results_df) > 0:
    cheaper = results_df[results_df["offgrid_cheaper"]]
    print(f"\n  Getestete Dörfer:          {len(results_df)}")
    print(
        f"  Offgrid günstiger:         {len(cheaper)} ({len(cheaper)/len(results_df)*100:.0f}%)"
    )
    print(
        f"  Ø LCOE:                    {results_df['lcoe_eur_kwh'].mean():.3f} EUR/kWh"
    )
    print(f"  Ø Solar pro Dorf:          {results_df['solar_kw'].mean():.1f} kW")
    print(
        f"  Ø CAPEX pro Dorf:          {results_df['capex_total_eur'].mean()/1000:.1f} kEUR"
    )
    print(f"  Gesamtbevoelkerung:        {results_df['population'].sum():,}")
    print(f"  Gesamt Solar:              {results_df['solar_kw'].sum():.1f} kW")
    print(
        f"  Gesamt CAPEX:              {results_df['capex_total_eur'].sum()/1000:.1f} kEUR"
    )
    print(f"  Gesamt CO2:                {results_df['co2_t_yr'].sum():.2f} t/Jahr")

    print(f"\n── Top 10 nach LCOE ──")
    print(
        results_df.nsmallest(10, "lcoe_eur_kwh")[
            [
                "village_id",
                "population",
                "dist_km",
                "solar_kw",
                "lcoe_eur_kwh",
                "capex_total_eur",
                "offgrid_cheaper",
            ]
        ].to_string(index=False)
    )

    results_df.to_csv("village_minigrid_results.csv", index=False)
    print(f"\n  Ergebnisse gespeichert: village_minigrid_results.csv")

    # ── Visualisierung ──────────────────────────────
    print("\n[5/5] Visualisierung erstellen...")
    fig, ax = plt.subplots(
        1, 1, figsize=(13, 13), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#d0e8f0")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_extent([33.5, 42.5, -5.0, 5.5])

    offgrid_ids = set(offgrid_df["gadm_id"].values)
    shapes_all = gpd.read_file(CONFIG["shapes_file"])
    for _, row in shapes_all.iterrows():
        color = "#CCFBF1" if row["GADM_ID"] in offgrid_ids else "#e8e8e8"
        ax.add_geometries(
            [row.geometry],
            ccrs.PlateCarree(),
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
        )

    ax.scatter(
        buses.x,
        buses.y,
        s=40,
        color="#0D2137",
        zorder=5,
        transform=ccrs.PlateCarree(),
        label="Netzknoten",
    )

    for _, vrow in results_df.iterrows():
        color = "#16a34a" if vrow["offgrid_cheaper"] else "#dc2626"
        size = max(20, min(200, vrow["population"] / 50))
        ax.plot(
            vrow["centroid_x"],
            vrow["centroid_y"],
            "o",
            color=color,
            markersize=size**0.5 * 2,
            transform=ccrs.PlateCarree(),
            zorder=8,
            alpha=0.8,
        )

    legend_handles = [
        mpatches.Patch(
            facecolor="#CCFBF1", label=f"Offgrid-Counties ({len(offgrid_df)})"
        ),
        mpatches.Patch(facecolor="#e8e8e8", label="On-grid Counties"),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="#16a34a",
            markersize=8,
            linestyle="None",
            label="Offgrid günstiger",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="#dc2626",
            markersize=8,
            linestyle="None",
            label="Netz günstiger",
        ),
        mlines.Line2D(
            [0],
            [0],
            marker="o",
            color="#0D2137",
            markersize=6,
            linestyle="None",
            label=f"Netzknoten ({len(buses)})",
        ),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.set_title(
        f"Village Mini-Grids – {len(results_df)} Dörfer in {len(offgrid_df)} Counties\n"
        f"Grün = Offgrid günstiger | Rot = Netzanschluss günstiger",
        fontsize=11,
        fontweight="bold",
    )
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.4, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    plt.tight_layout()
    plt.savefig("village_minigrids_map.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Karte gespeichert: village_minigrids_map.png")

print("\n" + "=" * 70)
print("FERTIG!")
print("=" * 70)
print("  - village_minigrid_results.csv")
print("  - village_minigrids_map.png")
