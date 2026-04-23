"""
build_offgrid.py
=================
Bachelorarbeit: Offgrid-Integration in PyPSA-Earth
OTH Regensburg | Betreuer: Anton Achhammer, Prof. Michael Sterner

PyPSA-Earth Skript: Identifiziert Offgrid-Regionen und fuegt isolierte
Mini-Grid Busse + Komponenten ins Netzwerk ein.

Generisch – funktioniert fuer jedes Land in config["countries"].
Elektrifizierungsdaten: {COUNTRY}_electricity_access.csv pro Land.

Ablauf:
    1. Laedt geloestes PyPSA-Earth Netzwerk
    2. Liest Offgrid-Parameter aus config.yaml (snakemake.config["offgrid"])
    3. Geo-Logik (C1-C5): Offgrid-Counties identifizieren
    4. Fuegt isolierte Mini-Grid Busse ins Netzwerk ein
    5. Optimiert erweitertes Netzwerk
    6. Speichert Ergebnisse

Einheiten in PyPSA (Quelle: PyPSA Dokumentation):
    - Leistung:      MW
    - Energie:       MWh
    - capital_cost:  EUR/MW/a  (annualisiert)
    - marginal_cost: EUR/MWh

Quellen:
    - Osiolo et al. (2019): 60 kWh/yr rural Kenya
    - PyPSA technology-data (TU Berlin/DEA): Solar/Batterie CAPEX
    - ERA5: Hersbach et al. (2020), doi:10.1002/qj.3803
    - ESMAP Mini Grid Design Manual (2019): Annuität r=8%, n=20yr
    - IEA Africa Energy Outlook (2022): Netzanschlusskosten
"""

import logging
import os
import warnings

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from shapely.geometry import Point

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Elektrifizierungsdaten
# ══════════════════════════════════════════════════════

def load_electrification_data_from_path(path):
    """
    Laedt Elektrifizierungsdaten aus einer CSV-Datei.

    Generisch – Dateiname bestimmt Land (z.B. KE_electricity_access.csv).
    Format:
        GADM_ID,                Access to electricity
        National Average,       23.0
        KE.1_1,                 9.6

    Returns:
        elec_dict (dict): {GADM_ID: rate}
        national_avg (float): nationaler Durchschnitt
    """
    if not os.path.exists(path):
        logger.warning(f"Elektrifizierungsdaten nicht gefunden: {path}")
        return None, 0.23

    df      = pd.read_csv(path)
    nat_row = df[df["GADM_ID"] == "National Average"]["Access to electricity"].values
    nat_avg = float(nat_row[0]) / 100 if len(nat_row) > 0 else 0.23
    df_reg  = df[df["GADM_ID"] != "National Average"].dropna(subset=["GADM_ID"])
    elec_dict = dict(zip(
        df_reg["GADM_ID"].str.strip(),
        df_reg["Access to electricity"].astype(float) / 100
    ))
    country = os.path.basename(path).split("_")[0]
    logger.info(f"  {country}: {len(elec_dict)} Regionen | Ø {nat_avg*100:.1f}%")
    return elec_dict, nat_avg


def load_all_electrification_data(elec_data_files):
    """
    Laedt Elektrifizierungsdaten fuer alle Laender.

    Iteriert ueber alle CSV-Dateien in snakemake.input.elec_data.
    Kombiniert alle Laender in einem Dictionary.

    Args:
        elec_data_files: str oder Liste von Pfaden

    Returns:
        elec_data_combined (dict): {GADM_ID: rate} fuer alle Laender
        national_avg (float): letzter nationaler Durchschnitt (Fallback)
    """
    # Snakemake gibt entweder str (1 Datei) oder Liste zurueck
    if isinstance(elec_data_files, str):
        elec_data_files = [elec_data_files]

    elec_data_combined = {}
    national_avg       = 0.23  # Fallback

    for path in elec_data_files:
        ed, nav = load_electrification_data_from_path(path)
        if ed:
            elec_data_combined.update(ed)
            national_avg = nav  # letzter Wert als Fallback

    if not elec_data_combined:
        logger.warning("Keine Elektrifizierungsdaten gefunden → Heuristik")
        return None, national_avg

    logger.info(f"Gesamt: {len(elec_data_combined)} Regionen aus {len(elec_data_files)} Laendern")
    return elec_data_combined, national_avg


def get_electrification_rate(gadm_id, distance_km, pop_density,
                              elec_data=None, national_avg=0.23):
    """
    Gibt Elektrifizierungsrate zurueck – echte Daten oder Heuristik.

    Prioritaet 1: Echte CSV-Daten
    Prioritaet 2: Heuristik nach Netzabstand + Bevoelkerungsdichte
    """
    if elec_data is not None:
        if gadm_id in elec_data:
            return elec_data[gadm_id], "Census"
        return national_avg, "National Avg"

    # Heuristik – Quelle: Weltbank (2022), eigene Ableitung
    if distance_km > 100:   base = 0.15
    elif distance_km > 50:  base = 0.30
    elif distance_km > 25:  base = 0.55
    else:                   base = 0.76

    if pop_density > 500:   f = 1.2
    elif pop_density > 100: f = 1.0
    elif pop_density > 25:  f = 0.8
    else:                   f = 0.6

    return round(min(base * f, 1.0), 4), "Heuristik"


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Solar Profile
# ══════════════════════════════════════════════════════

def get_solar_profile_from_era5(centroid_x, centroid_y, cutout, n_hours):
    """
    Echtes Solarprofil aus ERA5-Wetterdaten.
    Quelle: Hersbach et al. (2020), doi:10.1002/qj.3803
    """
    cutout_point = cutout.sel(
        x=slice(centroid_x - 0.3, centroid_x + 0.3),
        y=slice(centroid_y - 0.3, centroid_y + 0.3)
    )
    influx    = (cutout_point.data["influx_direct"] +
                 cutout_point.data["influx_diffuse"])
    influx_ts = influx.mean(dim=["x", "y"]).values[:n_hours]
    max_val   = influx_ts.max()
    if max_val > 0:
        return influx_ts / max_val
    return solar_profile_fallback(n_hours)


def load_profile(n_hours):
    """Tagesprofil fuer laendliche Haushalte."""
    hours = np.arange(n_hours)
    hod   = hours % 24
    p     = np.ones(n_hours) * 0.6
    p[hod >= 6]  = 0.8
    p[hod >= 9]  = 0.6
    p[hod >= 17] = 1.0
    p[hod >= 21] = 0.7
    p[hod >= 23] = 0.4
    return p / p.mean()


def solar_profile_fallback(n_hours):
    """Fallback Solarprofil falls ERA5 nicht verfuegbar."""
    hours = np.arange(n_hours)
    hod   = hours % 24
    s     = np.zeros(n_hours)
    s[hod >= 6]  = 0.3
    s[hod >= 9]  = 0.7
    s[hod >= 11] = 0.9
    s[hod >= 13] = 0.8
    s[hod >= 16] = 0.4
    s[hod >= 18] = 0.0
    return s


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Geo-Logik
# ══════════════════════════════════════════════════════

def identify_offgrid_counties(shapes, bus_geom, bus_load_mw,
                               bus_max_line_loading, elec_data,
                               national_avg, cfg):
    """
    Layer 1 – Geo-Logik: Identifiziert Offgrid-Counties.

    Prueft Kriterien C1-C5 fuer jede Region in shapes.
    Alle Kriterien konfigurierbar via config.yaml unter 'offgrid:'.

    C1: Netzabstand > max_distance_km
    C2: Elektrifizierungsrate < max_elec_rate
    C3: Bevoelkerungsdichte > min_pop_density
    C4: Baseline-Last < c4_max_load_mw  (optional)
    C5: Leitungsauslastung > c5_line_loading  (Szenario C, optional)

    Returns:
        DataFrame mit Offgrid-Counties
    """
    offgrid_rows = []

    for _, region in shapes.iterrows():
        gadm_id     = region["GADM_ID"]
        centroid    = region.geometry.centroid
        dist_km     = (bus_geom.geometry.distance(centroid) * 111).min()
        area_km2    = region.geometry.area * (111 ** 2)
        pop_density = region["pop"] / area_km2 if area_km2 > 0 else 0

        elec_rate, elec_src = get_electrification_rate(
            gadm_id, dist_km, pop_density, elec_data, national_avg
        )

        # C1: Netzabstand
        c1 = (dist_km > cfg["max_distance_km"]) if cfg["use_c1_distance"] else True
        # C2: Elektrifizierungsrate
        c2 = (elec_rate < cfg["max_elec_rate"]) if cfg["use_c2_elec_rate"] else True
        # C3: Bevoelkerungsdichte
        c3 = (pop_density > cfg["min_pop_density"]) if cfg["use_c3_pop_density"] else True
        # C4: Baseline-Last zu klein → Region zu klein fuer Hauptnetz
        nearest_bus  = bus_geom.geometry.distance(centroid).idxmin()
        nearest_load = bus_load_mw.get(nearest_bus, 0.0)
        c4 = (nearest_load < cfg["c4_max_load_mw"]) if cfg["use_c4_low_load"] else True
        # C5: Leitungsauslastung (Szenario C – Offgrid entlastet Netz)
        nearest_loading = bus_max_line_loading.get(nearest_bus, 0.0)
        c5 = (nearest_loading > cfg["c5_line_loading"]) if cfg["use_c5_congestion"] else True

        if c1 and c2 and c3 and c4 and c5:
            offgrid_rows.append({
                "gadm_id":    gadm_id,
                "population": int(region["pop"]),
                "elec_rate":  elec_rate,
                "elec_source":elec_src,
                "distance_km":round(dist_km, 1),
                "pop_density":round(pop_density, 1),
                "centroid_x": centroid.x,
                "centroid_y": centroid.y,
                "geometry":   region.geometry,
            })

    return pd.DataFrame(offgrid_rows)


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Mini-Grid hinzufügen
# ══════════════════════════════════════════════════════

def add_offgrid_bus(n, region_id, population, centroid_x, centroid_y,
                    cfg, cutout, annuity_factor, snapshots):
    """
    Fuegt einen isolierten Offgrid-Bus + Komponenten zum Netzwerk hinzu.

    PyPSA Einheiten (Quelle: docs.pypsa.org):
        - p_set / p_nom:  MW
        - capital_cost:   EUR/MW/a
        - marginal_cost:  EUR/MWh

    Kein Link zum Hauptnetz → vollstaendig isoliert.
    Alle capital_cost Werte: EUR/kW × 1000 × annuity_factor = EUR/MW/a
    """
    bus_name = f"offgrid_{region_id}"

    if bus_name in n.buses.index:
        logger.warning(f"Bus {bus_name} bereits vorhanden – ueberspringe")
        return None

    n_hours = len(snapshots)

    # ✅ Isolierten Bus hinzufuegen (kein Link zum Hauptnetz)
    n.add("Bus", bus_name,
          carrier="offgrid-AC",
          x=centroid_x,
          y=centroid_y,
          v_nom=0.4)  # 400V Niederspannung – typisch fuer Minigrids

    # ✅ Last in MW: kWh/yr / 8760h / 1000 = MW
    avg_load_mw = population * cfg["kwh_per_person_yr"] / 8760 / 1000
    lp  = load_profile(n_hours) * avg_load_mw
    lts = pd.Series(lp, index=snapshots)
    n.add("Load", f"load_{region_id}",
          bus=bus_name,
          p_set=lts)

    # Solar Profil – ERA5 oder Fallback
    if cutout is not None:
        try:
            sp        = get_solar_profile_from_era5(centroid_x, centroid_y, cutout, n_hours)
            solar_src = "ERA5"
        except Exception as e:
            logger.warning(f"ERA5 Fehler fuer {region_id}: {e} → Fallback")
            sp        = solar_profile_fallback(n_hours)
            solar_src = "Fallback"
    else:
        sp        = solar_profile_fallback(n_hours)
        solar_src = "Fallback"

    sts = pd.Series(sp, index=snapshots)

    # ✅ Solar: capital_cost in EUR/MW/a
    n.add("Generator", f"solar_{region_id}",
          bus=bus_name,
          carrier="solar",
          p_nom_extendable=True,
          p_nom_max=float("inf"),
          p_max_pu=sts,
          capital_cost=cfg["solar_capex"] * 1000 * annuity_factor,
          marginal_cost=0.01)

    # ✅ Batterie: capital_cost in EUR/MW/a
    n.add("StorageUnit", f"battery_{region_id}",
          bus=bus_name,
          carrier="battery",
          p_nom_extendable=True,
          max_hours=cfg["battery_max_hours"],
          capital_cost=cfg["battery_capex_kwh"] * 1000 * cfg["battery_max_hours"] * annuity_factor,
          marginal_cost=1.0,
          efficiency_store=0.95,
          efficiency_dispatch=0.95,
          cyclic_state_of_charge=True)

    # ✅ Diesel: capital_cost in EUR/MW/a
    n.add("Generator", f"diesel_{region_id}",
          bus=bus_name,
          carrier="diesel",
          p_nom_extendable=True,
          capital_cost=cfg["diesel_capex"] * 1000 * annuity_factor,
          marginal_cost=cfg["diesel_marginal"])

    # Load Shedding – Sicherheit
    n.add("Generator", f"shedding_{region_id}",
          bus=bus_name,
          carrier="load_shedding",
          p_nom=1e6,
          p_nom_extendable=False,
          marginal_cost=cfg["shedding_cost"])

    logger.info(f"  {region_id}: Bus + Solar + Batterie + Diesel [{solar_src}]")
    return solar_src


# ══════════════════════════════════════════════════════
# HILFSFUNKTIONEN – Ergebnisse auswerten
# ══════════════════════════════════════════════════════

def extract_results(n, offgrid_df, cfg, annuity_factor):
    """
    Liest Optimierungsergebnisse fuer alle Offgrid-Busse aus.

    LCOE:    total_cost_yr / total_load_mwh / 1000  [EUR/kWh]
    Autarkie: solar_gen / (solar_gen + diesel_gen)   [%]
    CO2:     diesel_gen_mwh * diesel_co2             [t/yr]
    """
    rows = []
    for _, row in offgrid_df.iterrows():
        rid = row["gadm_id"]
        try:
            solar_cap_mw   = n.generators.loc[f"solar_{rid}", "p_nom_opt"]
            battery_cap_mw = n.storage_units.loc[f"battery_{rid}", "p_nom_opt"]
            diesel_cap_mw  = n.generators.loc[f"diesel_{rid}", "p_nom_opt"]
            solar_gen_mwh  = n.generators_t.p[f"solar_{rid}"].sum()
            diesel_gen_mwh = n.generators_t.p[f"diesel_{rid}"].sum()

            # Kapazitaeten in kW fuer CSV
            solar_cap_kw   = solar_cap_mw   * 1000
            battery_cap_kw = battery_cap_mw * 1000
            diesel_cap_kw  = diesel_cap_mw  * 1000

            # CAPEX Berechnung (EUR, kW-Basis)
            capex_solar   = solar_cap_kw   * cfg["solar_capex"]
            capex_battery = battery_cap_kw * cfg["battery_max_hours"] * cfg["battery_capex_kwh"]
            capex_diesel  = diesel_cap_kw  * cfg["diesel_capex"]
            capex_total   = capex_solar + capex_battery + capex_diesel
            opex_yr       = (capex_solar + capex_battery) * 0.01
            capex_ann     = capex_total * annuity_factor
            total_cost_yr = capex_ann + opex_yr

            # Gesamtlast [MWh/yr]
            total_load_mwh = row["population"] * cfg["kwh_per_person_yr"] / 1000

            # ✅ LCOE [EUR/kWh]
            lcoe = total_cost_yr / total_load_mwh / 1000

            # ✅ Autarkie [%] – Solar-Anteil an Gesamterzeugung (max 100%)
            total_gen_mwh = solar_gen_mwh + diesel_gen_mwh
            autarky = (solar_gen_mwh / total_gen_mwh * 100) if total_gen_mwh > 0 else 0.0

            # ✅ CO2 [t/yr]
            co2_t = diesel_gen_mwh * cfg["diesel_co2"]

            # Netzanschluss Kosten (zum Vergleich)
            grid_capex    = row["distance_km"] * 15000 + 35000
            grid_total_yr = grid_capex * annuity_factor + grid_capex * 0.03
            offgrid_cheaper = total_cost_yr < grid_total_yr

            rows.append({
                "region":             rid,
                "population":         row["population"],
                "distance_km":        row["distance_km"],
                "elec_rate":          row["elec_rate"],
                "elec_source":        row["elec_source"],
                "solar_kw":           round(solar_cap_kw, 1),
                "battery_kw":         round(battery_cap_kw, 1),
                "battery_kwh":        round(battery_cap_kw * cfg["battery_max_hours"], 1),
                "diesel_kw":          round(diesel_cap_kw, 1),
                "capex_total_keur":   round(capex_total / 1000, 1),
                "total_cost_keur_yr": round(total_cost_yr / 1000, 1),
                "grid_capex_keur":    round(grid_capex / 1000, 1),
                "grid_total_keur_yr": round(grid_total_yr / 1000, 1),
                "offgrid_cheaper":    offgrid_cheaper,
                "lcoe_eur_kwh":       round(lcoe, 3),
                "autarky_pct":        round(autarky, 1),
                "co2_t_yr":           round(co2_t, 2),
                "centroid_x":         row["centroid_x"],
                "centroid_y":         row["centroid_y"],
            })
        except Exception as e:
            logger.warning(f"Auswertung fehlgeschlagen fuer {rid}: {e}")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
# HAUPTPROGRAMM
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s"
    )

    # ── Snakemake oder direkt aufrufen ────────────────────────────────
    if "snakemake" not in dir():
        import glob

        class MockSnakemake:
            class input:
                network = sorted(
                    [f for f in glob.glob("results/networks/*.nc")
                     if "offgrid" not in f]
                )[-1]
                shapes    = "resources/shapes/gadm_shapes.geojson"
                cutout    = "cutouts/cutout-2013-era5.nc"
                # Generisch: Liste von CSV-Dateien fuer alle Laender
                elec_data = glob.glob("data/elec_rates/*_electricity_access.csv")
            class output:
                network = "results/networks/elec_with_offgrid.nc"
                csv     = "results/offgrid_results.csv"
            class log:
                python  = "logs/build_offgrid.log"
            config = {
                "offgrid": {
                    "enable":            True,
                    "use_c1_distance":   True,
                    "use_c2_elec_rate":  True,
                    "use_c3_pop_density":True,
                    "use_c4_low_load":   False,
                    "use_c5_congestion": False,
                    "max_distance_km":   50,
                    "max_elec_rate":     0.50,
                    "min_pop_density":   10,
                    "c4_max_load_mw":    20.0,
                    "c5_line_loading":   0.50,
                    "kwh_per_person_yr": 60,
                    "solar_capex":       800,
                    "battery_capex_kwh": 250,
                    "battery_max_hours": 6,
                    "diesel_capex":      400,
                    "diesel_marginal":   300,
                    "diesel_co2":        0.27,
                    "shedding_cost":     5000,
                    "solver":            "gurobi",
                    "discount_rate":     0.08,
                    "asset_lifetime":    20,
                }
            }
        snakemake = MockSnakemake()
        logger.info("Direkter Aufruf – MockSnakemake aktiv")

    # ── Config lesen ──────────────────────────────────────────────────
    cfg = snakemake.config["offgrid"]

    # Offgrid Feature an/aus
    if not cfg.get("enable", True):
        logger.info("Offgrid deaktiviert (enable: false) → unveraendert kopieren")
        n = pypsa.Network(snakemake.input.network)
        n.export_to_netcdf(snakemake.output.network)
        pd.DataFrame().to_csv(snakemake.output.csv, index=False)
        exit(0)

    logger.info("=" * 60)
    logger.info("BUILD OFFGRID – Offgrid-Integration in PyPSA-Earth")
    logger.info("=" * 60)

    # ── Annuitätsfaktor ───────────────────────────────────────────────
    # Quelle: ESMAP Mini Grid Design Manual (2019)
    _r = cfg["discount_rate"]
    _n = cfg["asset_lifetime"]
    annuity_factor = _r * (1 + _r)**_n / ((1 + _r)**_n - 1)
    logger.info(f"Annuitätsfaktor: {annuity_factor:.4f} "
                f"(r={_r*100:.0f}%, n={_n}yr)")

    # ── Schritt 1: Netzwerk laden ─────────────────────────────────────
    logger.info(f"Lade Netzwerk: {snakemake.input.network}")
    n         = pypsa.Network(snakemake.input.network)
    snapshots = n.snapshots
    n_hours   = len(snapshots)
    logger.info(f"  {len(n.buses)} Busse | {n_hours} Zeitschritte")

    buses    = n.buses[["x", "y"]].copy()
    bus_geom = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses.x, buses.y),
        crs="EPSG:4326"
    )

    # C4: Last pro Bus
    bus_load_mw = {}
    if len(n.loads_t.p_set.columns) > 0:
        avg_load = n.loads_t.p_set.mean()
        for load_name in avg_load.index:
            bus_name = n.loads.loc[load_name, "bus"]
            bus_load_mw[bus_name] = bus_load_mw.get(bus_name, 0) + avg_load[load_name]

    # C5: Leitungsauslastung
    bus_max_line_loading = {}
    if len(n.lines_t.p0.columns) > 0 and len(n.lines) > 0:
        s_nom = (n.lines.s_nom_opt if "s_nom_opt" in n.lines.columns
                 else n.lines.s_nom)
        line_loading = (n.lines_t.p0.abs() / s_nom).max()
        for line_name, loading in line_loading.items():
            for bus in [n.lines.loc[line_name, "bus0"],
                        n.lines.loc[line_name, "bus1"]]:
                bus_max_line_loading[bus] = max(
                    bus_max_line_loading.get(bus, 0.0), loading
                )

    # ── Schritt 2: Shapes laden ───────────────────────────────────────
    logger.info(f"Lade Shapes: {snakemake.input.shapes}")
    shapes = gpd.read_file(snakemake.input.shapes)
    logger.info(f"  {len(shapes)} Regionen")

    # ── Schritt 3: Elektrifizierungsdaten laden ───────────────────────
    # Generisch: alle CSV-Dateien in snakemake.input.elec_data laden
    logger.info("Lade Elektrifizierungsdaten...")
    elec_data, national_avg = load_all_electrification_data(
        snakemake.input.elec_data
    )

    # ── Schritt 4: ERA5 Cutout laden ─────────────────────────────────
    cutout = None
    if os.path.exists(snakemake.input.cutout):
        try:
            cutout = atlite.Cutout(snakemake.input.cutout)
            logger.info(f"ERA5 geladen ✅")
        except Exception as e:
            logger.warning(f"ERA5 Fehler: {e} → Fallback")
    else:
        logger.warning("ERA5 nicht gefunden → Fallback")

    # ── Schritt 5: Geo-Logik ──────────────────────────────────────────
    logger.info("Geo-Logik: Offgrid-Counties identifizieren...")
    logger.info(f"  C1 (Abstand > {cfg['max_distance_km']} km):    "
                f"{'aktiv' if cfg['use_c1_distance'] else 'INAKTIV'}")
    logger.info(f"  C2 (Elec < {cfg['max_elec_rate']*100:.0f}%):         "
                f"{'aktiv' if cfg['use_c2_elec_rate'] else 'INAKTIV'}")
    logger.info(f"  C3 (Dichte > {cfg['min_pop_density']} P/km2):  "
                f"{'aktiv' if cfg['use_c3_pop_density'] else 'INAKTIV'}")
    logger.info(f"  C4 (Last < {cfg['c4_max_load_mw']} MW):        "
                f"{'aktiv' if cfg['use_c4_low_load'] else 'INAKTIV'}")
    logger.info(f"  C5 (Auslastung > {cfg['c5_line_loading']*100:.0f}%): "
                f"{'aktiv (Szenario C)' if cfg['use_c5_congestion'] else 'INAKTIV (Szenario A)'}")

    offgrid_df = identify_offgrid_counties(
        shapes, bus_geom, bus_load_mw, bus_max_line_loading,
        elec_data, national_avg, cfg
    )
    logger.info(f"  → {len(offgrid_df)} Offgrid-Counties identifiziert")

    if len(offgrid_df) == 0:
        logger.warning("Keine Offgrid-Counties! Netzwerk unveraendert speichern.")
        n.export_to_netcdf(snakemake.output.network)
        pd.DataFrame().to_csv(snakemake.output.csv, index=False)
        exit(0)

    # ── Schritt 6: Carrier hinzufügen ────────────────────────────────
    for carrier in ["offgrid-AC", "diesel", "load_shedding"]:
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)

    # ── Schritt 7: Mini-Grid Busse hinzufügen ─────────────────────────
    logger.info(f"Fuege {len(offgrid_df)} Mini-Grid Busse ein...")
    n_added = 0

    for _, row in offgrid_df.iterrows():
        rid    = row["gadm_id"]
        result = add_offgrid_bus(
            n              = n,
            region_id      = rid,
            population     = row["population"],
            centroid_x     = row["centroid_x"],
            centroid_y     = row["centroid_y"],
            cfg            = cfg,
            cutout         = cutout,
            annuity_factor = annuity_factor,
            snapshots      = snapshots,
        )
        if result is not None:
            n_added += 1

    logger.info(f"  → {n_added} Busse hinzugefuegt")
    logger.info(f"  → Netzwerk: {len(n.buses)} Busse total "
                f"({len(buses)} Hauptnetz + {n_added} Offgrid)")

    # ── Schritt 8: Netzwerk speichern ────────────────────────────────
    os.makedirs(os.path.dirname(snakemake.output.network), exist_ok=True)
    n.export_to_netcdf(snakemake.output.network)
    logger.info(f"Netzwerk gespeichert: {snakemake.output.network}")

    # ── Schritt 9: Optimieren ────────────────────────────────────────
    logger.info(f"Optimierung: {cfg['solver']} | {len(n.buses)} Busse")
    n.optimize(solver_name=cfg["solver"])

    # ── Schritt 10: Ergebnisse ───────────────────────────────────────
    results_df = extract_results(n, offgrid_df, cfg, annuity_factor)

    if len(results_df) > 0:
        results_df.to_csv(snakemake.output.csv, index=False)

        logger.info("=" * 60)
        logger.info("ZUSAMMENFASSUNG")
        logger.info("=" * 60)
        logger.info(f"  Mini-Grids:        {len(results_df)}")
        logger.info(f"  Gesamtbevölkerung: {results_df['population'].sum():,}")
        logger.info(f"  Gesamt Solar:      {results_df['solar_kw'].sum():.0f} kW")
        logger.info(f"  Ø LCOE:            {results_df['lcoe_eur_kwh'].mean():.3f} EUR/kWh")
        logger.info(f"  Ø Autarkie:        {results_df['autarky_pct'].mean():.1f}%")
        logger.info(f"  Gesamt CO2:        {results_df['co2_t_yr'].sum():.1f} t/Jahr")
        logger.info(f"  Ergebnisse:        {snakemake.output.csv}")

    # Finales Netzwerk nochmal speichern (mit Ergebnissen)
    n.export_to_netcdf(snakemake.output.network)
    logger.info(f"Finales Netzwerk: {snakemake.output.network}")
    logger.info("FERTIG ✅")