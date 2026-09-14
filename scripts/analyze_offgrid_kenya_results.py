# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Analyse Kenya offgrid smoke-run results against matching references."""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


logger = logging.getLogger(__name__)

OFFGRID_CARRIER = "offgrid AC"
OFFGRID_SOLAR_CARRIER = "offgrid solar"
OFFGRID_OIL_CARRIER = "offgrid oil"
OFFGRID_BATTERY_CARRIER = "offgrid battery"
OFFGRID_DEMAND_CARRIER = "offgrid demand"
OFFGRID_DEMAND_RELIEF_CARRIER = "offgrid demand relief"
LOAD_SHEDDING_CARRIER = "load shedding"
ACCESS_RATE_FILE = Path("data/custom/kenya_offgrid_access_rates_test.csv")
BASE_DEMAND_YEAR = 2030


@dataclass(frozen=True)
class RunPair:
    case_id: str
    label: str
    foresight: str
    model: str
    year: int
    offgrid_path: Path
    reference_path: Path


RUN_PAIRS = [
    RunPair(
        "overnight-electricity-2030",
        "Overnight Elec 2030",
        "overnight",
        "electricity",
        2030,
        Path("results/ke-offgrid-overnight-2030/networks/elec_s_6_ec_lcopt_Co2L-6h.nc"),
        Path("results/ke-reference-overnight-2030/networks/elec_s_6_ec_lcopt_Co2L-6h.nc"),
    ),
    RunPair(
        "overnight-electricity-2040",
        "Overnight Elec 2040",
        "overnight",
        "electricity",
        2040,
        Path("results/ke-offgrid-overnight-2040/networks/elec_s_6_ec_lcopt_Co2L-6h.nc"),
        Path("results/ke-reference-overnight-2040/networks/elec_s_6_ec_lcopt_Co2L-6h.nc"),
    ),
    RunPair(
        "overnight-sector-2030",
        "Overnight Sector 2030",
        "overnight",
        "sector",
        2030,
        Path(
            "results/ke-offgrid-overnight-2030/postnetworks/"
            "elec_s_6_ec_lcopt_Co2L-6h_144h_2030_0.071.nc"
        ),
        Path(
            "results/ke-reference-overnight-2030/postnetworks/"
            "elec_s_6_ec_lcopt_Co2L-6h_144h_2030_0.071.nc"
        ),
    ),
    RunPair(
        "overnight-sector-2040",
        "Overnight Sector 2040",
        "overnight",
        "sector",
        2040,
        Path(
            "results/ke-offgrid-overnight-2040/postnetworks/"
            "elec_s_6_ec_lcopt_Co2L-6h_144h_2040_0.071.nc"
        ),
        Path(
            "results/ke-reference-overnight-2040/postnetworks/"
            "elec_s_6_ec_lcopt_Co2L-6h_144h_2040_0.071.nc"
        ),
    ),
    RunPair(
        "myopic-sector-2030",
        "Myopic Sector 2030",
        "myopic",
        "sector",
        2030,
        Path(
            "results/ke-offgrid-myopic/postnetworks/"
            "elec_s_6_ec_lc1_Co2L-6h_144h_2030_0.071.nc"
        ),
        Path(
            "results/ke-reference-myopic/postnetworks/"
            "elec_s_6_ec_lc1_Co2L-6h_144h_2030_0.071.nc"
        ),
    ),
    RunPair(
        "myopic-sector-2040",
        "Myopic Sector 2040",
        "myopic",
        "sector",
        2040,
        Path(
            "results/ke-offgrid-myopic/postnetworks/"
            "elec_s_6_ec_lc1_Co2L-6h_144h_2040_0.071.nc"
        ),
        Path(
            "results/ke-reference-myopic/postnetworks/"
            "elec_s_6_ec_lc1_Co2L-6h_144h_2040_0.071.nc"
        ),
    ),
]


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "legend.frameon": False,
        }
    )


def snapshot_weights(n: pypsa.Network) -> pd.Series:
    weightings = n.snapshot_weightings
    for column in ("generators", "objective", "stores"):
        if column in weightings.columns:
            return pd.to_numeric(weightings[column], errors="coerce").fillna(1.0)
    return pd.Series(1.0, index=n.snapshots)


def weighted_twh(n: pypsa.Network, profiles: pd.DataFrame | pd.Series) -> float:
    if profiles is None or len(profiles) == 0:
        return 0.0
    if isinstance(profiles, pd.Series):
        profiles = profiles.to_frame()
    if profiles.empty:
        return 0.0
    return float(profiles.mul(snapshot_weights(n), axis=0).sum().sum() / 1e6)


def component_names(n: pypsa.Network, component: str, carrier: str) -> pd.Index:
    frame = getattr(n, component)
    if frame.empty or "carrier" not in frame.columns:
        return pd.Index([])
    return pd.Index(frame.index[frame.carrier.fillna("") == carrier])


def component_count(n: pypsa.Network, component: str, carrier: str) -> int:
    return len(component_names(n, component, carrier))


def nominal_capacity_mw(
    n: pypsa.Network, component: str, carrier: str, names: pd.Index | None = None
) -> float:
    frame = getattr(n, component)
    if names is None:
        names = component_names(n, component, carrier)
    if not len(names):
        return 0.0

    base_col = "p_nom" if "p_nom" in frame.columns else "e_nom"
    opt_col = f"{base_col}_opt"
    base = pd.to_numeric(frame.loc[names, base_col], errors="coerce").fillna(0.0)
    if opt_col in frame.columns:
        opt = pd.to_numeric(frame.loc[names, opt_col], errors="coerce")
        values = opt.where(opt.notna(), base)
        values = values.where((values != 0.0) | (base == 0.0), base)
    else:
        values = base
    return float(values.sum())


def link_delivery_twh(n: pypsa.Network, carrier: str, side: str = "p1") -> float:
    names = component_names(n, "links", carrier)
    if not len(names):
        return 0.0
    pnl = getattr(n.links_t, side, pd.DataFrame(index=n.snapshots))
    columns = [name for name in names if name in pnl.columns]
    if not columns:
        return 0.0
    return weighted_twh(n, (-pnl[columns]).clip(lower=0.0))


def generator_output_twh(
    n: pypsa.Network, carrier: str, names: pd.Index | None = None
) -> float:
    if names is None:
        names = component_names(n, "generators", carrier)
    columns = [name for name in names if name in n.generators_t.p.columns]
    if not columns:
        return 0.0
    return weighted_twh(n, n.generators_t.p[columns].clip(lower=0.0))


def storage_dispatch_twh(n: pypsa.Network, carrier: str) -> float:
    names = component_names(n, "storage_units", carrier)
    pnl = getattr(n.storage_units_t, "p_dispatch", pd.DataFrame(index=n.snapshots))
    columns = [name for name in names if name in pnl.columns]
    if not columns:
        return 0.0
    return weighted_twh(n, pnl[columns].clip(lower=0.0))


def load_profiles(n: pypsa.Network, loads: pd.Index | None = None) -> pd.DataFrame:
    loads = pd.Index(n.loads.index if loads is None else loads)
    profiles = pd.DataFrame(0.0, index=n.snapshots, columns=loads, dtype=float)
    dynamic = getattr(n.loads_t, "p_set", pd.DataFrame(index=n.snapshots))

    dynamic_cols = [load for load in loads if load in dynamic.columns]
    if dynamic_cols:
        profiles.loc[:, dynamic_cols] = dynamic[dynamic_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    static_cols = [load for load in loads if load not in dynamic.columns]
    for load in static_cols:
        if load in n.loads.index and "p_set" in n.loads.columns:
            profiles.loc[:, load] = pd.to_numeric(
                pd.Series(n.loads.at[load, "p_set"], index=n.snapshots),
                errors="coerce",
            ).fillna(0.0)

    return profiles.fillna(0.0)


def ac_buses(n: pypsa.Network) -> pd.Index:
    if "carrier" in n.buses.columns:
        buses = n.buses.index[n.buses.carrier.fillna("") == "AC"]
        if len(buses):
            return pd.Index(buses)
    return pd.Index(n.loads.bus.dropna().unique()).intersection(n.buses.index)


def offgrid_buses(n: pypsa.Network) -> pd.Index:
    return component_names(n, "buses", OFFGRID_CARRIER)


def offgrid_relevant_loads_by_parent(n: pypsa.Network) -> dict[str, list[str]]:
    parent_loads: dict[str, list[str]] = {}
    parents = set(ac_buses(n))
    load_carriers = n.loads.get("carrier", pd.Series("", index=n.loads.index)).fillna("")

    for load, bus in n.loads.bus.fillna("").items():
        bus = str(bus)
        parent = None

        if bus in parents:
            parent = bus
        elif bus.endswith(" low voltage"):
            candidate = bus[: -len(" low voltage")]
            if candidate in parents and load_carriers.at[load] in {"", "AC"}:
                parent = candidate

        if parent is not None:
            parent_loads.setdefault(parent, []).append(load)

    return parent_loads


def offgrid_load_shedding_generators(n: pypsa.Network) -> pd.Index:
    names = component_names(n, "generators", LOAD_SHEDDING_CARRIER)
    buses = set(offgrid_buses(n))
    return pd.Index([name for name in names if n.generators.at[name, "bus"] in buses])


def nodal_demand(n: pypsa.Network, run: RunPair, variant: str) -> pd.DataFrame:
    relief = component_names(n, "generators", OFFGRID_DEMAND_RELIEF_CARRIER)
    demand = component_names(n, "generators", OFFGRID_DEMAND_CARRIER)
    relief_p_nom = pd.to_numeric(
        n.generators.loc[relief, "p_nom_opt"]
        if len(relief) and "p_nom_opt" in n.generators.columns
        else pd.Series(dtype=float),
        errors="coerce",
    )
    demand_p_nom = pd.to_numeric(
        n.generators.loc[demand, "p_nom_opt"]
        if len(demand) and "p_nom_opt" in n.generators.columns
        else pd.Series(dtype=float),
        errors="coerce",
    )

    rows = []
    for parent, loads in sorted(offgrid_relevant_loads_by_parent(n).items()):
        demand_name = f"{parent} offgrid demand"
        relief_name = f"{parent} offgrid demand relief"
        gross = weighted_twh(n, load_profiles(n, pd.Index(loads)))
        offgrid = generator_output_twh(n, OFFGRID_DEMAND_CARRIER, pd.Index([demand_name]))
        relief_twh = generator_output_twh(
            n, OFFGRID_DEMAND_RELIEF_CARRIER, pd.Index([relief_name])
        )
        peak = (
            float(n.generators.at[demand_name, "p_nom_max"])
            if demand_name in n.generators.index
            else 0.0
        )
        rows.append(
            {
                "case_id": run.case_id,
                "case_label": run.label,
                "foresight": run.foresight,
                "model": run.model,
                "year": run.year,
                "variant": variant,
                "bus": parent,
                "gross_electricity_demand_twh": gross,
                "on_grid_electricity_demand_twh": max(gross - relief_twh, 0.0),
                "offgrid_demand_twh": offgrid,
                "offgrid_share": offgrid / gross if gross else 0.0,
                "offgrid_alpha": (
                    float(demand_p_nom.get(demand_name, 0.0)) / peak if peak else 0.0
                ),
                "relief_alpha": (
                    float(relief_p_nom.get(relief_name, 0.0)) / peak if peak else 0.0
                ),
                "load_count": len(loads),
            }
        )
    return pd.DataFrame(rows)


def line_metrics(n: pypsa.Network) -> dict[str, float]:
    if n.lines.empty:
        return {
            "line_count": 0,
            "line_s_nom_mw": 0.0,
            "line_s_nom_opt_mw": 0.0,
            "line_expansion_mw": 0.0,
            "line_volume_mwkm": 0.0,
            "line_expansion_mwkm": 0.0,
        }

    s_nom = pd.to_numeric(n.lines.get("s_nom", 0.0), errors="coerce").fillna(0.0)
    s_nom_opt = pd.to_numeric(n.lines.get("s_nom_opt", s_nom), errors="coerce").fillna(
        s_nom
    )
    length = pd.to_numeric(n.lines.get("length", 0.0), errors="coerce").fillna(0.0)
    expansion = (s_nom_opt - s_nom).clip(lower=0.0)

    return {
        "line_count": int(len(n.lines)),
        "line_s_nom_mw": float(s_nom.sum()),
        "line_s_nom_opt_mw": float(s_nom_opt.sum()),
        "line_expansion_mw": float(expansion.sum()),
        "line_volume_mwkm": float((s_nom_opt * length).sum()),
        "line_expansion_mwkm": float((expansion * length).sum()),
    }


def load_network(path: Path) -> pypsa.Network:
    pypsa_logger = logging.getLogger("pypsa")
    old_level = pypsa_logger.level
    pypsa_logger.setLevel(logging.WARNING)
    try:
        return pypsa.Network(path)
    finally:
        pypsa_logger.setLevel(old_level)


def run_metrics(
    n: pypsa.Network, run: RunPair, variant: str, path: Path
) -> dict[str, float | int | str | bool]:
    selected_loads = [
        load for loads in offgrid_relevant_loads_by_parent(n).values() for load in loads
    ]
    gross_demand = weighted_twh(n, load_profiles(n, pd.Index(selected_loads)))
    offgrid_demand = generator_output_twh(n, OFFGRID_DEMAND_CARRIER)
    offgrid_relief = generator_output_twh(n, OFFGRID_DEMAND_RELIEF_CARRIER)
    offgrid_shedding = generator_output_twh(
        n, LOAD_SHEDDING_CARRIER, offgrid_load_shedding_generators(n)
    )

    bus_carriers = n.buses.get("carrier", pd.Series("", index=n.buses.index)).fillna("")
    ac_bus_count = int((bus_carriers == "AC").sum())
    h2_bus_count = int((bus_carriers == "H2").sum())
    offgrid_bus_count = component_count(n, "buses", OFFGRID_CARRIER)
    classified_bus_count = ac_bus_count + h2_bus_count + offgrid_bus_count
    offgrid_oil_capacity = nominal_capacity_mw(n, "generators", OFFGRID_OIL_CARRIER)
    offgrid_oil_capacity += nominal_capacity_mw(n, "links", OFFGRID_OIL_CARRIER)

    metrics = {
        "case_id": run.case_id,
        "case_label": run.label,
        "foresight": run.foresight,
        "model": run.model,
        "year": run.year,
        "variant": variant,
        "path": str(path),
        "exists": path.exists(),
        "objective": float(getattr(n, "objective", 0.0) or 0.0),
        "objective_bn": float((getattr(n, "objective", 0.0) or 0.0) / 1e9),
        "bus_count": int(len(n.buses)),
        "ac_bus_count": ac_bus_count,
        "h2_bus_count": h2_bus_count,
        "offgrid_bus_count": offgrid_bus_count,
        "other_bus_count": int(len(n.buses) - classified_bus_count),
        "load_count": int(len(n.loads)),
        "generator_count": int(len(n.generators)),
        "link_count": int(len(n.links)),
        "storage_unit_count": int(len(n.storage_units)),
        "offgrid_demand_count": component_count(
            n, "generators", OFFGRID_DEMAND_CARRIER
        ),
        "offgrid_demand_relief_count": component_count(
            n, "generators", OFFGRID_DEMAND_RELIEF_CARRIER
        ),
        "offgrid_solar_count": component_count(n, "generators", OFFGRID_SOLAR_CARRIER),
        "offgrid_oil_generator_count": component_count(
            n, "generators", OFFGRID_OIL_CARRIER
        ),
        "offgrid_oil_link_count": component_count(n, "links", OFFGRID_OIL_CARRIER),
        "offgrid_battery_count": component_count(
            n, "storage_units", OFFGRID_BATTERY_CARRIER
        ),
        "total_load_twh": weighted_twh(n, load_profiles(n)),
        "gross_electricity_demand_twh": gross_demand,
        "on_grid_electricity_demand_twh": max(gross_demand - offgrid_relief, 0.0),
        "offgrid_demand_twh": offgrid_demand,
        "offgrid_demand_relief_twh": offgrid_relief,
        "offgrid_share": offgrid_demand / gross_demand if gross_demand else 0.0,
        "offgrid_solar_generation_twh": generator_output_twh(
            n, OFFGRID_SOLAR_CARRIER
        ),
        "offgrid_oil_generation_twh": generator_output_twh(n, OFFGRID_OIL_CARRIER),
        "offgrid_oil_link_delivery_twh": link_delivery_twh(n, OFFGRID_OIL_CARRIER),
        "offgrid_battery_dispatch_twh": storage_dispatch_twh(
            n, OFFGRID_BATTERY_CARRIER
        ),
        "offgrid_load_shedding_twh": offgrid_shedding,
        "offgrid_demand_p_nom_opt_mw": nominal_capacity_mw(
            n, "generators", OFFGRID_DEMAND_CARRIER
        ),
        "offgrid_solar_p_nom_opt_mw": nominal_capacity_mw(
            n, "generators", OFFGRID_SOLAR_CARRIER
        ),
        "offgrid_oil_p_nom_opt_mw": offgrid_oil_capacity,
        "offgrid_battery_p_nom_opt_mw": nominal_capacity_mw(
            n, "storage_units", OFFGRID_BATTERY_CARRIER
        ),
        "offgrid_load_shedding_p_nom_mw": nominal_capacity_mw(
            n,
            "generators",
            LOAD_SHEDDING_CARRIER,
            offgrid_load_shedding_generators(n),
        ),
    }
    metrics.update(line_metrics(n))
    return metrics


def dispatch_profiles(n: pypsa.Network, run: RunPair) -> pd.DataFrame:
    selected_loads = [
        load for group in offgrid_relevant_loads_by_parent(n).values() for load in group
    ]
    data = pd.DataFrame(index=n.snapshots)
    data["gross_electricity_demand_mw"] = load_profiles(
        n, pd.Index(selected_loads)
    ).sum(axis=1)

    for carrier, column in [
        (OFFGRID_DEMAND_RELIEF_CARRIER, "offgrid_demand_relief_mw"),
        (OFFGRID_DEMAND_CARRIER, "offgrid_demand_mw"),
        (OFFGRID_SOLAR_CARRIER, "offgrid_solar_mw"),
    ]:
        names = component_names(n, "generators", carrier)
        existing = [name for name in names if name in n.generators_t.p.columns]
        data[column] = (
            n.generators_t.p[existing].clip(lower=0.0).sum(axis=1) if existing else 0.0
        )

    oil_links = component_names(n, "links", OFFGRID_OIL_CARRIER)
    oil_link_cols = [name for name in oil_links if name in n.links_t.p1.columns]
    data["offgrid_oil_link_mw"] = (
        (-n.links_t.p1[oil_link_cols]).clip(lower=0.0).sum(axis=1)
        if oil_link_cols
        else 0.0
    )
    oil_generators = component_names(n, "generators", OFFGRID_OIL_CARRIER)
    oil_gen_cols = [name for name in oil_generators if name in n.generators_t.p.columns]
    data["offgrid_oil_generator_mw"] = (
        n.generators_t.p[oil_gen_cols].clip(lower=0.0).sum(axis=1)
        if oil_gen_cols
        else 0.0
    )
    battery = component_names(n, "storage_units", OFFGRID_BATTERY_CARRIER)
    battery_cols = [name for name in battery if name in n.storage_units_t.p_dispatch.columns]
    data["offgrid_battery_dispatch_mw"] = (
        n.storage_units_t.p_dispatch[battery_cols].clip(lower=0.0).sum(axis=1)
        if battery_cols
        else 0.0
    )
    shedding = offgrid_load_shedding_generators(n)
    shedding_cols = [name for name in shedding if name in n.generators_t.p.columns]
    data["offgrid_load_shedding_mw"] = (
        n.generators_t.p[shedding_cols].clip(lower=0.0).sum(axis=1)
        if shedding_cols
        else 0.0
    )
    data["on_grid_electricity_demand_mw"] = (
        data["gross_electricity_demand_mw"] - data["offgrid_demand_relief_mw"]
    ).clip(lower=0.0)
    data["snapshot"] = data.index.astype(str)
    data["case_id"] = run.case_id
    data["case_label"] = run.label
    return data.reset_index(drop=True)


def collect_results(repo_root: Path, allow_missing: bool = False):
    summary_rows = []
    nodal_rows = []
    dispatch_rows = []
    networks = {}
    missing = []

    for run in RUN_PAIRS:
        for variant, rel_path in [
            ("offgrid", run.offgrid_path),
            ("reference", run.reference_path),
        ]:
            path = repo_root / rel_path
            if not path.exists():
                missing.append(str(rel_path))
                continue
            network = load_network(path)
            networks[(run.case_id, variant)] = network
            summary_rows.append(run_metrics(network, run, variant, rel_path))
            nodal_rows.append(nodal_demand(network, run, variant))
            if variant == "offgrid":
                dispatch_rows.append(dispatch_profiles(network, run))

    if missing and not allow_missing:
        raise FileNotFoundError(
            "Missing expected result networks:\n- " + "\n- ".join(missing)
        )

    summary = pd.DataFrame(summary_rows)
    nodal = pd.concat(nodal_rows, ignore_index=True) if nodal_rows else pd.DataFrame()
    dispatch = (
        pd.concat(dispatch_rows, ignore_index=True) if dispatch_rows else pd.DataFrame()
    )
    comparisons = build_comparisons(summary)
    capacities = build_capacity_table(summary)
    lines = build_line_table(summary)
    split = build_demand_split_table(summary)
    generation = build_offgrid_generation_table(summary)
    components = demand_components(repo_root, networks)
    return (
        summary,
        comparisons,
        nodal,
        dispatch,
        capacities,
        lines,
        split,
        generation,
        components,
        networks,
    )


def build_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, group in summary.groupby("case_id", sort=False):
        if set(group.variant) != {"offgrid", "reference"}:
            continue
        offgrid = group[group.variant == "offgrid"].iloc[0]
        reference = group[group.variant == "reference"].iloc[0]
        rows.append(
            {
                "case_id": case_id,
                "case_label": offgrid.case_label,
                "foresight": offgrid.foresight,
                "model": offgrid.model,
                "year": int(offgrid.year),
                "objective_delta": offgrid.objective - reference.objective,
                "objective_delta_bn": offgrid.objective_bn - reference.objective_bn,
                "total_load_delta_twh": offgrid.total_load_twh
                - reference.total_load_twh,
                "gross_electricity_demand_delta_twh": offgrid.gross_electricity_demand_twh
                - reference.gross_electricity_demand_twh,
                "offgrid_share": offgrid.offgrid_share,
                "offgrid_demand_twh": offgrid.offgrid_demand_twh,
                "on_grid_electricity_demand_twh": offgrid.on_grid_electricity_demand_twh,
                "offgrid_solar_p_nom_opt_mw": offgrid.offgrid_solar_p_nom_opt_mw,
                "offgrid_oil_p_nom_opt_mw": offgrid.offgrid_oil_p_nom_opt_mw,
                "offgrid_battery_p_nom_opt_mw": offgrid.offgrid_battery_p_nom_opt_mw,
                "line_expansion_delta_mwkm": offgrid.line_expansion_mwkm
                - reference.line_expansion_mwkm,
            }
        )
    return pd.DataFrame(rows)


def build_capacity_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    capacity_columns = {
        "Offgrid demand peak": "offgrid_demand_p_nom_opt_mw",
        "Offgrid solar": "offgrid_solar_p_nom_opt_mw",
        "Offgrid battery": "offgrid_battery_p_nom_opt_mw",
        "Offgrid oil/diesel": "offgrid_oil_p_nom_opt_mw",
        "Offgrid load shedding": "offgrid_load_shedding_p_nom_mw",
    }
    offgrid = summary[summary.variant == "offgrid"]
    for _, row in offgrid.iterrows():
        for carrier, column in capacity_columns.items():
            rows.append(
                {
                    "case_id": row.case_id,
                    "case_label": row.case_label,
                    "carrier": carrier,
                    "capacity_mw": float(row[column]),
                }
            )
    return pd.DataFrame(rows)


def build_line_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "line_count",
        "line_s_nom_mw",
        "line_s_nom_opt_mw",
        "line_expansion_mw",
        "line_volume_mwkm",
        "line_expansion_mwkm",
    ]
    return summary[
        ["case_id", "case_label", "variant", "foresight", "model", "year"] + columns
    ].copy()


def build_demand_split_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    offgrid = summary[summary.variant == "offgrid"]
    for _, row in offgrid.iterrows():
        rows.extend(
            [
                {
                    "case_id": row.case_id,
                    "case_label": row.case_label,
                    "segment": "On-grid demand",
                    "demand_twh": row.on_grid_electricity_demand_twh,
                    "share": 1.0 - row.offgrid_share,
                },
                {
                    "case_id": row.case_id,
                    "case_label": row.case_label,
                    "segment": "Offgrid demand",
                    "demand_twh": row.offgrid_demand_twh,
                    "share": row.offgrid_share,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_offgrid_generation_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    offgrid = summary[summary.variant == "offgrid"]
    for _, row in offgrid.iterrows():
        for source, value in [
            ("Offgrid solar", row.offgrid_solar_generation_twh),
            ("Offgrid oil/diesel", row.offgrid_oil_generation_twh + row.offgrid_oil_link_delivery_twh),
            ("Offgrid battery", row.offgrid_battery_dispatch_twh),
            ("Offgrid load shedding", row.offgrid_load_shedding_twh),
        ]:
            rows.append(
                {
                    "case_id": row.case_id,
                    "case_label": row.case_label,
                    "source": source,
                    "delivery_twh": float(value),
                }
            )
    return pd.DataFrame(rows)


def write_tables(
    tables_dir: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    nodal: pd.DataFrame,
    dispatch: pd.DataFrame,
    capacities: pd.DataFrame,
    lines: pd.DataFrame,
    split: pd.DataFrame,
    generation: pd.DataFrame,
    components: pd.DataFrame,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "run_summary.csv", index=False)
    comparisons.to_csv(tables_dir / "comparison_summary.csv", index=False)
    nodal.to_csv(tables_dir / "nodal_demand.csv", index=False)
    dispatch.to_csv(tables_dir / "dispatch_examples.csv", index=False)
    capacities.to_csv(tables_dir / "capacity_summary.csv", index=False)
    lines.to_csv(tables_dir / "line_expansion.csv", index=False)
    split.to_csv(tables_dir / "demand_split.csv", index=False)
    generation.to_csv(tables_dir / "offgrid_generation.csv", index=False)
    components.to_csv(tables_dir / "demand_components.csv", index=False)


def savefig(fig: plt.Figure, figures_dir: Path, name: str) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def add_subtitle(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color="#555555")


def wrapped_case_labels(labels) -> list[str]:
    return [
        str(label)
        .replace("Overnight Elec", "Overnight\nElec")
        .replace("Overnight Sector", "Overnight\nSector")
        .replace("Myopic Sector", "Myopic\nSector")
        for label in labels
    ]


def short_case_label(label: str) -> str:
    return (
        str(label)
        .replace("Overnight Elec", "Ov Elec")
        .replace("Overnight Sector", "Ov Sector")
        .replace("Myopic Sector", "My Sector")
    )


def run_resource_dir(run: RunPair) -> Path:
    parts = run.offgrid_path.parts
    if parts and parts[0] == "results":
        return Path("resources") / parts[1]
    if parts and parts[0] == "networks":
        return Path("resources") / parts[1]
    return Path("resources")


def read_profile_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=0)
    data.index = pd.to_datetime(data.index)
    data.columns = data.columns.astype(str)
    return data.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def align_profile_to_network_snapshots(
    profile: pd.DataFrame, snapshots: pd.Index
) -> pd.DataFrame:
    snapshots = pd.Index(snapshots)
    if profile.index.equals(snapshots):
        return profile

    frequency = getattr(snapshots, "freqstr", None)
    if frequency is None and len(snapshots) >= 3:
        frequency = pd.infer_freq(snapshots)
    if frequency is None and len(snapshots) >= 2:
        frequency = snapshots[1] - snapshots[0]
    if frequency is not None and len(profile.index.difference(snapshots)) > 0:
        offset = frequency.casefold() if isinstance(frequency, str) else frequency
        resampled = profile.resample(offset).mean()
        if len(snapshots.difference(resampled.index)) == 0:
            return resampled.reindex(snapshots).fillna(0.0)

    return profile.reindex(snapshots).fillna(0.0)


def read_busmap(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    source_col = "Bus" if "Bus" in data.columns else data.columns[0]
    target_col = "busmap" if "busmap" in data.columns else data.columns[1]
    def normalise(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        text = str(value).strip()
        if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
            return text[:-2]
        return text

    return (
        data[[source_col, target_col]]
        .dropna()
        .drop_duplicates(source_col)
        .assign(
            **{
                source_col: lambda df: df[source_col].map(normalise),
                target_col: lambda df: df[target_col].map(normalise),
            }
        )
        .set_index(source_col)[target_col]
        .astype(str)
    )


def compose_busmaps(simplify_busmap_file: Path, cluster_busmap_file: Path) -> pd.Series:
    simplify_busmap = read_busmap(simplify_busmap_file)
    cluster_busmap = read_busmap(cluster_busmap_file)
    composed = simplify_busmap.map(cluster_busmap).dropna()
    direct_cluster = cluster_busmap[~cluster_busmap.index.isin(composed.index)]
    return pd.concat([composed, direct_cluster]).astype(str)


def aggregate_profile_to_buses(
    profile: pd.DataFrame, buses: pd.Index, busmap: pd.Series
) -> pd.DataFrame:
    buses = pd.Index(buses).astype(str)
    result = pd.DataFrame(0.0, index=profile.index, columns=buses)
    direct = pd.Index(profile.columns).intersection(buses)
    if len(direct):
        result.loc[:, direct] += profile.loc[:, direct]

    remaining = pd.Index(profile.columns).difference(direct)
    mapped = busmap.reindex(remaining).dropna()
    mapped = mapped[mapped.isin(buses)]
    for bus, source_cols in mapped.groupby(mapped):
        result.loc[:, bus] += profile.loc[:, source_cols.index].sum(axis=1)
    return result


def access_rates_for_run(repo_root: Path, run: RunPair, buses: pd.Index) -> pd.Series:
    path = repo_root / ACCESS_RATE_FILE
    if not path.exists():
        return pd.Series(1.0, index=buses, dtype=float)

    data = pd.read_csv(path)
    if "year" in data.columns:
        data = data[data.year.astype(str) == str(run.year)]
    if data.empty:
        return pd.Series(1.0, index=buses, dtype=float)

    row = data.iloc[-1]
    if "access_rate" in data.columns:
        rate = float(row.access_rate)
    elif {"access_rate_urban", "access_rate_rural"} <= set(data.columns):
        urban_share = float(row.get("urban_share", 0.5))
        if urban_share > 1:
            urban_share /= 100.0
        urban = float(row.access_rate_urban)
        rural = float(row.access_rate_rural)
        if max(urban, rural) > 1:
            urban /= 100.0
            rural /= 100.0
        rate = urban * urban_share + rural * (1.0 - urban_share)
    else:
        rate = 1.0
    return pd.Series(rate, index=buses, dtype=float).clip(lower=0.0, upper=1.0)


def base_demand_by_bus(repo_root: Path, run: RunPair, n: pypsa.Network) -> pd.DataFrame:
    buses = ac_buses(n)
    resources = run_resource_dir(run)
    base_profile = (
        repo_root / resources / f"offgrid_base_demand_profiles_{BASE_DEMAND_YEAR}.csv"
    )
    simplify_busmap_file = repo_root / resources / "bus_regions" / "busmap_elec_s.csv"
    cluster_busmap_file = repo_root / resources / "bus_regions" / "busmap_elec_s_6.csv"
    if (
        not base_profile.exists()
        or not simplify_busmap_file.exists()
        or not cluster_busmap_file.exists()
    ):
        return pd.DataFrame(0.0, index=n.snapshots, columns=buses)

    profile = align_profile_to_network_snapshots(read_profile_csv(base_profile), n.snapshots)
    busmap = compose_busmaps(simplify_busmap_file, cluster_busmap_file)
    return aggregate_profile_to_buses(profile, buses, busmap)


def demand_components(repo_root: Path, networks: dict[tuple[str, str], pypsa.Network]):
    rows = []
    for run in RUN_PAIRS:
        key = (run.case_id, "offgrid")
        if key not in networks:
            continue
        n = networks[key]
        parent_loads = offgrid_relevant_loads_by_parent(n)
        buses = ac_buses(n)
        total = pd.DataFrame(0.0, index=n.snapshots, columns=buses)
        for bus, loads in parent_loads.items():
            total.loc[:, bus] = load_profiles(n, pd.Index(loads)).sum(axis=1)

        base = base_demand_by_bus(repo_root, run, n).reindex_like(total).fillna(0.0)
        access = access_rates_for_run(repo_root, run, buses)
        served_projection = total.multiply(access.reindex(buses).fillna(1.0), axis=1)
        growth = (served_projection - base).clip(lower=0.0)
        uplift = (total - served_projection).clip(lower=0.0)
        competitive = (total - base).clip(lower=0.0)

        for bus in buses:
            values = {
                f"{BASE_DEMAND_YEAR} base demand": weighted_twh(n, base[bus]),
                "Projected growth above base": weighted_twh(n, growth[bus]),
                "Electrification uplift": weighted_twh(n, uplift[bus]),
                "Eligible demand": weighted_twh(n, competitive[bus]),
                "Total demand": weighted_twh(n, total[bus]),
            }
            for segment, value in values.items():
                rows.append(
                    {
                        "case_id": run.case_id,
                        "case_label": run.label,
                        "foresight": run.foresight,
                        "model": run.model,
                        "year": run.year,
                        "bus": bus,
                        "segment": segment,
                        "demand_twh": value,
                    }
                )
    return pd.DataFrame(rows)


def plot_run_status(summary: pd.DataFrame, figures_dir: Path) -> Path:
    columns = [
        "case_label",
        "variant",
        "model",
        "bus_count",
        "ac_bus_count",
        "h2_bus_count",
        "offgrid_bus_count",
        "other_bus_count",
        "offgrid_demand_count",
        "offgrid_demand_relief_count",
        "offgrid_solar_count",
        "offgrid_oil_generator_count",
        "offgrid_oil_link_count",
        "offgrid_battery_count",
    ]
    table = summary[columns].copy()
    table.columns = [
        "Case",
        "Variant",
        "Model",
        "Total",
        "AC",
        "H2",
        "Offgrid AC",
        "Other",
        "Demand sink",
        "Demand relief",
        "Solar",
        "Oil gen",
        "Oil link",
        "Battery",
    ]
    table["Case"] = table["Case"].map(short_case_label)

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.axis("off")
    ax.set_title(
        "1. Run status and offgrid component check",
        loc="left",
        pad=14,
        fontweight="bold",
    )
    cell_colours = []
    for _, row in table.iterrows():
        color = "#e8f2ec" if row["Variant"] == "offgrid" else "#f1f1f1"
        cell_colours.append([color] * len(table.columns))
    mpl_table = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        cellColours=cell_colours,
        colWidths=[
            0.12,
            0.065,
            0.06,
            0.045,
            0.04,
            0.04,
            0.07,
            0.045,
            0.07,
            0.075,
            0.045,
            0.05,
            0.05,
            0.05,
        ],
        colLoc="center",
        cellLoc="center",
        loc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(7)
    mpl_table.scale(1.0, 1.35)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    add_subtitle(
        fig,
        "V3 check: offgrid variants contain one isolated offgrid bus plus demand sink/relief pair per AC bus; references contain none.",
    )
    return savefig(fig, figures_dir, "01_run_status_matrix.png")


def plot_demand_comparison(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    components: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    ordered_cases = [run.label for run in RUN_PAIRS]
    ordered_segments = [
        f"{BASE_DEMAND_YEAR} base demand",
        "Projected growth above base",
        "Electrification uplift",
    ]
    pivot = components.pivot_table(
        index="case_label",
        columns="segment",
        values="demand_twh",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=ordered_cases, columns=ordered_segments, fill_value=0.0)
    reference = summary[summary.variant == "reference"].set_index("case_label")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    colors = ["#4c78a8", "#f28e2b", "#59a14f"]
    for segment, color in zip(ordered_segments, colors):
        values = pivot[segment].values
        ax.bar(x, values, bottom=bottom, label=segment, color=color)
        bottom += values

    ref_values = (
        reference.reindex(ordered_cases)["gross_electricity_demand_twh"]
        .fillna(0.0)
        .values
    )
    ax.scatter(x, ref_values, marker="_", s=700, color="#111111", label="Reference total")
    for i, total in enumerate(bottom):
        eligible = pivot.iloc[i]["Projected growth above base"] + pivot.iloc[i]["Electrification uplift"]
        share = eligible / total if total else 0.0
        ax.text(i, total, f"{eligible:.1f} TWh eligible\n({share:.0%})", ha="center", va="bottom", fontsize=8)

    if len(bottom) and np.nanmax(bottom) > 0:
        ax.set_ylim(0, np.nanmax(bottom) * 1.22)
    ax.set_ylabel("Electricity demand [TWh/a]")
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_case_labels(pivot.index), rotation=0, ha="center")
    ax.legend(ncol=2, loc="upper left")
    fig.suptitle("2. Demand components", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    add_subtitle(
        fig,
        f"Stacked bars decompose offgrid gross demand into {BASE_DEMAND_YEAR} baseline, projected demand growth above baseline, and electrification uplift.",
    )
    return savefig(fig, figures_dir, "02_demand_comparison.png")


def plot_nodal_demand(
    nodal: pd.DataFrame, components: pd.DataFrame, figures_dir: Path
) -> Path:
    cases = [run.label for run in RUN_PAIRS]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharey=False)
    axes = axes.flatten()
    for ax, label in zip(axes, cases):
        data = components[
            (components.case_label == label)
            & components.segment.isin(
                [
                    f"{BASE_DEMAND_YEAR} base demand",
                    "Projected growth above base",
                    "Electrification uplift",
                ]
            )
        ].copy()
        pivot = data.pivot_table(
            index="bus",
            columns="segment",
            values="demand_twh",
            aggfunc="sum",
            fill_value=0.0,
        )
        ordered_segments = [
            f"{BASE_DEMAND_YEAR} base demand",
            "Projected growth above base",
            "Electrification uplift",
        ]
        for column in ordered_segments:
            if column not in pivot.columns:
                pivot[column] = 0.0
        pivot = pivot[ordered_segments]
        pivot = pivot.assign(total=pivot.sum(axis=1)).sort_values("total", ascending=False)
        x = np.arange(len(pivot))
        bottom = np.zeros(len(pivot))
        colors = ["#4c78a8", "#f28e2b", "#59a14f"]
        for segment, color in zip(ordered_segments, colors):
            values = pivot[segment].values
            ax.bar(x, values, bottom=bottom, color=color, label=segment)
            bottom += values
        ax.set_title(label, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("TWh/a")
    for ax in axes[len(cases) :]:
        ax.axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("3. Nodal demand components", x=0.01, ha="left", fontweight="bold")
    add_subtitle(
        fig,
        f"Bars show demand components per final AC bus: {BASE_DEMAND_YEAR} base stays on-grid, growth plus access uplift is eligible for the split.",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return savefig(fig, figures_dir, "03_nodal_demand_distribution.png")


def plot_demand_split(
    split: pd.DataFrame, components: pd.DataFrame, figures_dir: Path
) -> Path:
    ordered_cases = [run.label for run in RUN_PAIRS]
    component_pivot = components.pivot_table(
        index="case_label",
        columns="segment",
        values="demand_twh",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=ordered_cases, fill_value=0.0)
    split_pivot = split.pivot_table(
        index="case_label",
        columns="segment",
        values="demand_twh",
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(index=ordered_cases, fill_value=0.0)

    base = component_pivot.get(
        f"{BASE_DEMAND_YEAR} base demand", pd.Series(0.0, index=ordered_cases)
    )
    growth = component_pivot.get(
        "Projected growth above base", pd.Series(0.0, index=ordered_cases)
    )
    uplift = component_pivot.get(
        "Electrification uplift", pd.Series(0.0, index=ordered_cases)
    )
    eligible = growth + uplift
    offgrid = split_pivot.get("Offgrid demand", pd.Series(0.0, index=ordered_cases))
    eligible_on_grid = (eligible - offgrid).clip(lower=0.0)

    pivot = pd.DataFrame(
        {
            f"{BASE_DEMAND_YEAR} base on-grid": base,
            "Eligible demand served on-grid": eligible_on_grid,
            "Eligible demand served offgrid": offgrid,
        }
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    colors = ["#4c78a8", "#f28e2b", "#59a14f"]
    for column, color in zip(pivot.columns, colors):
        values = pivot[column].values
        ax.bar(x, values, bottom=bottom, label=column, color=color)
        bottom += values
    for i, total in enumerate(bottom):
        eligible_total = eligible.iloc[i] if i < len(eligible) else 0.0
        offgrid_value = offgrid.iloc[i] if i < len(offgrid) else 0.0
        share = offgrid_value / eligible_total if eligible_total else 0.0
        ax.text(i, total, f"{share:.1%} of eligible\noffgrid", ha="center", va="bottom", fontsize=8)
    if len(bottom) and np.nanmax(bottom) > 0:
        ax.set_ylim(0, np.nanmax(bottom) * 1.18)
    ax.set_title("4. Endogenous on-grid/offgrid demand split", loc="left", fontweight="bold")
    ax.set_ylabel("Electricity demand [TWh/a]")
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_case_labels(pivot.index), rotation=0, ha="center")
    ax.legend(ncol=2, loc="upper right")
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    add_subtitle(
        fig,
        "Only projected growth and electrification uplift are eligible for alpha; the configured base-demand profile is fixed on-grid.",
    )
    return savefig(fig, figures_dir, "04_demand_split.png")


def plot_capacity(capacities: pd.DataFrame, figures_dir: Path) -> Path:
    pivot = capacities.pivot_table(
        index="case_label", columns="carrier", values="capacity_mw", aggfunc="sum", fill_value=0.0
    )
    ordered_cases = [run.label for run in RUN_PAIRS]
    ordered_carriers = [
        "Offgrid demand peak",
        "Offgrid solar",
        "Offgrid battery",
        "Offgrid oil/diesel",
        "Offgrid load shedding",
    ]
    pivot = pivot.reindex(index=ordered_cases, columns=ordered_carriers, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot))
    width = 0.15
    colors = ["#9b5d3a", "#f2cf5b", "#8da0cb", "#595959", "#e15759"]
    offsets = (np.arange(len(ordered_carriers)) - (len(ordered_carriers) - 1) / 2) * width
    for offset, carrier, color in zip(offsets, ordered_carriers, colors):
        ax.bar(x + offset, pivot[carrier], width, label=carrier, color=color)
    ax.set_title("5. Offgrid demand and technology capacities", loc="left", fontweight="bold")
    ax.set_ylabel("MW")
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_case_labels(pivot.index), rotation=0, ha="center")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    add_subtitle(
        fig,
        "Offgrid demand peak is the optimised eligible-demand amplitude. Solar, battery, oil/diesel and load shedding are isolated offgrid options.",
    )
    return savefig(fig, figures_dir, "05_offgrid_capacity.png")


def plot_offgrid_energy(generation: pd.DataFrame, figures_dir: Path) -> Path:
    pivot = generation.pivot_table(
        index="case_label", columns="source", values="delivery_twh", aggfunc="sum", fill_value=0.0
    )
    ordered_cases = [run.label for run in RUN_PAIRS]
    ordered_sources = [
        "Offgrid solar",
        "Offgrid battery",
        "Offgrid oil/diesel",
        "Offgrid load shedding",
    ]
    pivot = pivot.reindex(index=ordered_cases, columns=ordered_sources, fill_value=0.0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(pivot))
    width = 0.18
    colors = ["#f2cf5b", "#8da0cb", "#595959", "#e15759"]
    offsets = (np.arange(len(ordered_sources)) - (len(ordered_sources) - 1) / 2) * width
    for offset, source, color in zip(offsets, ordered_sources, colors):
        ax.bar(x + offset, pivot[source], width, label=source, color=color)
    ax.set_title("5b. Offgrid technology dispatch and load shedding", loc="left", fontweight="bold")
    ax.set_ylabel("TWh/a")
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_case_labels(pivot.index), rotation=0, ha="center")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    add_subtitle(
        fig,
        "Energy use is the relevant check for load shedding: fixed feasibility capacity can be large while dispatch remains zero.",
    )
    return savefig(fig, figures_dir, "05b_offgrid_energy.png")


def plot_line_expansion(lines: pd.DataFrame, figures_dir: Path) -> Path:
    labels = [run.label for run in RUN_PAIRS]
    offgrid = lines[lines.variant == "offgrid"].set_index("case_label").reindex(labels)
    reference = lines[lines.variant == "reference"].set_index("case_label").reindex(labels)
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].bar(
        x - width / 2,
        reference["line_expansion_mwkm"],
        width,
        label="Reference",
        color="#7f8c8d",
    )
    axes[0].bar(
        x + width / 2,
        offgrid["line_expansion_mwkm"],
        width,
        label="Offgrid",
        color="#2b8cbe",
    )
    axes[0].set_ylabel("MWkm")
    axes[0].set_title("AC line expansion", loc="left")
    axes[0].legend(ncol=2, loc="upper left")

    axes[1].bar(
        x - width / 2,
        reference["line_volume_mwkm"],
        width,
        label="Reference",
        color="#7f8c8d",
    )
    axes[1].bar(
        x + width / 2,
        offgrid["line_volume_mwkm"],
        width,
        label="Offgrid",
        color="#2b8cbe",
    )
    axes[1].set_ylabel("MWkm")
    axes[1].set_title("AC line volume after optimisation", loc="left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(wrapped_case_labels(labels), rotation=0, ha="center")
    axes[1].legend(ncol=2, loc="upper left")
    fig.suptitle("6. Transmission expansion indicators", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])
    add_subtitle(
        fig,
        "No local grid-connection link exists in v3; line metrics only describe the AC transmission network.",
    )
    return savefig(fig, figures_dir, "06_line_expansion.png")


def plot_objective_delta(comparisons: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5))
    values = comparisons.set_index("case_label")["objective_delta_bn"].reindex(
        [run.label for run in RUN_PAIRS]
    )
    colors = ["#4daf4a" if value <= 0 else "#e15759" for value in values]
    x = np.arange(len(values))
    ax.bar(x, values.values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    for i, value in enumerate(values.values):
        ax.text(
            i,
            value,
            f"{value:+.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax.set_title("7. Objective delta: offgrid minus reference", loc="left", fontweight="bold")
    ax.set_ylabel("Objective delta [bn]")
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_case_labels(values.index), rotation=0, ha="center")
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    add_subtitle(
        fig,
        "With no demand uplift, a correctly optional offgrid split should not solve more expensive than the reference beyond numerical tolerance.",
    )
    return savefig(fig, figures_dir, "07_objective_delta.png")


def plot_dispatch_examples(dispatch: pd.DataFrame, figures_dir: Path) -> Path:
    examples = ["Overnight Elec 2030", "Overnight Sector 2040"]
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 7), sharex=False)
    if len(examples) == 1:
        axes = [axes]

    for ax, label in zip(axes, examples):
        data = dispatch[dispatch.case_label == label].head(32).copy()
        if data.empty:
            ax.axis("off")
            continue
        x = np.arange(len(data))
        ax.plot(
            x,
            data["gross_electricity_demand_mw"],
            color="#111111",
            linewidth=1.4,
            label="Gross electricity demand",
        )
        ax.plot(
            x,
            data["on_grid_electricity_demand_mw"],
            color="#4c78a8",
            linewidth=1.2,
            label="On-grid demand after relief",
        )
        ax.stackplot(
            x,
            data["offgrid_solar_mw"],
            data["offgrid_oil_generator_mw"] + data["offgrid_oil_link_mw"],
            data["offgrid_battery_dispatch_mw"],
            data["offgrid_load_shedding_mw"],
            labels=[
                "Offgrid solar",
                "Offgrid oil/diesel",
                "Offgrid battery",
                "Offgrid load shedding",
            ],
            colors=["#f2cf5b", "#595959", "#8da0cb", "#e15759"],
            alpha=0.75,
        )
        ax.plot(
            x,
            data["offgrid_demand_mw"],
            color="#59a14f",
            linewidth=1.2,
            linestyle="--",
            label="Offgrid demand",
        )
        ax.set_title(label, loc="left")
        ax.set_ylabel("MW")
        step = max(len(x) // 8, 1)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(data["snapshot"].iloc[::step], rotation=25, ha="right", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("8. Dispatch examples", x=0.01, ha="left", fontweight="bold")
    add_subtitle(
        fig,
        "First 32 snapshots show gross demand, on-grid residual demand and isolated local generation against eligible offgrid demand.",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.90])
    return savefig(fig, figures_dir, "08_dispatch_examples.png")


def write_manifest(output_dir: Path, figures: list[Path], tables: list[str]) -> None:
    manifest = {
        "description": "Kenya offgrid smoke-run analysis outputs",
        "figures": [str(path.relative_to(output_dir)) for path in figures],
        "tables": [f"tables/{name}" for name in tables],
        "notes": [
            "Offgrid demand is represented by sign=-1 generators on isolated offgrid buses.",
            "If a base demand profile is configured, only demand above that baseline is eligible for the offgrid split.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("results/offgrid_analysis/kenya"),
        type=Path,
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    configure_plotting()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    (
        summary,
        comparisons,
        nodal,
        dispatch,
        capacities,
        lines,
        split,
        generation,
        components,
        _,
    ) = collect_results(repo_root, allow_missing=args.allow_missing)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    write_tables(
        tables_dir,
        summary,
        comparisons,
        nodal,
        dispatch,
        capacities,
        lines,
        split,
        generation,
        components,
    )

    figures = [
        plot_run_status(summary, figures_dir),
        plot_demand_comparison(summary, comparisons, components, figures_dir),
        plot_nodal_demand(nodal, components, figures_dir),
        plot_demand_split(split, components, figures_dir),
        plot_capacity(capacities, figures_dir),
        plot_offgrid_energy(generation, figures_dir),
        plot_line_expansion(lines, figures_dir),
        plot_objective_delta(comparisons, figures_dir),
        plot_dispatch_examples(dispatch, figures_dir),
    ]
    write_manifest(
        output_dir,
        figures,
        [
            "run_summary.csv",
            "comparison_summary.csv",
            "nodal_demand.csv",
            "dispatch_examples.csv",
            "capacity_summary.csv",
            "line_expansion.csv",
            "demand_split.csv",
            "offgrid_generation.csv",
            "demand_components.csv",
        ],
    )

    logger.info("Analysed %s run variants.", len(summary))
    logger.info("Wrote %s figures to %s.", len(figures), figures_dir)


if __name__ == "__main__":
    main()
