# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Attach optional isolated offgrid electricity systems before optimisation.

Relevant Settings
-----------------

.. code:: yaml

    offgrid:
        enable:
        access_rate_file:
        min_access_rate:
        costs:

Inputs
------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: prepared
  electricity network, or the corresponding sector-coupled pre-network.
- ``resources/costs_*.csv``: processed electricity or sector cost assumptions.
- optional ``offgrid.access_rate_file``: CSV with direct ``access_rate`` values
  or urban/rural access-rate columns, keyed by final bus name
  (``bus``/``region``/``name``/``node``) or by ``country``.

Outputs
-------

- ``*_offgrid.nc``: input network plus one isolated offgrid bus per final AC
  bus, offgrid PV, battery storage, oil backup supply, and zero-cost demand
  split generators.

Description
-----------

The rule :mod:`add_offgrid_components` keeps each selected electricity load at
its original on-grid bus. For every final AC bus it adds an isolated offgrid bus
and two artificial generators with identical extendable nominal capacity:

- ``{bus} offgrid demand relief`` injects at the original on-grid demand bus.
- ``{bus} offgrid demand`` consumes at the isolated offgrid bus via ``sign=-1``.

Both generators use the same normalised regional load profile. Their shared
capacity is the optimisation variable for the offgrid demand share. If it is
zero, the network can reproduce the reference solution. If it is positive, the
same profile share is removed from on-grid demand and served by the isolated
offgrid system.

If electricity access rates are provided, existing load is interpreted as
served demand and scaled to total demand via ``D_total = D_served / access_rate``.
PV, battery, and oil capacities are optimised. In sector-coupled networks oil is
represented as a link from an oil bus to the offgrid bus, with an optional CO2
output to ``co2 atmosphere``.
"""

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa

from _helpers import (
    configure_logging,
    create_logger,
    read_csv_nafix,
    sanitize_carriers,
    sanitize_locations,
)

logger = create_logger(__name__)

OFFGRID_BUS_SUFFIX = " offgrid"
OFFGRID_SOLAR_SUFFIX = " offgrid solar"
OFFGRID_BATTERY_SUFFIX = " offgrid battery"
OFFGRID_OIL_SUFFIX = " offgrid oil"
OFFGRID_DEMAND_SUFFIX = " offgrid demand"
OFFGRID_DEMAND_RELIEF_SUFFIX = " offgrid demand relief"

OFFGRID_CARRIER = "offgrid AC"
OFFGRID_SOLAR_CARRIER = "offgrid solar"
OFFGRID_BATTERY_CARRIER = "offgrid battery"
OFFGRID_OIL_CARRIER = "offgrid oil"
OFFGRID_DEMAND_CARRIER = "offgrid demand"
OFFGRID_DEMAND_RELIEF_CARRIER = "offgrid demand relief"

DIRECT_ACCESS_COLUMNS = ("access_rate", "electricity_access", "electrification_rate")
URBAN_ACCESS_COLUMNS = (
    "access_rate_urban",
    "electricity_access_urban",
    "electrification_rate_urban",
)
RURAL_ACCESS_COLUMNS = (
    "access_rate_rural",
    "electricity_access_rural",
    "electrification_rate_rural",
)
URBAN_SHARE_COLUMNS = ("urban_share", "urban_fraction", "share_urban")


def _is_unset(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "false", "none", "null"}
    return False


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cost_value(
    costs: pd.DataFrame,
    technology: str,
    columns: list[str],
    default: float | None = None,
) -> float:
    if technology not in costs.index:
        if default is not None:
            return default
        raise KeyError(f"Technology '{technology}' is missing from the cost table.")

    for column in columns:
        if column in costs.columns:
            value = _as_float(costs.at[technology, column])
            if value is not None:
                return value

    if default is not None:
        return default

    raise KeyError(
        f"None of the columns {columns} is available for '{technology}' in the cost table."
    )


def _capital_cost(costs: pd.DataFrame, technology: str) -> float:
    return _cost_value(costs, technology, ["capital_cost", "fixed"])


def _marginal_cost(
    costs: pd.DataFrame, technology: str, default: float = 0.0
) -> float:
    return _cost_value(costs, technology, ["marginal_cost", "VOM"], default=default)


def _efficiency(costs: pd.DataFrame, technology: str, default: float = 1.0) -> float:
    return _cost_value(costs, technology, ["efficiency"], default=default)


def _lifetime(costs: pd.DataFrame, technology: str, default: float = np.inf) -> float:
    return _cost_value(costs, technology, ["lifetime"], default=default)


def _co2_intensity(costs: pd.DataFrame, technology: str, default: float = 0.0) -> float:
    return _cost_value(
        costs, technology, ["co2_emissions", "CO2 intensity"], default=default
    )


def _annuity(lifetime: float, discount_rate: float) -> float:
    lifetime = float(lifetime)
    discount_rate = float(discount_rate)
    if lifetime <= 0:
        raise ValueError("offgrid.costs.asset_lifetime must be greater than zero.")
    if discount_rate == 0:
        return 1.0 / lifetime
    return discount_rate / (1.0 - 1.0 / (1.0 + discount_rate) ** lifetime)


def _explicit_offgrid_costs(offgrid_config: dict[str, Any]) -> dict[str, float] | None:
    custom_costs = offgrid_config.get("costs", False)
    if _is_unset(custom_costs):
        return None
    if not isinstance(custom_costs, dict):
        raise TypeError("offgrid.costs must be false or a mapping.")

    required = [
        "solar_capex",
        "battery_capex_kwh",
        "battery_max_hours",
        "diesel_capex",
        "diesel_marginal",
        "discount_rate",
        "asset_lifetime",
    ]
    missing = [key for key in required if _as_float(custom_costs.get(key)) is None]
    if missing:
        raise ValueError("offgrid.costs is missing numeric values for: " + ", ".join(missing))

    discount_rate = float(custom_costs["discount_rate"])
    lifetime = float(custom_costs["asset_lifetime"])
    battery_max_hours = float(custom_costs["battery_max_hours"])
    if battery_max_hours <= 0:
        raise ValueError("offgrid.costs.battery_max_hours must be greater than zero.")

    annuity = _annuity(lifetime, discount_rate)
    return {
        "solar_capital_cost": float(custom_costs["solar_capex"]) * 1e3 * annuity,
        "battery_capital_cost": float(custom_costs["battery_capex_kwh"])
        * 1e3
        * battery_max_hours
        * annuity,
        "battery_max_hours": battery_max_hours,
        "diesel_capital_cost": float(custom_costs["diesel_capex"]) * 1e3 * annuity,
        "diesel_marginal_cost": float(custom_costs["diesel_marginal"]),
        "asset_lifetime": lifetime,
    }


def _offgrid_capital_cost(
    costs: pd.DataFrame,
    technology: str,
    offgrid_config: dict[str, Any],
) -> float:
    explicit = _explicit_offgrid_costs(offgrid_config)
    if explicit is None:
        return _capital_cost(costs, technology)
    if technology == "solar":
        return explicit["solar_capital_cost"]
    if technology == "battery":
        return explicit["battery_capital_cost"]
    if technology == "oil":
        return explicit["diesel_capital_cost"]
    return _capital_cost(costs, technology)


def _offgrid_marginal_cost(
    costs: pd.DataFrame,
    technology: str,
    offgrid_config: dict[str, Any],
    default: float = 0.0,
) -> float:
    explicit = _explicit_offgrid_costs(offgrid_config)
    if explicit is not None and technology == "oil":
        return explicit["diesel_marginal_cost"]
    return _marginal_cost(costs, technology, default=default)


def _offgrid_lifetime(
    costs: pd.DataFrame,
    technology: str,
    offgrid_config: dict[str, Any],
    default: float = np.inf,
) -> float:
    explicit = _explicit_offgrid_costs(offgrid_config)
    if explicit is not None and technology in {"solar", "battery", "oil"}:
        return explicit["asset_lifetime"]
    return _lifetime(costs, technology, default=default)


def _offgrid_battery_max_hours(
    electricity: dict[str, Any], offgrid_config: dict[str, Any]
) -> float:
    explicit = _explicit_offgrid_costs(offgrid_config)
    if explicit is not None:
        return explicit["battery_max_hours"]
    return electricity.get("max_hours", {}).get("battery", 6)


def _normalise_access_rates(access_rates: pd.Series) -> pd.Series:
    rates = pd.to_numeric(access_rates, errors="coerce")
    values = rates.dropna()

    if values.empty:
        return rates

    if (values < 0).any():
        raise ValueError("Access rates must be non-negative.")

    if values.max() > 1:
        if values.min() <= 1:
            raise ValueError(
                "Access rates must be provided consistently either as fractions "
                "(0-1) or percentages (0-100)."
            )
        if values.max() > 100:
            raise ValueError("Access rates above 100% are invalid.")
        rates = rates / 100.0

    return rates


def _snapshot_weights(
    index: pd.Index, snapshot_weightings: pd.Series | pd.DataFrame | None
) -> pd.Series:
    if snapshot_weightings is None:
        return pd.Series(1.0, index=index)

    if isinstance(snapshot_weightings, pd.DataFrame):
        if "generators" in snapshot_weightings.columns:
            weights = snapshot_weightings["generators"]
        elif "objective" in snapshot_weightings.columns:
            weights = snapshot_weightings["objective"]
        else:
            weights = snapshot_weightings.iloc[:, 0]
    else:
        weights = snapshot_weightings

    return pd.to_numeric(weights.reindex(index), errors="coerce").fillna(1.0)


def _demand_multipliers(
    access_rates: pd.Series,
    buses: pd.Index,
    min_access_rate: float = 0.05,
) -> pd.Series:
    min_access_rate = float(min_access_rate)
    if min_access_rate <= 0 or min_access_rate > 1:
        raise ValueError("offgrid.min_access_rate must be greater than 0 and <= 1.")

    rates = access_rates.reindex(buses)
    rates = _normalise_access_rates(rates).fillna(1.0)
    rates = rates.clip(lower=min_access_rate, upper=1.0)

    return 1.0 / rates


def calculate_total_demand(
    served_load: pd.DataFrame,
    access_rates: pd.Series,
    min_access_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Scale served demand to total demand using electricity access rates.

    Existing regional load is interpreted as served electricity demand. If the
    access rate is below one, total demand follows
    ``D_total = D_served / access_rate`` with access rates clipped to
    ``[min_access_rate, 1]``.
    """
    if served_load.empty:
        return served_load.copy()

    multipliers = _demand_multipliers(
        access_rates, served_load.columns, min_access_rate=min_access_rate
    )

    return served_load.multiply(multipliers, axis=1).clip(lower=0.0)


def calculate_competitive_demand(
    total_demand: pd.DataFrame,
    base_demand: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Return the load share eligible for the endogenous grid-vs-offgrid split.

    If a base demand profile is supplied, this represents already electrified
    demand and remains fixed on-grid. Only demand above that profile can be
    shifted to isolated offgrid supply.
    """
    total_demand = total_demand.clip(lower=0.0)
    if base_demand is None:
        return total_demand

    base = base_demand.reindex_like(total_demand).fillna(0.0).clip(lower=0.0)
    return (total_demand - base).clip(lower=0.0)


def get_ac_buses(n: pypsa.Network) -> pd.Index:
    """Return final AC buses which define the offgrid spatial units."""
    if "carrier" in n.buses.columns:
        ac_buses = n.buses.index[n.buses.carrier.fillna("") == "AC"]
        if not ac_buses.empty:
            return pd.Index(ac_buses)

    load_buses = pd.Index(n.loads.bus.dropna().unique())
    return load_buses.intersection(n.buses.index)


def _normalise_scope(scope: str | None) -> str:
    if scope is None:
        return "electricity"
    scope = scope.lower()
    if scope in {"elec", "electricity", "power"}:
        return "electricity"
    if scope in {"sec", "sector", "sector-coupled", "sector_coupled"}:
        return "sector"
    raise ValueError(f"Unsupported offgrid scope '{scope}'.")


def _load_profile(n: pypsa.Network, load: str) -> pd.Series:
    if load in n.loads_t.p_set.columns:
        return pd.to_numeric(n.loads_t.p_set[load], errors="coerce").fillna(0.0)
    return pd.Series(_as_float(n.loads.at[load, "p_set"], 0.0), index=n.snapshots)


def _load_parent_map(
    n: pypsa.Network, parent_buses: pd.Index, sector_scope: bool
) -> pd.Series:
    """
    Map electric loads that can be split between grid and offgrid by AC parent.

    In sector scope only the original AC electricity load is considered. Sector
    loads such as heat, transport, and industry remain attached to their
    existing buses and are not part of this first offgrid split.
    """
    if n.loads.empty:
        return pd.Series(dtype=object)

    parent_set = set(parent_buses)
    carriers = n.loads.get("carrier", pd.Series("", index=n.loads.index)).fillna("")
    load_to_parent = {}

    for load, bus in n.loads.bus.fillna("").items():
        bus = str(bus)
        parent = None

        if bus in parent_set:
            parent = bus
        elif sector_scope and bus.endswith(" low voltage"):
            candidate = bus[: -len(" low voltage")]
            if candidate in parent_set:
                parent = candidate

        if parent is None:
            continue

        if sector_scope and carriers.at[load] not in {"", "AC"}:
            continue

        load_to_parent[load] = parent

    return pd.Series(load_to_parent, dtype=object)


def aggregate_served_load_by_bus(
    n: pypsa.Network,
    buses: pd.Index | list[str],
    load_to_parent: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate selected existing Load components by AC parent bus."""
    buses = pd.Index(buses)
    served_load = pd.DataFrame(0.0, index=n.snapshots, columns=buses)

    if n.loads.empty:
        return served_load

    if load_to_parent is None:
        load_to_parent = _load_parent_map(n, buses, sector_scope=False)

    for load, parent in load_to_parent.items():
        if parent not in served_load.columns:
            continue
        served_load.loc[:, parent] += _load_profile(n, load)

    return served_load


def _read_demand_profiles_csv(path: str | Path) -> pd.DataFrame:
    demand = read_csv_nafix(path, index_col=0)
    demand.index = pd.to_datetime(demand.index)
    demand.columns = demand.columns.astype(str)
    return demand.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _align_demand_to_snapshots(
    demand: pd.DataFrame, snapshots: pd.Index, path: str | Path
) -> pd.DataFrame:
    snapshots = pd.Index(snapshots)
    if demand.index.equals(snapshots):
        return demand

    frequency = getattr(snapshots, "freqstr", None)
    if frequency is None and len(snapshots) >= 3:
        frequency = pd.infer_freq(snapshots)
    if frequency is None and len(snapshots) >= 2:
        frequency = snapshots[1] - snapshots[0]
    if frequency is not None and len(demand.index.difference(snapshots)) > 0:
        offset = frequency.casefold() if isinstance(frequency, str) else frequency
        resampled = demand.resample(offset).mean()
        missing = snapshots.difference(resampled.index)
        if len(missing) == 0:
            return resampled.reindex(snapshots).fillna(0.0)

    missing_snapshots = snapshots.difference(demand.index)
    if len(missing_snapshots):
        raise ValueError(
            f"Offgrid base demand file '{path}' misses {len(missing_snapshots)} "
            "snapshots required by the network."
        )
    return demand.reindex(snapshots).fillna(0.0)


def _normalise_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def _read_busmap(busmap_file: str | Path | bool | None) -> pd.Series:
    if _is_unset(busmap_file):
        return pd.Series(dtype=object)

    busmap = read_csv_nafix(busmap_file)
    if {"Bus", "busmap"} <= set(busmap.columns):
        source_col, target_col = "Bus", "busmap"
    elif {"bus", "busmap"} <= set(busmap.columns):
        source_col, target_col = "bus", "busmap"
    elif busmap.shape[1] >= 2:
        source_col, target_col = busmap.columns[:2]
    else:
        raise ValueError(
            f"Offgrid base demand busmap '{busmap_file}' needs at least two columns."
        )

    mapping = busmap[[source_col, target_col]].dropna().drop_duplicates(source_col)
    mapping[source_col] = mapping[source_col].map(_normalise_label)
    mapping[target_col] = mapping[target_col].map(_normalise_label)
    return mapping.set_index(source_col)[target_col]


def _compose_busmaps(
    simplify_busmap_file: str | Path | bool | None,
    cluster_busmap_file: str | Path | bool | None,
) -> pd.Series:
    simplify_busmap = _read_busmap(simplify_busmap_file)
    cluster_busmap = _read_busmap(cluster_busmap_file)

    if simplify_busmap.empty:
        return cluster_busmap
    if cluster_busmap.empty:
        return simplify_busmap

    composed = simplify_busmap.map(cluster_busmap).dropna()
    direct_cluster = cluster_busmap[~cluster_busmap.index.isin(composed.index)]
    return pd.concat([composed, direct_cluster]).astype(str)


def load_base_demand_data(
    base_demand_file: str | Path | bool | None,
    snapshots: pd.Index,
    parent_buses: pd.Index,
    busmap_file: str | Path | bool | None = None,
    simplify_busmap_file: str | Path | bool | None = None,
) -> pd.DataFrame | None:
    """Load and aggregate an existing-demand profile to final AC buses."""
    if _is_unset(base_demand_file):
        return None

    path = Path(base_demand_file)
    if not path.exists():
        raise FileNotFoundError(f"Offgrid base demand file does not exist: {path}")

    demand = _align_demand_to_snapshots(
        _read_demand_profiles_csv(path), snapshots, path
    )

    parent_buses = pd.Index(parent_buses).astype(str)
    aggregated = pd.DataFrame(0.0, index=snapshots, columns=parent_buses)
    direct_columns = pd.Index(demand.columns).intersection(parent_buses)
    if len(direct_columns):
        aggregated.loc[:, direct_columns] += demand.loc[:, direct_columns]

    remaining_columns = pd.Index(demand.columns).difference(direct_columns)
    busmap = _compose_busmaps(simplify_busmap_file, busmap_file)
    if not busmap.empty and len(remaining_columns):
        mapped = busmap.reindex(remaining_columns).dropna()
        mapped = mapped[mapped.isin(parent_buses)]
        for parent, source_columns in mapped.groupby(mapped):
            aggregated.loc[:, parent] += demand.loc[:, source_columns.index].sum(axis=1)

    logger.info(
        "Loaded offgrid base demand from '%s' for %s AC buses.",
        path,
        int((aggregated.sum(axis=0) > 0.0).sum()),
    )
    return aggregated.clip(lower=0.0)


def _scale_loads_to_total_demand(
    n: pypsa.Network,
    load_to_parent: pd.Series,
    demand_multipliers: pd.Series,
) -> None:
    for load, parent in load_to_parent.items():
        multiplier = float(demand_multipliers.get(parent, 1.0))

        if load in n.loads_t.p_set.columns:
            n.loads_t.p_set.loc[:, load] = (
                pd.to_numeric(n.loads_t.p_set[load], errors="coerce").fillna(0.0)
                * multiplier
            )

        if "p_set" in n.loads.columns:
            n.loads.at[load, "p_set"] = (
                _as_float(n.loads.at[load, "p_set"], 0.0) or 0.0
            ) * multiplier


def _relief_buses_by_parent(
    n: pypsa.Network, parent_buses: pd.Index, load_to_parent: pd.Series
) -> pd.Series:
    relief_buses = pd.Series(parent_buses, index=parent_buses, dtype=object)

    for parent in parent_buses:
        loads = load_to_parent.index[load_to_parent == parent]
        if len(loads) == 0:
            continue

        buses = pd.Index(n.loads.loc[loads, "bus"].dropna().astype(str).unique())
        if len(buses) == 1:
            relief_buses.at[parent] = buses[0]
            continue

        raise ValueError(
            "Offgrid demand split currently requires all selected electric loads "
            f"for AC bus '{parent}' to be attached to one on-grid demand bus. "
            f"Found: {', '.join(buses)}"
        )

    missing = relief_buses[~relief_buses.isin(n.buses.index)]
    if not missing.empty:
        raise ValueError(
            "Offgrid demand relief buses are missing from the network: "
            + ", ".join(missing.astype(str).head(5))
        )

    return relief_buses


def _find_column(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


def _rate_series_from_values(values: pd.DataFrame) -> pd.Series:
    rate_col = _find_column(values.columns, DIRECT_ACCESS_COLUMNS)
    if rate_col is not None:
        return _normalise_access_rates(values[rate_col])

    urban_col = _find_column(values.columns, URBAN_ACCESS_COLUMNS)
    rural_col = _find_column(values.columns, RURAL_ACCESS_COLUMNS)
    if urban_col is None or rural_col is None:
        raise ValueError(
            "Offgrid access rate file must contain either 'access_rate' or both "
            "'access_rate_urban' and 'access_rate_rural'."
        )

    urban = _normalise_access_rates(values[urban_col])
    rural = _normalise_access_rates(values[rural_col])

    share_col = _find_column(values.columns, URBAN_SHARE_COLUMNS)
    if share_col is None:
        logger.warning(
            "Urban/rural access rates were provided without an urban share column. "
            "Using an unweighted 50/50 average."
        )
        urban_share = pd.Series(0.5, index=values.index)
    else:
        urban_share = pd.to_numeric(values[share_col], errors="coerce")
        if urban_share.dropna().max() > 1:
            urban_share = urban_share / 100.0
        urban_share = urban_share.clip(lower=0.0, upper=1.0).fillna(0.5)

    return urban * urban_share + rural * (1.0 - urban_share)


def load_access_rate_data(
    access_rate_file: str | Path | bool | None,
    buses: pd.Index,
    bus_countries: pd.Series,
    planning_horizon: str | int | None = None,
) -> pd.Series:
    """Load access rates from a bus/region/node/name keyed or country keyed CSV."""
    access_rates = pd.Series(1.0, index=buses, dtype=float)

    if _is_unset(access_rate_file):
        return access_rates

    path = Path(access_rate_file)
    if not path.exists():
        raise FileNotFoundError(f"Offgrid access rate file does not exist: {path}")

    data = read_csv_nafix(path)
    if data.empty:
        logger.warning(
            "Offgrid access rate file is empty. Assuming full access for all buses."
        )
        return access_rates

    year_col = _find_column(
        data.columns, ("year", "planning_horizon", "planning_horizons")
    )
    if year_col is not None and planning_horizon is not None:
        years = data[year_col].astype(str).str.strip()
        filtered = data.loc[years == str(planning_horizon)]
        if filtered.empty:
            raise ValueError(
                f"Offgrid access rate file contains '{year_col}', but no row for "
                f"planning horizon {planning_horizon}."
            )
        data = filtered.drop(columns=[year_col])

    key_col = _find_column(data.columns, ("bus", "region", "name", "node", "country"))
    if key_col is None:
        raise ValueError(
            "Offgrid access rate file must contain one of: bus, region, name, node, country."
        )

    values = data.dropna(subset=[key_col]).drop_duplicates(key_col, keep="last")
    values = values.set_index(key_col)
    values.index = values.index.astype(str)
    rate_values = _rate_series_from_values(values)

    if str(key_col).strip().lower() == "country":
        mapped_rates = bus_countries.reindex(buses).astype(str).map(rate_values)
        access_rates = mapped_rates.fillna(1.0).astype(float)
    else:
        access_rates.update(rate_values)

    missing = access_rates.index[access_rates.isna()]
    if not missing.empty:
        logger.warning(
            "Missing offgrid access rates for %s buses. Assuming full access.",
            len(missing),
        )
        access_rates.loc[missing] = 1.0

    return access_rates.astype(float)


def _generator_time_series(
    n: pypsa.Network, generators: pd.Index, attr: str, default: float
) -> pd.DataFrame:
    dynamic = getattr(n.generators_t, attr)
    profiles = pd.DataFrame(index=n.snapshots)

    for generator in generators:
        if generator in dynamic.columns:
            profiles[generator] = pd.to_numeric(
                dynamic[generator], errors="coerce"
            ).fillna(default)
        elif attr in n.generators.columns:
            profiles[generator] = _as_float(n.generators.at[generator, attr], default)
        else:
            profiles[generator] = default

    return profiles


def _weighted_profile(
    profiles: pd.DataFrame, weights: pd.Series | None = None
) -> pd.Series:
    if profiles.empty:
        return pd.Series(dtype=float)

    if weights is None or weights.sum() <= 0:
        return profiles.mean(axis=1)

    weights = weights.reindex(profiles.columns).fillna(0.0)
    return profiles.multiply(weights, axis=1).sum(axis=1) / weights.sum()


def solar_profiles_by_bus(
    n: pypsa.Network,
    buses: pd.Index,
    source_carrier: str = "solar",
) -> pd.DataFrame:
    """Build offgrid PV availability profiles from existing grid solar profiles."""
    profiles_by_bus = pd.DataFrame(0.0, index=n.snapshots, columns=buses)
    generators = n.generators[n.generators.carrier == source_carrier]

    if generators.empty:
        logger.warning(
            "No generators with carrier '%s' found. Offgrid solar availability is zero.",
            source_carrier,
        )
        return profiles_by_bus

    generator_profiles = _generator_time_series(
        n, pd.Index(generators.index), "p_max_pu", default=1.0
    )
    global_profile = _weighted_profile(
        generator_profiles,
        pd.to_numeric(generators.get("p_nom_max", pd.Series()), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0),
    )

    bus_countries = (
        n.buses.get("country", pd.Series("", index=n.buses.index))
        .reindex(buses)
        .fillna("")
    )

    for bus in buses:
        local_generators = pd.Index(generators.index[generators.bus == bus])

        if local_generators.empty:
            country = bus_countries.at[bus]
            same_country_buses = n.buses.index[
                n.buses.get("country", pd.Series("", index=n.buses.index)).fillna("")
                == country
            ]
            local_generators = pd.Index(
                generators.index[generators.bus.isin(same_country_buses)]
            )

        if local_generators.empty:
            profiles_by_bus.loc[:, bus] = global_profile
            continue

        local_profiles = generator_profiles[local_generators]
        local_weights = pd.to_numeric(
            generators.loc[local_generators, "p_nom_max"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        profiles_by_bus.loc[:, bus] = _weighted_profile(
            local_profiles, local_weights.fillna(1.0)
        )

    return profiles_by_bus.clip(lower=0.0, upper=1.0)


def solar_potentials_by_bus(
    n: pypsa.Network,
    buses: pd.Index,
    source_carrier: str = "solar",
) -> pd.Series:
    """Return finite source-carrier solar potential by parent AC bus."""
    potentials = pd.Series(0.0, index=buses, dtype=float)
    if n.generators.empty or "carrier" not in n.generators.columns:
        return potentials

    generators = n.generators[n.generators.carrier == source_carrier]
    if generators.empty or "p_nom_max" not in generators.columns:
        logger.warning(
            "No finite '%s' p_nom_max values found. Offgrid solar potential is zero.",
            source_carrier,
        )
        return potentials

    finite_potential = pd.to_numeric(generators.p_nom_max, errors="coerce")
    finite_potential = finite_potential.replace([np.inf, -np.inf], np.nan).dropna()
    finite_potential = finite_potential[finite_potential > 0.0]
    if finite_potential.empty:
        logger.warning(
            "No finite positive '%s' p_nom_max values found. Offgrid solar potential is zero.",
            source_carrier,
        )
        return potentials

    grouped = finite_potential.groupby(generators.loc[finite_potential.index, "bus"]).sum()
    potentials.update(grouped.reindex(buses))
    missing = potentials.index[potentials <= 0.0]
    if len(missing):
        logger.warning(
            "No finite on-grid solar potential found for %s offgrid buses. "
            "Setting their offgrid solar potential to zero. Examples: %s",
            len(missing),
            ", ".join(missing.astype(str).head(5)),
        )
    return potentials.fillna(0.0).clip(lower=0.0)


def _add_carrier(
    n: pypsa.Network,
    carrier: str,
    nice_name: str,
    color: str,
    co2_emissions: float = 0.0,
) -> None:
    if carrier not in n.carriers.index:
        n.add(
            "Carrier",
            carrier,
            nice_name=nice_name,
            color=color,
            co2_emissions=co2_emissions,
        )
    else:
        if "co2_emissions" in n.carriers.columns:
            n.carriers.loc[carrier, "co2_emissions"] = co2_emissions


def _add_offgrid_carriers(
    n: pypsa.Network, costs: pd.DataFrame, sector_scope: bool
) -> None:
    oil_co2 = 0.0 if sector_scope else _co2_intensity(costs, "oil", default=0.0)

    _add_carrier(n, OFFGRID_CARRIER, "Offgrid electricity", "#555555")
    _add_carrier(n, OFFGRID_SOLAR_CARRIER, "Offgrid solar", "#d6b100")
    _add_carrier(n, OFFGRID_BATTERY_CARRIER, "Offgrid battery", "#6d8299")
    _add_carrier(n, OFFGRID_OIL_CARRIER, "Offgrid oil", "#2c2c2c", oil_co2)
    _add_carrier(n, OFFGRID_DEMAND_CARRIER, "Offgrid demand", "#9b5d3a")
    _add_carrier(
        n,
        OFFGRID_DEMAND_RELIEF_CARRIER,
        "Offgrid demand relief",
        "#b07d62",
    )


def _emission_price_from_opts(co2_config: dict[str, Any], opts: str | None) -> float:
    if not opts:
        return 0.0

    for opt in str(opts).split("-"):
        if "Ep" not in opt:
            continue

        values = re.findall(r"[0-9]*\.?[0-9]+$", opt)
        if values:
            return float(values[0])
        return _as_float(co2_config.get("emission_price"), 0.0) or 0.0

    return 0.0


def _battery_efficiency(costs: pd.DataFrame, storage_techs: dict[str, Any]) -> float:
    battery_tech = storage_techs.get("battery", {})
    inverter = battery_tech.get("bicharger") or battery_tech.get("charger")
    if inverter and inverter in costs.index:
        return _efficiency(costs, inverter, default=1.0) ** 0.5
    if "battery inverter" in costs.index:
        return _efficiency(costs, "battery inverter", default=1.0) ** 0.5
    return 1.0


def _attach_offgrid_buses(
    n: pypsa.Network, parent_buses: pd.Index, offgrid_buses: pd.Index
) -> None:
    existing = offgrid_buses.intersection(n.buses.index)
    if not existing.empty:
        raise ValueError(
            "The network already contains offgrid buses. Existing examples: "
            + ", ".join(list(existing[:5]))
        )

    n.madd(
        "Bus",
        offgrid_buses,
        location=parent_buses,
        carrier=OFFGRID_CARRIER,
        x=n.buses.loc[parent_buses, "x"].values if "x" in n.buses else 0.0,
        y=n.buses.loc[parent_buses, "y"].values if "y" in n.buses else 0.0,
        country=(
            n.buses.loc[parent_buses, "country"].values
            if "country" in n.buses
            else ""
        ),
    )


def _normalised_demand_profile(total_demand: pd.DataFrame) -> pd.DataFrame:
    peak_demand = total_demand.max().replace(0.0, np.nan)
    profile = total_demand.divide(peak_demand, axis=1)
    return profile.fillna(0.0).clip(lower=0.0, upper=1.0)


def _attach_offgrid_demand_split_generators(
    n: pypsa.Network,
    parent_buses: pd.Index,
    offgrid_buses: pd.Index,
    relief_buses: pd.Series,
    total_demand: pd.DataFrame,
) -> None:
    demand_names = parent_buses + OFFGRID_DEMAND_SUFFIX
    relief_names = parent_buses + OFFGRID_DEMAND_RELIEF_SUFFIX
    existing = demand_names.union(relief_names).intersection(n.generators.index)
    if not existing.empty:
        raise ValueError(
            "The network already contains offgrid demand split generators. "
            "Existing examples: " + ", ".join(list(existing[:5]))
        )

    peak_demand = total_demand.max().reindex(parent_buses).fillna(0.0).clip(lower=0.0)
    profile = _normalised_demand_profile(total_demand.reindex(columns=parent_buses))

    relief_profile = profile.rename(columns=dict(zip(parent_buses, relief_names)))
    demand_profile = profile.rename(columns=dict(zip(parent_buses, demand_names)))

    n.madd(
        "Generator",
        relief_names,
        bus=relief_buses.reindex(parent_buses).values,
        location=parent_buses,
        carrier=OFFGRID_DEMAND_RELIEF_CARRIER,
        sign=1.0,
        p_nom_extendable=True,
        p_nom_max=peak_demand.values,
        p_min_pu=relief_profile,
        p_max_pu=relief_profile,
        capital_cost=0.0,
        marginal_cost=0.0,
        lifetime=np.inf,
    )
    n.madd(
        "Generator",
        demand_names,
        bus=offgrid_buses,
        location=parent_buses,
        carrier=OFFGRID_DEMAND_CARRIER,
        sign=-1.0,
        p_nom_extendable=True,
        p_nom_max=peak_demand.values,
        p_min_pu=demand_profile,
        p_max_pu=demand_profile,
        capital_cost=0.0,
        marginal_cost=0.0,
        lifetime=np.inf,
    )


def _attach_offgrid_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    parent_buses: pd.Index,
    offgrid_buses: pd.Index,
    offgrid_config: dict[str, Any],
    source_carrier: str = "solar",
) -> None:
    generator_names = parent_buses + OFFGRID_SOLAR_SUFFIX
    p_max_pu = solar_profiles_by_bus(n, parent_buses, source_carrier=source_carrier)
    p_max_pu = p_max_pu.rename(columns=dict(zip(parent_buses, generator_names)))
    p_nom_max = solar_potentials_by_bus(
        n, parent_buses, source_carrier=source_carrier
    )

    n.madd(
        "Generator",
        generator_names,
        bus=offgrid_buses,
        carrier=OFFGRID_SOLAR_CARRIER,
        p_nom_extendable=True,
        p_nom_max=p_nom_max.reindex(parent_buses).values,
        p_max_pu=p_max_pu,
        capital_cost=_offgrid_capital_cost(costs, source_carrier, offgrid_config),
        marginal_cost=_offgrid_marginal_cost(costs, source_carrier, offgrid_config),
        efficiency=_efficiency(costs, source_carrier),
        lifetime=_offgrid_lifetime(costs, source_carrier, offgrid_config),
    )


def _attach_offgrid_battery(
    n: pypsa.Network,
    costs: pd.DataFrame,
    parent_buses: pd.Index,
    offgrid_buses: pd.Index,
    electricity: dict[str, Any],
    storage_techs: dict[str, Any],
    offgrid_config: dict[str, Any],
) -> None:
    storage_names = parent_buses + OFFGRID_BATTERY_SUFFIX
    max_hours = _offgrid_battery_max_hours(electricity, offgrid_config)
    efficiency = _battery_efficiency(costs, storage_techs)

    n.madd(
        "StorageUnit",
        storage_names,
        bus=offgrid_buses,
        carrier=OFFGRID_BATTERY_CARRIER,
        p_nom_extendable=True,
        capital_cost=_offgrid_capital_cost(costs, "battery", offgrid_config),
        marginal_cost=_offgrid_marginal_cost(costs, "battery", offgrid_config),
        max_hours=max_hours,
        efficiency_store=efficiency,
        efficiency_dispatch=efficiency,
        cyclic_state_of_charge=True,
        lifetime=_offgrid_lifetime(costs, "battery", offgrid_config),
    )


def _attach_offgrid_oil_generators(
    n: pypsa.Network,
    costs: pd.DataFrame,
    parent_buses: pd.Index,
    offgrid_buses: pd.Index,
    co2_config: dict[str, Any],
    opts: str | None,
    offgrid_config: dict[str, Any],
) -> None:
    generator_names = parent_buses + OFFGRID_OIL_SUFFIX
    efficiency = _efficiency(costs, "oil")
    marginal_cost = _offgrid_marginal_cost(costs, "oil", offgrid_config)
    emission_price = _emission_price_from_opts(co2_config, opts)
    if emission_price:
        marginal_cost += emission_price * _co2_intensity(costs, "oil") / efficiency

    n.madd(
        "Generator",
        generator_names,
        bus=offgrid_buses,
        carrier=OFFGRID_OIL_CARRIER,
        p_nom_extendable=True,
        capital_cost=_offgrid_capital_cost(costs, "oil", offgrid_config),
        marginal_cost=marginal_cost,
        efficiency=efficiency,
        lifetime=_offgrid_lifetime(costs, "oil", offgrid_config),
    )


def _sector_oil_supply_marginal_cost(
    costs: pd.DataFrame, offgrid_config: dict[str, Any]
) -> float:
    if _explicit_offgrid_costs(offgrid_config) is not None:
        return 0.0
    return _cost_value(costs, "oil", ["fuel"], default=0.0)


def _ensure_sector_oil_bus(
    n: pypsa.Network, costs: pd.DataFrame, offgrid_config: dict[str, Any]
) -> str:
    if "Earth oil" in n.buses.index:
        return "Earth oil"

    if "oil" not in n.carriers.index:
        n.add("Carrier", "oil", co2_emissions=0.0)
    elif "co2_emissions" in n.carriers.columns:
        n.carriers.loc["oil", "co2_emissions"] = 0.0

    n.add("Bus", "Earth oil", carrier="oil", location="Earth")
    n.add(
        "Store",
        "Earth oil Store",
        bus="Earth oil",
        e_nom_extendable=True,
        e_cyclic=True,
        carrier="oil",
    )
    n.add(
        "Generator",
        "Earth oil",
        bus="Earth oil",
        p_nom_extendable=True,
        carrier="oil",
        marginal_cost=_sector_oil_supply_marginal_cost(costs, offgrid_config),
    )

    return "Earth oil"


def _sector_oil_buses(
    n: pypsa.Network,
    costs: pd.DataFrame,
    parent_buses: pd.Index,
    offgrid_config: dict[str, Any],
):
    oil_buses = []
    fallback_oil_bus = None
    bus_locations = n.buses.get("location", pd.Series("", index=n.buses.index))

    for bus in parent_buses:
        candidates = [
            f"{bus} oil",
            n.buses.index[
                (n.buses.carrier.fillna("") == "oil")
                & (bus_locations.fillna("") == bus)
            ],
        ]

        oil_bus = None
        for candidate in candidates:
            if isinstance(candidate, pd.Index):
                if not candidate.empty:
                    oil_bus = candidate[0]
                    break
            elif candidate in n.buses.index:
                oil_bus = candidate
                break

        if oil_bus is None:
            fallback_oil_bus = fallback_oil_bus or _ensure_sector_oil_bus(
                n, costs, offgrid_config
            )
            oil_bus = fallback_oil_bus

        oil_buses.append(oil_bus)

    return pd.Index(oil_buses)


def _ensure_sector_oil_supply(
    n: pypsa.Network,
    costs: pd.DataFrame,
    oil_buses: pd.Index,
    offgrid_config: dict[str, Any],
) -> None:
    if "oil" not in n.carriers.index:
        n.add("Carrier", "oil", co2_emissions=0.0)
    elif "co2_emissions" in n.carriers.columns:
        n.carriers.loc["oil", "co2_emissions"] = 0.0

    for bus in pd.Index(oil_buses).unique():
        has_store = (
            not n.stores.empty
            and "bus" in n.stores.columns
            and ((n.stores.bus == bus) & (n.stores.carrier == "oil")).any()
        )
        if not has_store:
            store_name = f"{bus} Store"
            if store_name not in n.stores.index:
                n.add(
                    "Store",
                    store_name,
                    bus=bus,
                    e_nom_extendable=True,
                    e_cyclic=True,
                    carrier="oil",
                )

        has_generator = (
            not n.generators.empty
            and "bus" in n.generators.columns
            and ((n.generators.bus == bus) & (n.generators.carrier == "oil")).any()
        )
        if not has_generator:
            generator_name = bus if bus not in n.generators.index else f"{bus} fuel"
            n.add(
                "Generator",
                generator_name,
                bus=bus,
                p_nom_extendable=True,
                carrier="oil",
                marginal_cost=_sector_oil_supply_marginal_cost(costs, offgrid_config),
            )


def _attach_offgrid_oil_links(
    n: pypsa.Network,
    costs: pd.DataFrame,
    parent_buses: pd.Index,
    offgrid_buses: pd.Index,
    offgrid_config: dict[str, Any],
) -> None:
    link_names = parent_buses + OFFGRID_OIL_SUFFIX
    efficiency = _efficiency(costs, "oil")
    oil_buses = _sector_oil_buses(n, costs, parent_buses, offgrid_config)
    _ensure_sector_oil_supply(n, costs, oil_buses, offgrid_config)

    kwargs = {}
    if "co2 atmosphere" in n.buses.index:
        kwargs["bus2"] = "co2 atmosphere"
        kwargs["efficiency2"] = _co2_intensity(costs, "oil")
    else:
        logger.warning(
            "No 'co2 atmosphere' bus found. Offgrid oil links will not emit to a CO2 bus."
        )

    n.madd(
        "Link",
        link_names,
        bus0=oil_buses,
        bus1=offgrid_buses,
        carrier=OFFGRID_OIL_CARRIER,
        p_nom_extendable=True,
        efficiency=efficiency,
        capital_cost=efficiency
        * _offgrid_capital_cost(costs, "oil", offgrid_config),
        marginal_cost=efficiency
        * _offgrid_marginal_cost(costs, "oil", offgrid_config),
        lifetime=_offgrid_lifetime(costs, "oil", offgrid_config),
        **kwargs,
    )


def add_offgrid_components(
    n: pypsa.Network,
    costs: pd.DataFrame,
    offgrid_config: dict[str, Any],
    *,
    scope: str = "electricity",
    electricity: dict[str, Any] | None = None,
    storage_techs: dict[str, Any] | None = None,
    solving: dict[str, Any] | None = None,
    co2: dict[str, Any] | None = None,
    opts: str | None = None,
    access_rate_file: str | Path | bool | None = None,
    base_demand_file: str | Path | bool | None = None,
    busmap_file: str | Path | bool | None = None,
    simplify_busmap_file: str | Path | bool | None = None,
    planning_horizon: str | int | None = None,
) -> pypsa.Network:
    """Add isolated offgrid systems and endogenous profile-share demand split."""
    offgrid_config = offgrid_config or {}
    if _is_unset(offgrid_config.get("enable", False)):
        logger.info("Offgrid components disabled.")
        return n

    load_shedding = (solving or {}).get("options", {}).get("load_shedding")
    if not load_shedding:
        logger.warning(
            "offgrid.enable is true, but solving.options.load_shedding is disabled. "
            "Load shedding will not be available on isolated offgrid buses."
        )

    sector_scope = _normalise_scope(scope) == "sector"
    parent_buses = get_ac_buses(n)
    if parent_buses.empty:
        raise ValueError("No AC buses found for offgrid component placement.")

    offgrid_buses = parent_buses + OFFGRID_BUS_SUFFIX
    load_to_parent = _load_parent_map(n, parent_buses, sector_scope)
    served_load = aggregate_served_load_by_bus(n, parent_buses, load_to_parent)

    bus_countries = n.buses.get("country", pd.Series("", index=n.buses.index))
    access_file = access_rate_file
    if access_file is None:
        access_file = offgrid_config.get("access_rate_file", False)

    access_rates = load_access_rate_data(
        access_file, parent_buses, bus_countries, planning_horizon=planning_horizon
    )
    min_access_rate = offgrid_config.get("min_access_rate", 0.05)
    total_demand = calculate_total_demand(
        served_load, access_rates, min_access_rate=min_access_rate
    )
    base_demand = load_base_demand_data(
        base_demand_file,
        n.snapshots,
        parent_buses,
        busmap_file=busmap_file,
        simplify_busmap_file=simplify_busmap_file,
    )
    competitive_demand = calculate_competitive_demand(total_demand, base_demand)
    demand_multipliers = _demand_multipliers(
        access_rates, parent_buses, min_access_rate=min_access_rate
    )

    _scale_loads_to_total_demand(n, load_to_parent, demand_multipliers)
    if not _is_unset(offgrid_config.get("demand_scaling_only", False)):
        logger.info(
            "Applied access-rate demand scaling only; skipping offgrid component "
            "attachment."
        )
        return n

    relief_buses = _relief_buses_by_parent(n, parent_buses, load_to_parent)

    _add_offgrid_carriers(n, costs, sector_scope)
    _attach_offgrid_buses(n, parent_buses, offgrid_buses)
    _attach_offgrid_demand_split_generators(
        n, parent_buses, offgrid_buses, relief_buses, competitive_demand
    )
    _attach_offgrid_solar(
        n,
        costs,
        parent_buses,
        offgrid_buses,
        offgrid_config,
        source_carrier="solar",
    )
    _attach_offgrid_battery(
        n,
        costs,
        parent_buses,
        offgrid_buses,
        electricity=electricity or {},
        storage_techs=storage_techs or {},
        offgrid_config=offgrid_config,
    )

    if sector_scope:
        _attach_offgrid_oil_links(
            n, costs, parent_buses, offgrid_buses, offgrid_config
        )
    else:
        _attach_offgrid_oil_generators(
            n,
            costs,
            parent_buses,
            offgrid_buses,
            co2 or {},
            opts,
            offgrid_config,
        )

    weights = _snapshot_weights(total_demand.index, n.snapshot_weightings)
    logger.info(
        "Added isolated offgrid demand split for %s AC buses. "
        "Annual electric demand: %.2f TWh; competitive demand: %.2f TWh.",
        len(parent_buses),
        total_demand.multiply(weights, axis=0).sum().sum() / 1e6,
        competitive_demand.multiply(weights, axis=0).sum().sum() / 1e6,
    )

    return n


def _named_value(values: Any, key: str, default: Any = None) -> Any:
    if hasattr(values, key):
        return getattr(values, key)

    try:
        return values[key]
    except (AttributeError, KeyError, TypeError):
        return default


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_offgrid_components", simpl="", clusters=6, ll="copt", opts="Co2L"
        )

    configure_logging(snakemake)

    network_path = _named_value(snakemake.input, "network", snakemake.input[0])
    costs_path = _named_value(
        snakemake.input,
        "costs",
        _named_value(snakemake.input, "tech_costs", None),
    )
    output_path = _named_value(snakemake.output, "network", snakemake.output[0])
    access_rate_file = _named_value(snakemake.input, "access_rate_file", None)
    base_demand_file = _named_value(snakemake.input, "base_demand_profiles", None)
    busmap_file = _named_value(
        snakemake.input,
        "cluster_busmap",
        _named_value(snakemake.input, "busmap", None),
    )
    simplify_busmap_file = _named_value(snakemake.input, "simplify_busmap", None)

    n = pypsa.Network(network_path)
    costs = read_csv_nafix(costs_path, index_col=0)

    n = add_offgrid_components(
        n,
        costs,
        _named_value(snakemake.params, "offgrid", {}),
        scope=_named_value(snakemake.params, "scope", "electricity"),
        electricity=_named_value(snakemake.params, "electricity", {}),
        storage_techs=_named_value(snakemake.params, "storage_techs", {}),
        solving=_named_value(snakemake.params, "solving", {}),
        co2=_named_value(snakemake.params, "co2", {}),
        opts=_named_value(
            snakemake.params, "opts", getattr(snakemake.wildcards, "opts", None)
        ),
        access_rate_file=access_rate_file,
        base_demand_file=base_demand_file,
        busmap_file=busmap_file,
        simplify_busmap_file=simplify_busmap_file,
        planning_horizon=_named_value(
            snakemake.params,
            "planning_horizon",
            getattr(snakemake.wildcards, "planning_horizons", None),
        ),
    )

    sanitize_carriers(n, snakemake.config)
    sanitize_locations(n)
    n.export_to_netcdf(output_path)
