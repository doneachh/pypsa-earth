# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
from pathlib import Path

import pandas as pd
import pypsa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from add_offgrid_components import (
    add_offgrid_components,
    calculate_competitive_demand,
    calculate_total_demand,
    load_base_demand_data,
    load_access_rate_data,
)
from solve_network import add_offgrid_demand_split_constraints
from solve_network import add_offgrid_solar_potential_constraints


def cost_table(use_sector_costs=False):
    capital_column = "fixed" if use_sector_costs else "capital_cost"
    costs = pd.DataFrame(
        index=["solar", "battery", "battery inverter", "oil"],
        columns=[
            "capital_cost",
            "fixed",
            "marginal_cost",
            "VOM",
            "fuel",
            "efficiency",
            "lifetime",
            "co2_emissions",
            "CO2 intensity",
        ],
        data=0.0,
    )
    costs.loc["solar", capital_column] = 100.0
    costs.loc["solar", "efficiency"] = 1.0
    costs.loc["solar", "lifetime"] = 25
    costs.loc["battery", capital_column] = 50.0
    costs.loc["battery", "lifetime"] = 15
    costs.loc["battery inverter", "efficiency"] = 0.81
    costs.loc["oil", capital_column] = 20.0
    costs.loc["oil", "marginal_cost"] = 120.0
    costs.loc["oil", "VOM"] = 4.0
    costs.loc["oil", "fuel"] = 40.0
    costs.loc["oil", "efficiency"] = 0.4
    costs.loc["oil", "lifetime"] = 30
    costs.loc["oil", "co2_emissions"] = 0.25
    costs.loc["oil", "CO2 intensity"] = 0.25
    return costs


def toy_network():
    n = pypsa.Network()
    snapshots = pd.date_range("2013-01-01", periods=2, freq="h")
    n.set_snapshots(snapshots)

    for carrier in ["AC", "solar", "oil"]:
        n.add("Carrier", carrier)

    n.add("Bus", "A", carrier="AC", x=0.0, y=0.0, country="NG")
    n.add("Bus", "B", carrier="AC", x=1.0, y=1.0, country="BJ")
    n.add(
        "Load",
        "A load",
        bus="A",
        carrier="AC",
        p_set=pd.Series([10.0, 20.0], index=snapshots),
    )
    n.add(
        "Load",
        "B load",
        bus="B",
        carrier="AC",
        p_set=pd.Series([8.0, 8.0], index=snapshots),
    )
    n.add(
        "Generator",
        "A solar",
        bus="A",
        carrier="solar",
        p_nom_extendable=True,
        p_nom_max=100.0,
        p_max_pu=pd.Series([0.5, 0.0], index=snapshots),
    )
    n.add(
        "Generator",
        "B solar",
        bus="B",
        carrier="solar",
        p_nom_extendable=True,
        p_nom_max=100.0,
        p_max_pu=pd.Series([0.25, 0.0], index=snapshots),
    )
    return n


def offgrid_config(access_rate_file=False):
    return {
        "enable": True,
        "access_rate_file": str(access_rate_file) if access_rate_file else False,
        "min_access_rate": 0.05,
    }


def explicit_offgrid_costs():
    return {
        "solar_capex": 800,
        "battery_capex_kwh": 250,
        "battery_max_hours": 6,
        "diesel_capex": 400,
        "diesel_marginal": 300,
        "discount_rate": 0.071,
        "asset_lifetime": 20,
    }


def annuity(lifetime, discount_rate):
    if discount_rate == 0:
        return 1 / lifetime
    return discount_rate / (1 - 1 / (1 + discount_rate) ** lifetime)


def test_calculate_total_demand_from_access_rate():
    served_load = pd.DataFrame({"A": [10.0, 20.0], "B": [5.0, 5.0]})
    access_rates = pd.Series({"A": 0.5, "B": 0.8})

    result = calculate_total_demand(served_load, access_rates, min_access_rate=0.05)

    expected = pd.DataFrame({"A": [20.0, 40.0], "B": [6.25, 6.25]})
    pd.testing.assert_frame_equal(result, expected)


def test_calculate_competitive_demand_above_base_profile_only():
    total_demand = pd.DataFrame({"A": [20.0, 40.0], "B": [5.0, 5.0]})
    base_demand = pd.DataFrame({"A": [10.0, 30.0], "B": [8.0, 3.0]})

    result = calculate_competitive_demand(total_demand, base_demand)

    expected = pd.DataFrame({"A": [10.0, 10.0], "B": [0.0, 2.0]})
    pd.testing.assert_frame_equal(result, expected)


def test_load_base_demand_aggregates_through_busmap(tmp_path):
    base_demand_file = tmp_path / "base_demand.csv"
    busmap_file = tmp_path / "busmap.csv"
    base_demand_file.write_text(
        "time,0,1,2\n"
        "2013-01-01 00:00:00,1,2,4\n"
        "2013-01-01 01:00:00,3,4,8\n"
    )
    busmap_file.write_text("Bus,busmap\n0,A\n1,A\n2,B\n")
    snapshots = pd.date_range("2013-01-01", periods=2, freq="h")

    result = load_base_demand_data(
        base_demand_file, snapshots, pd.Index(["A", "B"]), busmap_file
    )

    expected = pd.DataFrame(
        {"A": [3.0, 7.0], "B": [4.0, 8.0]},
        index=snapshots,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_load_base_demand_aggregates_through_simplify_and_cluster_busmaps(tmp_path):
    base_demand_file = tmp_path / "base_demand.csv"
    simplify_busmap_file = tmp_path / "busmap_elec_s.csv"
    cluster_busmap_file = tmp_path / "busmap_elec_s_6.csv"
    base_demand_file.write_text(
        "time,0,1,2\n"
        "2013-01-01 00:00:00,1,2,4\n"
        "2013-01-01 01:00:00,3,4,8\n"
    )
    simplify_busmap_file.write_text(",0\n0,10.0\n1,10.0\n2,20.0\n")
    cluster_busmap_file.write_text("Bus,busmap\n10,A\n20,B\n")
    snapshots = pd.date_range("2013-01-01", periods=2, freq="h")

    result = load_base_demand_data(
        base_demand_file,
        snapshots,
        pd.Index(["A", "B"]),
        cluster_busmap_file,
        simplify_busmap_file,
    )

    expected = pd.DataFrame(
        {"A": [3.0, 7.0], "B": [4.0, 8.0]},
        index=snapshots,
    )
    pd.testing.assert_frame_equal(result, expected)


def test_load_base_demand_resamples_to_network_snapshots(tmp_path):
    base_demand_file = tmp_path / "base_demand.csv"
    busmap_file = tmp_path / "busmap.csv"
    hourly = pd.date_range("2013-01-01", periods=12, freq="h")
    pd.DataFrame(
        {"0": range(12)},
        index=hourly,
    ).rename_axis("time").to_csv(base_demand_file)
    busmap_file.write_text("Bus,busmap\n0,A\n")
    snapshots = pd.date_range("2013-01-01", periods=2, freq="6h")

    result = load_base_demand_data(
        base_demand_file, snapshots, pd.Index(["A"]), busmap_file
    )

    expected = pd.DataFrame({"A": [2.5, 8.5]}, index=snapshots)
    pd.testing.assert_frame_equal(result, expected)


def test_load_country_urban_rural_access_rates_with_share(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    access_rate_file.write_text(
        "country,access_rate_urban,access_rate_rural,urban_share\nNG,90,60,0.5\n"
    )
    buses = pd.Index(["A", "B"])
    bus_countries = pd.Series({"A": "NG", "B": "BJ"})

    result = load_access_rate_data(access_rate_file, buses, bus_countries)

    expected = pd.Series({"A": 0.75, "B": 1.0})
    pd.testing.assert_series_equal(result, expected)


def test_load_access_rates_filters_by_planning_horizon(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    access_rate_file.write_text(
        "year,country,access_rate_urban,access_rate_rural,urban_share\n"
        "2030,NG,80,20,0.5\n"
        "2040,NG,60,10,0.5\n"
    )
    buses = pd.Index(["A"])
    bus_countries = pd.Series({"A": "NG"})

    result = load_access_rate_data(
        access_rate_file, buses, bus_countries, planning_horizon=2040
    )

    expected = pd.Series({"A": 0.35})
    pd.testing.assert_series_equal(result, expected)


def test_add_offgrid_components_to_electricity_network(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    access_rate_file.write_text("bus,access_rate\nA,0.5\nB,1.0\n")
    n = toy_network()

    add_offgrid_components(
        n,
        cost_table(),
        offgrid_config(access_rate_file),
        electricity={"max_hours": {"battery": 6}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": 100}},
        scope="electricity",
    )

    assert {"A offgrid", "B offgrid"} <= set(n.buses.index)
    assert {"A offgrid solar", "B offgrid solar"} <= set(n.generators.index)
    assert {"A offgrid oil", "B offgrid oil"} <= set(n.generators.index)
    assert {"A offgrid battery", "B offgrid battery"} <= set(n.storage_units.index)
    assert {"A offgrid demand", "B offgrid demand"} <= set(n.generators.index)
    assert {"A offgrid demand relief", "B offgrid demand relief"} <= set(
        n.generators.index
    )

    assert "A local demand" not in n.buses.index
    assert "A grid connection" not in n.links.index
    assert "A offgrid supply" not in n.links.index

    assert n.loads.at["A load", "bus"] == "A"
    assert n.loads.at["B load", "bus"] == "B"
    pd.testing.assert_series_equal(
        n.loads_t.p_set["A load"],
        pd.Series([20.0, 40.0], index=n.snapshots, name="A load"),
    )
    pd.testing.assert_series_equal(
        n.loads_t.p_set["B load"],
        pd.Series([8.0, 8.0], index=n.snapshots, name="B load"),
    )

    assert n.generators.at["A offgrid demand relief", "bus"] == "A"
    assert n.generators.at["A offgrid demand relief", "sign"] == 1.0
    assert n.generators.at["A offgrid demand", "bus"] == "A offgrid"
    assert n.generators.at["A offgrid demand", "sign"] == -1.0
    assert n.generators.at["A offgrid demand", "p_nom_max"] == 40.0
    pd.testing.assert_series_equal(
        n.generators_t.p_max_pu["A offgrid demand"],
        pd.Series([0.5, 1.0], index=n.snapshots, name="A offgrid demand"),
    )
    assert n.generators.at["A offgrid oil", "carrier"] == "offgrid oil"
    assert n.carriers.at["offgrid oil", "co2_emissions"] == 0.25
    assert n.generators.at["A offgrid solar", "capital_cost"] == pytest.approx(100.0)
    assert n.generators.at["A offgrid solar", "p_nom_max"] == pytest.approx(100.0)
    assert n.storage_units.at["A offgrid battery", "capital_cost"] == pytest.approx(
        50.0
    )
    assert n.storage_units.at["A offgrid battery", "max_hours"] == pytest.approx(6.0)
    assert n.generators.at["A offgrid oil", "capital_cost"] == pytest.approx(20.0)
    assert n.generators.at["A offgrid oil", "marginal_cost"] == pytest.approx(120.0)


def test_demand_scaling_only_adds_no_offgrid_components(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    access_rate_file.write_text("bus,access_rate\nA,0.5\nB,1.0\n")
    n = toy_network()
    config = offgrid_config(access_rate_file)
    config["demand_scaling_only"] = True

    add_offgrid_components(
        n,
        cost_table(),
        config,
        electricity={"max_hours": {"battery": 6}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": 100}},
        scope="electricity",
    )

    assert "A offgrid" not in n.buses.index
    assert "A offgrid solar" not in n.generators.index
    assert "A offgrid demand" not in n.generators.index
    assert "A offgrid battery" not in n.storage_units.index
    pd.testing.assert_series_equal(
        n.loads_t.p_set["A load"],
        pd.Series([20.0, 40.0], index=n.snapshots, name="A load"),
    )
    pd.testing.assert_series_equal(
        n.loads_t.p_set["B load"],
        pd.Series([8.0, 8.0], index=n.snapshots, name="B load"),
    )


def test_base_demand_keeps_existing_profile_on_grid(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    base_demand_file = tmp_path / "base_demand.csv"
    busmap_file = tmp_path / "busmap.csv"
    access_rate_file.write_text("bus,access_rate\nA,0.5\nB,1.0\n")
    base_demand_file.write_text(
        "time,0,1,B\n"
        "2013-01-01 00:00:00,2,4,7\n"
        "2013-01-01 01:00:00,3,5,6\n"
    )
    busmap_file.write_text("Bus,busmap\n0,A\n1,A\n")
    n = toy_network()

    add_offgrid_components(
        n,
        cost_table(),
        offgrid_config(access_rate_file),
        electricity={"max_hours": {"battery": 6}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": 100}},
        scope="electricity",
        base_demand_file=base_demand_file,
        busmap_file=busmap_file,
    )

    pd.testing.assert_series_equal(
        n.loads_t.p_set["A load"],
        pd.Series([20.0, 40.0], index=n.snapshots, name="A load"),
    )
    assert n.generators.at["A offgrid demand", "p_nom_max"] == pytest.approx(32.0)
    pd.testing.assert_series_equal(
        n.generators_t.p_max_pu["A offgrid demand"],
        pd.Series([14.0 / 32.0, 1.0], index=n.snapshots, name="A offgrid demand"),
    )
    assert n.generators.at["B offgrid demand", "p_nom_max"] == pytest.approx(2.0)


def test_explicit_offgrid_costs_are_annualised_for_offgrid_components():
    n = toy_network()
    config = offgrid_config()
    config["costs"] = explicit_offgrid_costs()

    add_offgrid_components(
        n,
        cost_table(),
        config,
        electricity={"max_hours": {"battery": 6}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": 100}},
        scope="electricity",
    )

    annualisation = annuity(20, 0.071)
    assert n.generators.at["A offgrid solar", "capital_cost"] == pytest.approx(
        800 * 1e3 * annualisation
    )
    assert n.storage_units.at["A offgrid battery", "capital_cost"] == pytest.approx(
        250 * 1e3 * 6 * annualisation
    )
    assert n.storage_units.at["A offgrid battery", "max_hours"] == pytest.approx(6.0)
    assert n.generators.at["A offgrid oil", "capital_cost"] == pytest.approx(
        400 * 1e3 * annualisation
    )
    assert n.generators.at["A offgrid oil", "marginal_cost"] == pytest.approx(300.0)


def test_add_offgrid_oil_as_sector_link(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    access_rate_file.write_text("bus,access_rate\nA,0.5\nB,1.0\n")
    n = toy_network()
    n.add("Bus", "A low voltage", carrier="low voltage", location="A")
    n.add("Bus", "B low voltage", carrier="low voltage", location="B")
    n.loads.loc["A load", "bus"] = "A low voltage"
    n.loads.loc["B load", "bus"] = "B low voltage"
    n.add("Load", "A heat", bus="A low voltage", carrier="heat", p_set=3.0)
    n.add("Bus", "A oil", carrier="oil", location="A")
    n.add("Bus", "B oil", carrier="oil", location="B")
    n.add("Bus", "co2 atmosphere", carrier="co2")

    add_offgrid_components(
        n,
        cost_table(use_sector_costs=True),
        offgrid_config(access_rate_file),
        electricity={"max_hours": {"battery": 6}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": 100}},
        scope="sector",
    )

    assert n.loads.at["A load", "bus"] == "A low voltage"
    assert n.loads.at["A heat", "bus"] == "A low voltage"
    assert n.generators.at["A offgrid demand relief", "bus"] == "A low voltage"
    assert "A grid connection" not in n.links.index
    assert "A offgrid supply" not in n.links.index
    assert "A offgrid oil" in n.links.index
    assert "A offgrid oil" not in n.generators.index
    assert n.links.at["A offgrid oil", "bus0"] == "A oil"
    assert n.links.at["A offgrid oil", "bus1"] == "A offgrid"
    assert n.links.at["A offgrid oil", "bus2"] == "co2 atmosphere"
    assert n.links.at["A offgrid oil", "efficiency2"] == 0.25
    assert "A oil Store" in n.stores.index
    assert "A oil" in n.generators.index
    assert "Earth oil" not in n.buses.index


def test_explicit_sector_diesel_cost_is_all_in_at_link(tmp_path):
    access_rate_file = tmp_path / "access_rates.csv"
    access_rate_file.write_text("bus,access_rate\nA,0.5\nB,1.0\n")
    n = toy_network()
    n.add("Bus", "A low voltage", carrier="low voltage", location="A")
    n.add("Bus", "B low voltage", carrier="low voltage", location="B")
    n.loads.loc["A load", "bus"] = "A low voltage"
    n.loads.loc["B load", "bus"] = "B low voltage"
    n.add("Bus", "A oil", carrier="oil", location="A")
    n.add("Bus", "B oil", carrier="oil", location="B")
    n.add("Bus", "co2 atmosphere", carrier="co2")
    config = offgrid_config(access_rate_file)
    config["costs"] = explicit_offgrid_costs()

    add_offgrid_components(
        n,
        cost_table(use_sector_costs=True),
        config,
        electricity={"max_hours": {"battery": 4}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": 100}},
        scope="sector",
    )

    annualisation = annuity(20, 0.071)
    efficiency = cost_table(use_sector_costs=True).at["oil", "efficiency"]
    assert n.links.at["A offgrid oil", "capital_cost"] == pytest.approx(
        efficiency * 400 * 1e3 * annualisation
    )
    assert n.links.at["A offgrid oil", "marginal_cost"] == pytest.approx(
        efficiency * 300
    )
    assert n.generators.at["A oil", "marginal_cost"] == pytest.approx(0.0)


def single_bus_choice_network():
    n = pypsa.Network()
    snapshots = pd.date_range("2013-01-01", periods=2, freq="h")
    n.set_snapshots(snapshots)

    for carrier in ["AC", "solar", "oil", "grid supply"]:
        n.add("Carrier", carrier)

    n.add("Bus", "A", carrier="AC", x=0.0, y=0.0, country="NG")
    n.add(
        "Load",
        "A load",
        bus="A",
        carrier="AC",
        p_set=pd.Series([10.0, 10.0], index=snapshots),
    )
    n.add(
        "Generator",
        "solar profile",
        bus="A",
        carrier="solar",
        p_nom=0.0,
        p_nom_max=20.0,
        p_max_pu=pd.Series([1.0, 1.0], index=snapshots),
    )
    n.add(
        "Generator",
        "grid supply",
        bus="A",
        carrier="grid supply",
        p_nom_extendable=True,
        capital_cost=0.0,
        marginal_cost=10.0,
    )
    return n


def solve_reference():
    n = single_bus_choice_network()
    status, condition = n.optimize(solver_name="highs")
    assert status == "ok"
    assert condition == "optimal"
    return n.objective


def solve_demand_split(offgrid_solar_capital_cost):
    pytest.importorskip("highspy")

    costs = cost_table()
    costs.loc["solar", "capital_cost"] = offgrid_solar_capital_cost
    costs.loc["battery", "capital_cost"] = 1000.0
    costs.loc["oil", "capital_cost"] = 1000.0
    costs.loc["oil", "marginal_cost"] = 500.0
    n = single_bus_choice_network()

    add_offgrid_components(
        n,
        costs,
        offgrid_config(),
        electricity={"max_hours": {"battery": 1}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": False}},
        scope="electricity",
    )

    def extra_functionality(n, snapshots):
        add_offgrid_demand_split_constraints(n, snapshots)
        add_offgrid_solar_potential_constraints(n, snapshots)

    status, condition = n.optimize(
        solver_name="highs", extra_functionality=extra_functionality
    )
    assert status == "ok"
    assert condition == "optimal"

    peak = n.generators.at["A offgrid demand", "p_nom_max"]
    alpha = n.generators.at["A offgrid demand", "p_nom_opt"] / peak
    return alpha, n.objective


def test_optimisation_switches_between_grid_and_isolated_offgrid():
    reference_objective = solve_reference()

    alpha, objective = solve_demand_split(offgrid_solar_capital_cost=1000.0)
    assert alpha == pytest.approx(0.0, abs=1e-6)
    assert objective == pytest.approx(reference_objective, abs=1e-6)

    alpha, objective = solve_demand_split(offgrid_solar_capital_cost=1.0)
    assert alpha == pytest.approx(1.0, abs=1e-6)
    assert objective < reference_objective


def test_shared_solar_potential_limits_grid_and_offgrid_solar():
    pytest.importorskip("highspy")
    n = single_bus_choice_network()
    n.generators.loc["solar profile", "p_nom_extendable"] = True
    n.generators.loc["solar profile", "p_nom_max"] = 5.0
    n.generators.loc["solar profile", "capital_cost"] = 0.0
    n.generators.loc["grid supply", "marginal_cost"] = 100.0
    costs = cost_table()
    costs.loc["solar", "capital_cost"] = 0.0
    costs.loc["battery", "capital_cost"] = 1000.0
    costs.loc["oil", "capital_cost"] = 1000.0
    costs.loc["oil", "marginal_cost"] = 1000.0

    add_offgrid_components(
        n,
        costs,
        offgrid_config(),
        electricity={"max_hours": {"battery": 1}},
        storage_techs={"battery": {"bicharger": "battery inverter"}},
        solving={"options": {"load_shedding": False}},
        scope="electricity",
    )

    def extra_functionality(n, snapshots):
        add_offgrid_demand_split_constraints(n, snapshots)
        add_offgrid_solar_potential_constraints(n, snapshots)

    status, condition = n.optimize(
        solver_name="highs", extra_functionality=extra_functionality
    )
    assert status == "ok"
    assert condition == "optimal"

    shared_solar = (
        n.generators.at["solar profile", "p_nom_opt"]
        + n.generators.at["A offgrid solar", "p_nom_opt"]
    )
    assert shared_solar <= 5.0 + 1e-6
