
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
import re
import yaml
import pandas as pd
idx_slice = pd.IndexSlice
from itertools import product
import pypsa
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Comment out for debugging and development
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Comment out for debugging and development

from plot_helpers import (
    chdir_to_root_dir,
    to_csv_nafix,
)

chdir_to_root_dir()

#%%

run_name_prefix = "scenarios_H2G" # Experiment name

sdir = Path.cwd() / "results"/ f"{run_name_prefix}_summary_v2"
sdir.mkdir(exist_ok=True, parents=True)

run_names = [
    "H2G_A_CD_2035",
    "H2G_A_CD_2050",
    # "H2G_A_EG_2035", 
    # "H2G_A_EG_2050", 
    # "H2G_A_ET_2035",
    # "H2G_A_ET_2050",
    # "H2G_A_GH_2035",
    # "H2G_A_GH_2050",
    # "H2G_A_KE_2035",
    # "H2G_A_KE_2050",
    # "H2G_A_MA_2035",
    # "H2G_A_MA_2050",
    # "H2G_A_NA_2035",
    # "H2G_A_NA_2050",
    # "H2G_A_NG_2035",
    # "H2G_A_NG_2050",
    # "H2G_A_TN_2035",
    # "H2G_A_TN_2050",
    # "H2G_A_TZ_2035",
    # "H2G_A_TZ_2050",
    # "H2G_A_ZA_2035", 
    # "H2G_A_ZA_2050"
]

#%%

def extract_wildcards_from_filename(filename):
    """Extract scenario parameters from network filename using regex.
    
    Expected format: elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc
    Example: elec_s_195_ec_lc1.0_Co2L_3H_2030_0.071_AP_428export.nc
    
    Using the Snakemake wildcard constraints of PyPSA-Earth for the pattern matching.
    """
    pattern = (
        r"elec_s([a-zA-Z0-9]*)_"
        r"([0-9]+(?:m|flex)?|all|min)_"
        r"ec_l([vcl](?:[0-9.]+|opt|all)|all)_"
        r"([^_]*)_"
        r"([a-zA-Z0-9+\-.]+)_"
        r"(20[2-9][0-9]|2100)_"
        r"([0-9.]+)_"
        r"([a-zA-Z0-9+\-.]+)_"
        r"([0-9]+(?:\.[0-9]+)?)export\.nc"
    )
    match = re.search(pattern, filename)
    if match:
        simpl, clusters, ll, opts, sopts, year, discountrate, demand, h2export = match.groups()
        return {
            "simpl": simpl if simpl else "",
            "clusters": clusters,
            "ll": ll,
            "opts": opts if opts else "",
            "sopts": sopts,
            "year": int(year),
            "discountrate": float(discountrate),
            "demand": demand,
            "h2export": float(h2export),
        }
    return None

def collect_files_from_directories(all_postnetworks_dir):
    """
    Collects all existing files from the directories specified in all_postnetworks_dir.
    """
    files_in_folder = {}
    for run_name, path in all_postnetworks_dir.items():
        if path.exists() and path.is_dir():
            files_in_folder[f"{run_name}"] = list(path.glob('*'))
        else:
            files_in_folder[f"{run_name}"] = []

    # Print the collected files for each country
    for key, files in files_in_folder.items():
        print(f"Files in {key}:")
        for file in files:
            print(f"  {file}")

    return files_in_folder

def init_stats_dict(network_files, keys, name):
    
    stats_dict = {
        key: pd.concat([pd.DataFrame(index=network_files.index)]) #, keys=[key], names=[name]
        for key in keys}

    return stats_dict

def save_stats_dict(stats_dict, stats_name, path_dir):
    for key, df in stats_dict.items():
        to_csv_nafix(df, path_dir / f"{stats_name}_{key}.csv")
        print(f"Saved {key} to {path_dir / f'{stats_name}_{key}.csv'}")

# NOTE: Config file names must be based on the run name in a folder with run_name_prefix
configs_dict = {
    run_name: yaml.safe_load(
        Path(f"configs/{run_name_prefix}/config.{run_name}.yaml").read_text()
    )
    for run_name in run_names
}

postnetworks_dict = {
    run_name: Path.cwd() / configs_dict[run_name]["results_dir"] / f"{run_name}" / "postnetworks"
    for run_name in run_names
}

files_in_folder_dict = collect_files_from_directories(postnetworks_dict)

# Build nc_files DataFrame by parsing filenames directly
cols = ["run_name_prefix", "run_name", "country", "year", "simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand", "h2export"]
nc_files_data = []

for run_name in run_names:
    config = configs_dict[run_name]
    countries = config["countries"]
    
    # NOTE: The following code assumes only one country is selected per run
    if len(countries) > 1:
        raise ValueError(f"Run {run_name} has multiple countries: {countries}.")
    country = countries[0] if countries else "unknown"
    
    for file in files_in_folder_dict.get(run_name, []):
        wcs = extract_wildcards_from_filename(file.name)
        if wcs:
            nc_files_data.append({
                "run_name_prefix": run_name_prefix,
                "run_name": run_name,
                "country": country,
                "file": file,
                **wcs,
            })
            print(f"Adding nc file for {run_name}: {file.name}")

nc_files = pd.DataFrame(nc_files_data).set_index(cols) if nc_files_data else pd.DataFrame(columns=cols + ["file"]).set_index(cols)

if nc_files.empty:
    raise ValueError("No files found for the given run names and wildcards. Please check the configurations and file names.")

#%%

# initialise dicts per metric (market balance, optimal capacities, costs, marginal prices) with dataframes per bus_carrier or other groups

balance_dict = init_stats_dict(nc_files, keys=[
    "AC", "H2", "oil", "gas", "co2 stored", "co2", "biogas",
    # , "methanol", "NH3", "steel", "HBI" "freshwater" "solid biomass",
    ], name="bus_carrier")

optimal_capacity_dict = init_stats_dict(nc_files, keys=["AC", "H2"], name="bus_carrier") #, "methanol", "NH3", "steel", "HBI"

costs_dict = init_stats_dict(nc_files, keys=["capex", "opex"], name="costs")

load_avg_marginal_price = pd.DataFrame(index=nc_files.index, columns=["H2 export bus"]) #"H2 export", , "FT export", "NH3 export"
load_avg_marginal_price.columns.name = "bus" # NB: this is spatially resolved.

pypsa.options.params.statistics.nice_names = False
pypsa.options.params.statistics.drop_zero = False
pypsa.options.params.statistics.round = 6

for nc_files_idx in nc_files.index:
    
    n = pypsa.Network(nc_files.at[nc_files_idx,"file"])

    # energy balance per bus_carrier in TWh
    for bus_carrier in balance_dict.keys():

        ds = (
            n.stats.energy_balance(
                bus_carrier=bus_carrier, 
                groupby="carrier",
                aggregate_across_components=True
            )
            .div(1e6)
            .round(1)
        )

        balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

        if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():

            ds = (
                n.stats.energy_balance(
                    bus_carrier="low voltage", 
                    groupby="carrier",                     
                    aggregate_across_components=True
                )
                .div(1e6)
                .round(1)
            )

            balance_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

            balance_dict[bus_carrier] = balance_dict[bus_carrier].drop("electricity distribution grid", axis=1)  

    # optimal capacities (installed + expanded) per bus_carrier in GW_output_unit
    for bus_carrier in optimal_capacity_dict.keys():

        ds = (
            n.stats.optimal_capacity(
                components=["Generator", "Link", "StorageUnit"], 
                bus_carrier=bus_carrier, 
                groupby="carrier",
                aggregate_across_components=True
            )
            .loc[lambda x: x > 0]
            .div(1e3)
            .round(1)
        )

        # if bus_carrier == "H2":
        #     ds = ds/n.links.groupby("carrier").mean(numeric_only=True).efficiency
          
        optimal_capacity_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

        if bus_carrier == "AC" and "low voltage" in n.buses.carrier.unique():

            ds = (
                n.stats.optimal_capacity(
                    components=["Generator", "Link", "StorageUnit"], 
                    bus_carrier="low voltage", 
                    groupby="carrier",
                    aggregate_across_components=True
                )
                .loc[lambda x: x > 0]
                .div(1e3)
                .round(1)
            )
            
            if not ds.empty:
                optimal_capacity_dict[bus_carrier].loc[nc_files_idx, ds.index] = ds.values

                optimal_capacity_dict[bus_carrier] = optimal_capacity_dict[bus_carrier].drop("electricity distribution grid", axis=1)

    # system capex per carrier in billion currency unit
    ds = n.stats.capex().dropna().groupby("carrier").sum().div(1e9).round(4)

    costs_dict["capex"].loc[nc_files_idx, ds.index] = ds.values

    # system opex per carrier in billion currency unit
    ds = n.stats.opex().dropna().groupby("carrier").sum().div(1e9).round(4)

    costs_dict["opex"].loc[nc_files_idx, ds.index] = ds.values
    
    # ASSUMPTIONS: assume marginal costs of last unit can be earned as export price for all export
    # adding those revenues for export as negative opex costs
    # TODO: decide how to integrate revenues from exports in the costs_dict
    # if bus_carrier + " export" in n.loads_t.p.columns.unique():
    #     H2_export_price = n.buses_t.marginal_price["H2 export"].mean() # NB: hourly pattern is interesting!
    #     costs_dict["opex"].at[nc_files_idx,"H2 export"] = -(
    #         n.loads_t.p_set["H2 export"].mul(n.buses_t.marginal_price["H2 export"]).sum()/1e9
    #     )

    # load averaged marginal prices per bus_carrier in currency/MWh

    bus_carriers_to_price = ["H2", "NH3", "FT", "HBI", "steel", "STEEL", "industry methanol", "shipping methanol", "MEOH"]
    
    prices_load_weighted = n.stats.prices(groupby_time=True, weighting="load")
    
    # Filter and assign prices for buses matching carrier names (with or without " export" suffix)
    for bus_carrier in bus_carriers_to_price:
        matching_buses = n.buses.index[n.buses.index.str.contains(bus_carrier)]
        for bus in matching_buses:
            if bus in prices_load_weighted.index:
                load_avg_marginal_price.at[nc_files_idx, bus] = prices_load_weighted[bus]


# %%
to_csv_nafix(nc_files, sdir / "nc_files.csv")

save_stats_dict(balance_dict, "balance_dict", sdir)
save_stats_dict(optimal_capacity_dict, "optimal_capacity_dict", sdir)
save_stats_dict(costs_dict, "costs_dict", sdir)

to_csv_nafix(load_avg_marginal_price, sdir / "load_avg_marginal_price.csv")
print(f"Saved load_avg_marginal_price to {sdir / 'load_avg_marginal_price.csv'}")

# %%

h2_buses = n.buses[n.buses.index.str.endswith("H2")].index
h2_prices_per_t = n.buses_t.marginal_price[h2_buses] * 33.3333 / 1000
h2_prices_per_t.round(0).plot(legend=False, title="H2 €/kg", drawstyle='steps')
plt.show()
# %%
