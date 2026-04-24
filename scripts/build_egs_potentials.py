# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText:  PyPSA-Earth and PyPSA-Eur Authors
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Build EGS overlap, potentials, and optional capacity-factor profiles for PyPSA-Earth.
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Polygon
from _helpers import (
    configure_logging,
    create_logger,
    mock_snakemake,
    two_2_three_digits_country,
)

logger = create_logger(__name__)

HEAT_PER_GW_EL_FACTOR = 2347.0
CRF = 0.09 # calculated with 0.08 interest rate
CAPEX_TO_OPEX = 0.02 #assumtion 2% of Capex = Opex
HOURS = 8760.0


def read_network_regions(network_regions_file):
    """
    Read PyPSA-Earth onshore regions and use the region name as index.
    """
    regions = gpd.read_file(network_regions_file)

    if 'name' not in regions.columns:
        raise ValueError(
            f"Network regions file '{network_regions_file}' does not contain a 'name' column."
        )

    regions = regions.set_index('name', drop=True).set_crs(epsg=4326, allow_override=True)
    return regions



def prepare_egs_data(egs_file, countries, network_regions_file):
    """
    Build a region-aggregated PyPSA-compatible EGS potential table from a custom CSV input.

    Expected input columns:
    lon, lat, gid1,
    Pout_Gringarten_MW, LCOE_Gringarten_Eurct_per_kWh,
    Pout_VolumeMethod_MW, LCOE_VolumeMethod_Eurct_per_kWh,
    Pout_Sustainable_MW, LCOE_Sustainable_Eurct_per_kWh
    """
    df = pd.read_csv(egs_file)

    required_columns = {
        'lon',
        'lat',
        'gid1',
        'Pout_Gringarten_MW',
        'LCOE_Gringarten_Eurct_per_kWh',
        'Pout_VolumeMethod_MW',
        'LCOE_VolumeMethod_Eurct_per_kWh',
        'Pout_Sustainable_MW',
        'LCOE_Sustainable_Eurct_per_kWh',
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required input columns: {missing_columns}")

    countries = two_2_three_digits_country(countries)
    
    if isinstance(countries, str):
        countries = [countries]

    countries = [str(c).upper() for c in countries]
    
    df = df.drop(
        columns=[
            'Pout_VolumeMethod_MW',
            'LCOE_VolumeMethod_Eurct_per_kWh',
            'Pout_Sustainable_MW',
            'LCOE_Sustainable_Eurct_per_kWh',
        ]
    )

    df['gid1'] = df['gid1'].astype(str).str.strip().str.upper()
    df['country_code'] = df['gid1'].str.extract(r'^([A-Z]{3})', expand=False)
    df = df.loc[df['country_code'].isin(countries)].copy()

    if df.empty:
        raise ValueError(
            f"No EGS rows left after filtering by configured countries: {countries}"
        )

    df = df.rename(
        columns={
            'lon': 'Lon',
            'lat': 'Lat',
            'Pout_Gringarten_MW': 'PowerSust_MW',
            'LCOE_Gringarten_Eurct_per_kWh': 'LCOE_Eur_per_kWh',
        }
    )

    df = df.dropna(subset=['Lon', 'Lat', 'PowerSust_MW', 'LCOE_Eur_per_kWh'])
    df = df.loc[df['PowerSust_MW'] > 0.0].copy()

    if df.empty:
        raise ValueError('No valid EGS rows left after dropping invalid or zero-output rows.')

    # EUR/kWh =>  EUR/MWh
    df['LCOE_EUR_per_MWh'] = df['LCOE_Eur_per_kWh'] * 1000.0

    # Energy produced per year in MWh
    df['Leistung_MWh'] = df['PowerSust_MW'] * HOURS

    # Annualized CAPEX-like quantity, then converted to EUR/GW
    capex = (df['LCOE_EUR_per_MWh'] * df['Leistung_MWh']) / (CRF + CAPEX_TO_OPEX)

    df['PowerSust'] = df['PowerSust_MW'] / 1000.0  # MW -> GW
    df['HeatSust'] = df['PowerSust'] * HEAT_PER_GW_EL_FACTOR
    df['CAPEX'] = capex / df['PowerSust']  # EUR/GW
   

    point_gdf = gpd.GeoDataFrame(
        df[['gid1', 'country_code', 'Lon', 'Lat', 'CAPEX', 'PowerSust', 'HeatSust']],
        geometry=gpd.points_from_xy(df['Lon'], df['Lat']),
        crs='EPSG:4326',
    )

    regions = read_network_regions(network_regions_file).reset_index()[['name', 'geometry']]

    # Assign each point directly to a network region
    joined = gpd.sjoin(
        point_gdf,
        regions,
        how='left',
        predicate='intersects',
    ).rename(columns={'name': 'region'})

    missing_regions = int(joined['region'].isna().sum())
    if missing_regions > 0:
        logger.warning(
            f"{missing_regions} EGS points could not be assigned to any network region and will be dropped."
        )

    joined = joined.dropna(subset=['region']).copy()

    if joined.empty:
        raise ValueError(
            'No EGS points could be assigned to the provided network regions.'
        )

    region_agg = (
        joined.groupby('region', as_index=True)
        .agg(
            CAPEX=('CAPEX', 'mean'),
            PowerSust=('PowerSust', 'sum'),
            HeatSust=('HeatSust', 'sum'),
            n_points=('region', 'size'),
        )
        .sort_index()
    )

    region_agg.index.name = 'region'
    region_agg['p_nom_max'] = region_agg['PowerSust']

    return region_agg





def get_capacity_factors(network_regions_file, air_temperatures_file, snapshots=None):
    """
    Performance of EGS is higher for lower temperatures, due to more efficient
    air cooling Data from Ricks et al.: The Role of Flexible Geothermal Power
    in Decarbonized Elec Systems.
   
    Replicates the PyPSA-Eur temperature-based EGS capacity factor adjustment.
    """

    # these values are taken from the paper's
    # Supplementary Figure 20 from https://zenodo.org/records/7093330
    # and relate deviations of the ambient temperature from the year-average
    # ambient temperature to EGS capacity factors.
    delta_t = [-15, -10, -5, 0, 5, 10, 15, 20]
    cf = [1.17, 1.13, 1.07, 1.0, 0.925, 0.84, 0.75, 0.65]

    x = np.linspace(-15, 20, 200)
    y = np.interp(x, delta_t, cf)

    upper_x = np.linspace(20, 25, 50)
    m_upper = (y[-1] - y[-2]) / (x[-1] - x[-2])
    upper_y = upper_x * m_upper - x[-1] * m_upper + y[-1]

    lower_x = np.linspace(-20, -15, 50)
    m_lower = (y[1] - y[0]) / (x[1] - x[0])
    lower_y = lower_x * m_lower - x[0] * m_lower + y[0]

    x = np.hstack((lower_x, x, upper_x))
    y = np.hstack((lower_y, y, upper_y))

    network_regions = gpd.read_file(network_regions_file).set_crs(epsg=4326, allow_override=True)
    region_names = network_regions["name"]

    air_temp = xr.open_dataset(air_temperatures_file)
    if snapshots is None:
        snapshots = pd.DatetimeIndex(air_temp.indexes["time"])

    capacity_factors = pd.DataFrame(index=snapshots)

    for bus in region_names:
        temp = air_temp.sel(name=bus).to_dataframe()["temperature"]
        temp = temp.reindex(snapshots)
        capacity_factors[bus] = np.interp((temp - temp.mean()).values, x, y)

    capacity_factors.index.name = "snapshot"
    return capacity_factors

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "build_egs_potentials",
        )

    configure_logging(snakemake)

    #egs_config = snakemake.params.sector["enhanced_geothermal"]
    #sustainability_factor = egs_config.get("sustainability_factor", 0.0025)

    egs_data = prepare_egs_data(
        egs_file=snakemake.input.egs_input,
        countries=snakemake.params.countries,
        network_regions_file=snakemake.input.regions,
    )



    egs_data["p_nom_max"] = egs_data["PowerSust"] #/ sustainability_factor

    egs_data[[ "HeatSust", "p_nom_max", "CAPEX"]].to_csv(
        snakemake.output.egs_potentials
    )

    if hasattr(snakemake.input, "temp_air_total") and hasattr(
    snakemake.output, "egs_capacity_factors"
    ):
        snapshots = pd.date_range(
            freq="h",
            **snakemake.params.snapshots,
        )
        capacity_factors = get_capacity_factors(
            snakemake.input.regions,
            snakemake.input.temp_air_total,
            snapshots,
        )
        capacity_factors.to_csv(snakemake.output.egs_capacity_factors)
