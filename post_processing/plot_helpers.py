import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from pathlib import Path
import os
import sys
from typing import Callable, Any, Hashable, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

NA_VALUES = ["NULL", "", "N/A", "NAN", "NaN", "nan", "Nan", "n/a", "null"]

# PYPSA_V1 = bool(re.match(r"^1\.\d", pypsa.__version__))

# index cols of summary tables
INDEX_COLS = ["run_name_prefix", "run_name", "country", "year", "simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand", "h2export"] #, "eopts"

##### Data collection #####

def chdir_to_root_dir(root_dir=None):
    """
    Change the current working directory to the project root directory and add it to sys.path.
    
    Parameters
    ----------
    root_dir : str, Path, or None, optional
        The target root directory to change to. If None (default), automatically sets
        the root directory to the parent of the current file's directory (i.e., one level
        up from where this script is located). If provided, should be a valid path string
        or Path object.
    
    Side Effects
    ------------
    - Changes the current working directory using os.chdir()
    - Adds the root directory to sys.path for module imports
    
    Notes
    -----
    This function only executes if the current file's directory has valid path parts.
    It's designed to be safe when called from various contexts but should be used
    carefully in production code due to its global state modifications.
    """
    file_dir = Path(__file__).parent.resolve()
    
    if file_dir.parts:
        if root_dir is None:
            root_dir = file_dir.parent
        else:
            root_dir = Path(root_dir).resolve()
        os.chdir(root_dir)
        sys.path.append(str(root_dir))


def read_csv_nafix(file, **kwargs):
    "Function to open a csv as pandas file and standardize the na value"
    if "keep_default_na" not in kwargs:
        kwargs["keep_default_na"] = False
    if "na_values" not in kwargs:
        kwargs["na_values"] = NA_VALUES

    if os.stat(file).st_size > 0:
        return pd.read_csv(file, **kwargs)
    else:
        return pd.DataFrame()


def to_csv_nafix(df, path, **kwargs):
    if "na_rep" in kwargs:
        del kwargs["na_rep"]
    # if len(df) > 0:
    if not df.empty or not df.columns.empty:
        return df.to_csv(path, **kwargs, na_rep=NA_VALUES[0])
    else:
        with open(path, "w") as fp:
            pass


def read_stats_dict(stats_name, path_dir, keys=[]):
    stats_dict = {}
    index_cols = ["run_name_prefix", "run_name", "country", "year", "simpl", "clusters", "ll", "opts", "sopts", "discountrate", "demand", "h2export"] #, "eopts"
    for key in keys:
        stats_dict[key] = read_csv_nafix(path_dir / f"{stats_name}_{key}.csv", index_col=index_cols)
        stats_dict[key].index.set_names(index_cols, inplace=True)
        print(f"Imported {key} from {path_dir / f'{stats_name}_{key}.csv'}")
    return stats_dict


def set_scen_col_H2G(df, index_levels_to_drop=[]):
    """
    Set combined scenario str (for plots).
    NB: This is specific to the experiment and should be adapted for other experiments.
    """
    df = df.copy()
    
    country_info = df["country"]

    # demand_info = df["demand"]#.map({"Exp": ""})

    # Add export info based on 'eopts', handling NaN and string values
    export_info = df["h2export"].astype(float).div(33.33).round(1).astype(str) + "MtH2export"

    df["scen"] = country_info + "-" + export_info

    df = df.drop(columns=index_levels_to_drop) #"ll"
    #df["scen"] = df.apply(lambda row: "_".join([f"{col}_{row[col]}" for col in df.columns if col not in cols_to_ignore]), axis=1)
    #df["scen"] = df.apply(lambda row: "_".join([f"{row[col]}" for col in df.columns if col not in cols_to_ignore]), axis=1)

    return df

def prepare_dataframe(
    stats_df: pd.DataFrame,
    idx_group: Any,
    index_levels_to_drop,
    set_scen_col_name: Callable[[pd.DataFrame], pd.DataFrame],
    id_vars: Optional[List[str]] = None,
    groupby_vars: Optional[List[str]] = None,
    round_decimals: int = 1,
    drop_zero: bool = True,
    zero_threshold: float = 1e-10,
    rename_function: Optional[Callable[[str], str]] = None
) -> pd.DataFrame:
    """
    Prepare and transform a statistics DataFrame for plotting.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Input statistics DataFrame
    idx_group : Any
        Index group selector for filtering the DataFrame
    set_scen_col_name : Callable[[pd.DataFrame], pd.DataFrame]
        Function to set scenario column names
    id_vars : List[str], optional
        List of identifier variables for melting. If None, uses default.
    groupby_vars : List[str], optional
        List of variables for grouping. If None, uses id_vars + ["variable"].
    round_decimals : int, default 1
        Number of decimal places to round to
    drop_zero : bool, default True
        Whether to drop rows with zero values
    zero_threshold : float, default 1e-10
        Threshold below which values are considered zero
    rename_function : Callable[[str], str], optional
        Function to rename variable names before grouping. If provided, 
        will be applied to the 'variable' column after melting.
        
    Returns
    -------
    pd.DataFrame
        Processed DataFrame ready for plotting
        
    Raises
    ------
    ValueError
        If required columns are missing or if DataFrame is empty
    KeyError
        If idx_group is not found in DataFrame index
    """
    # Input validation
    if stats_df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Set default id_vars if not provided
    if id_vars is None:
        id_vars = ["run_name_prefix", "run_name", "scen", "year", "country"]
    
    # Validate id_vars exist in DataFrame columns after reset_index
    try:
        df = stats_df.copy().loc[idx_group].reset_index()
    except (KeyError, IndexError) as e:
        raise KeyError(f"Index group {idx_group} not found in DataFrame: {e}")
    
    if df.empty:
        raise ValueError(f"No data found for index group: {idx_group}")
    
    # Apply scenario column naming function
    try:
        df = set_scen_col_name(df, index_levels_to_drop=index_levels_to_drop)
    except Exception as e:
        raise ValueError(f"Error applying set_scen_col_name function: {e}")
    
    # Check if id_vars exist in the DataFrame
    missing_id_vars = set(id_vars) - set(df.columns)
    if missing_id_vars:
        raise ValueError(f"Missing required id_vars in DataFrame: {missing_id_vars}")

    # Determine value columns (columns that are not id_vars)
    value_vars = [col for col in df.columns if col not in id_vars]
    
    if not value_vars:
        raise ValueError("No value columns found for melting after excluding id_vars")
    
    # Melt the DataFrame
    try:
        df_melted = df.melt(id_vars=id_vars, value_vars=value_vars)
    except Exception as e:
        raise ValueError(f"Error during DataFrame melting: {e}")
    
    # Apply rename function to variable column if provided
    if rename_function is not None:
        try:
            df_melted['variable'] = df_melted['variable'].map(rename_function)
        except Exception as e:
            raise ValueError(f"Error applying rename_function: {e}")
    
    # Set default groupby_vars if not provided
    if groupby_vars is None:
        groupby_vars = id_vars + ["variable"]
    
    # Validate groupby_vars exist in melted DataFrame
    missing_groupby_vars = set(groupby_vars) - set(df_melted.columns)
    if missing_groupby_vars:
        raise ValueError(f"Missing required groupby_vars in melted DataFrame: {missing_groupby_vars}")
    
    unexpected_value_columns = set(df_melted.columns) - set(groupby_vars) - {"value"}
    if unexpected_value_columns:
        raise ValueError(f"Unexpected value columns found in melted DataFrame: {unexpected_value_columns}.")

    # Group by and sum
    try:
        df_grouped = df_melted.groupby(groupby_vars, as_index=False).sum(numeric_only=True)
    except Exception as e:
        raise ValueError(f"Error during groupby operation: {e}")
    
    # Round values
    if 'value' in df_grouped.columns:
        df_grouped['value'] = df_grouped['value'].round(round_decimals)
    else:
        raise ValueError(f"Something went wrong. Column 'value' not found after groupby.")
        # # Round all numeric columns
        # numeric_cols = df_grouped.select_dtypes(include=[np.number]).columns
        # df_grouped[numeric_cols] = df_grouped[numeric_cols].round(round_decimals)
    
    # Drop zero values if requested
    if drop_zero and 'value' in df_grouped.columns:
        df_grouped = df_grouped[abs(df_grouped["value"]) > zero_threshold]

    df_grouped = df_grouped[df_grouped["variable"]!= "load shedding"]
    
    return df_grouped


def get_supply_demand_from_balance(
    stats_df: pd.DataFrame,
    threshold: float = 0.01,
    round_decimals: int = 1,
    id_vars: Optional[List[str]] = None,
    groupby_vars: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get supply and demand from balance with validation and flexible groupby options.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        Input statistics DataFrame with 'value' column
    threshold : float, default 0.01
        Threshold for separating supply (>=threshold) from demand (<=-threshold)
        and for filtering out small values when drop_below_threshold is True
    round_decimals : int, default 1
        Number of decimal places to round to
    id_vars : List[str], optional
        List of identifier variables for grouping. If None, uses default.
    groupby_vars : List[str], optional
        List of variables for detailed grouping. If None, uses id_vars + ["variable"].
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (supply_df, supply_sum_df, demand_df, demand_sum_df)
        
    Raises
    ------
    ValueError
        If required columns are missing or if DataFrame is empty
    """
    if 'value' not in stats_df.columns:
        raise ValueError("DataFrame must contain 'value' column")
    
    if id_vars is None:
        id_vars = ["run_name_prefix", "run_name", "scen", "year", "country"]
    
    if groupby_vars is None:
        groupby_vars = id_vars + ["variable"]
    
    missing_id_vars = set(id_vars) - set(stats_df.columns)
    if missing_id_vars:
        raise ValueError(f"Missing required id_vars in DataFrame: {missing_id_vars}")
    
    missing_groupby_vars = set(groupby_vars) - set(stats_df.columns)
    if missing_groupby_vars:
        raise ValueError(f"Missing required groupby_vars in DataFrame: {missing_groupby_vars}")
    
    expected_columns = set(groupby_vars) | {"value"}
    unexpected_value_columns = set(stats_df.columns) - expected_columns
    if unexpected_value_columns:
        raise ValueError(f"Unexpected value columns found in melted DataFrame: {unexpected_value_columns}.")

    supply_df = stats_df[stats_df["value"] >= threshold].copy()
    supply_df = supply_df.groupby(groupby_vars, as_index=False).sum(numeric_only=True).round(round_decimals)
    supply_sum_df = supply_df.groupby(id_vars, as_index=False).sum(numeric_only=True).round(round_decimals)

    # demand_df = stats_df[abs(stats_df["value"]) >= threshold].copy()
    demand_df = stats_df[stats_df["value"] <= -threshold].copy()
    demand_df["value"] *= -1  # Convert to positive values
    demand_df = demand_df.groupby(groupby_vars, as_index=False).sum(numeric_only=True).round(round_decimals)
    demand_sum_df = demand_df.groupby(id_vars, as_index=False).sum(numeric_only=True).round(round_decimals)

    return supply_df, supply_sum_df, demand_df, demand_sum_df


def update_layout(fig):
    # Only apply textangle and textposition to bar traces to avoid conflicts with scatter traces
    fig.update_traces(textposition='inside', textangle=0, selector=dict(type='bar'))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(tickangle=25)
    # fig.update_yaxes(matches=None)
    # # Add ticks and show y-axis scale (numbers) for all y-axes in subplot
    # fig.update_yaxes(
    #     # ticks="outside",
    #     # ticklen=5,
    #     # tickwidth=1,
    #     # tickcolor='black',
    #     # showline=True,
    #     # linewidth=1,
    #     # linecolor='black',
    #     showticklabels=True  # Ensure y-axis numbers are shown
    # )
    return

def add_production_consumption_legend_groups(fig, df):
    """
    Add legend groups for Production (positive values) and Consumption (negative values).
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure to modify
    df : pandas.DataFrame
        The dataframe containing the data with 'variable' and 'value' columns
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The modified figure with legend groups
    """
    # Add legend groups for Production (positive values) and Consumption (negative values)
    for trace in fig.data:
        # Get the variable name from the trace
        variable_name = trace.name
        
        # Check if this variable typically has positive or negative values in the data
        variable_data = df[df['variable'] == variable_name]['value']
        
        if len(variable_data) > 0:
            # Determine if this variable is primarily production (positive) or consumption (negative)
            avg_value = variable_data.mean()
            
            if avg_value > 0:
                trace.legendgroup = "Production"
                trace.legendgrouptitle = {"text": "Production"}
            else:
                trace.legendgroup = "Consumption" 
                trace.legendgrouptitle = {"text": "Consumption"}
    
    return fig

def add_totals_to_plot(fig, totals_df, **kwargs):
    """
    Add totals as invisible text traces above stacked bars for plots with positive values only.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure to add totals to
    totals_df : pandas.DataFrame
        DataFrame containing totals with columns: ['scen', 'year', 'value'] (and optionally others)
    **kwargs : dict
        Optional formatting parameters:
        - textfont_size : int, default 16
        - textfont_color : str, default 'black'  
        - y_offset : float, default 1 (amount to offset text above bars)
        - text_format : str, default '.0f' (format string for values)
        - value_column : str, default 'value' (name of column containing totals)
        - y_margin : float, default 0.1 (fraction of max value to add as top margin)
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The modified figure with totals added
    """
    # Extract kwargs with defaults
    textfont_size = kwargs.get('textfont_size', 14)
    textfont_color = kwargs.get('textfont_color', 'black')
    y_offset = kwargs.get('y_offset', 1)
    text_format = kwargs.get('text_format', '.0f')
    value_column = kwargs.get('value_column', 'value')
    
    # Prepare totals data for plotting
    totals_data = totals_df.reset_index()
    if value_column != 'total':
        totals_data = totals_data.rename(columns={value_column: 'total'})
    
    # Add totals trace for each year subplot
    for year in totals_data['year'].unique():
        year_data = totals_data[totals_data['year'] == year]
        
        # Determine which subplot this year corresponds to
        years_list = sorted(totals_data['year'].unique())
        col_num = years_list.index(year) + 1
        
        # Format the text values
        text_values = [f"{val:{text_format}}" for val in year_data['total']]
        
        fig.add_trace(
            go.Scatter(
                x=year_data['scen'],
                y=year_data['total'] + y_offset,
                mode='text',
                text=text_values,
                textposition='top center',
                textfont=dict(size=textfont_size, color=textfont_color, weight="bold"),
                showlegend=False,
                hoverinfo='skip',
                xaxis=f'x{col_num}' if col_num > 1 else 'x',
                yaxis=f'y{col_num}' if col_num > 1 else 'y'
            )
        )
    
    return fig


def add_balance_totals_to_plot(fig, supply_sum_df, demand_sum_df, **kwargs):
    """
    Add totals as invisible text traces for plots with both positive and negative values (barmode="relative").
    Shows supply totals above positive bars and demand totals below negative bars.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure to add totals to
    supply_sum_df : pandas.DataFrame
        DataFrame containing supply totals with columns: ['scen', 'year', 'value'] (and optionally others)
    demand_sum_df : pandas.DataFrame
        DataFrame containing demand totals with columns: ['scen', 'year', 'value'] (and optionally others)
    **kwargs : dict
        Optional formatting parameters:
        - textfont_size : int, default 14
        - textfont_color : str, default 'black'  
        - y_offset : float, default 1 (amount to offset text from bars)
        - text_format : str, default '.0f' (format string for values)
        - value_column : str, default 'value' (name of column containing totals)
        - supply_label : str, default '' (optional prefix label for supply totals)
        - demand_label : str, default '' (optional prefix label for demand totals)
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The modified figure with supply and demand totals added
    """
    # Extract kwargs with defaults
    textfont_size = kwargs.get('textfont_size', 14)
    textfont_color = kwargs.get('textfont_color', 'black')
    y_offset = kwargs.get('y_offset', 1)
    text_format = kwargs.get('text_format', '.0f')
    value_column = kwargs.get('value_column', 'value')
    supply_label = kwargs.get('supply_label', '')
    demand_label = kwargs.get('demand_label', '')
    
    # Prepare supply totals data
    supply_data = supply_sum_df.reset_index()
    if value_column != 'total':
        supply_data = supply_data.rename(columns={value_column: 'total'})
    
    # Prepare demand totals data
    demand_data = demand_sum_df.reset_index()
    if value_column != 'total':
        demand_data = demand_data.rename(columns={value_column: 'total'})
    
    # Add supply totals trace for each year subplot (above positive bars)
    for year in supply_data['year'].unique():
        year_data = supply_data[supply_data['year'] == year]
        
        # Determine which subplot this year corresponds to
        years_list = sorted(supply_data['year'].unique())
        col_num = years_list.index(year) + 1
        
        # Format the text values for supply (positive values above bars)
        text_values = [f"{supply_label}{val:{text_format}}" if val > 0 else "" 
                      for val in year_data['total']]
        
        fig.add_trace(
            go.Scatter(
                x=year_data['scen'],
                y=year_data['total'] + y_offset,
                mode='text',
                text=text_values,
                textposition='top center',
                textfont=dict(size=textfont_size, color=textfont_color, weight="bold"),
                showlegend=False,
                hoverinfo='skip',
                xaxis=f'x{col_num}' if col_num > 1 else 'x',
                yaxis=f'y{col_num}' if col_num > 1 else 'y'
            )
        )
    
    # Add demand totals trace for each year subplot (below negative bars)
    for year in demand_data['year'].unique():
        year_data = demand_data[demand_data['year'] == year]
        
        # Determine which subplot this year corresponds to
        years_list = sorted(demand_data['year'].unique())
        col_num = years_list.index(year) + 1
        
        # Format the text values for demand (negative values below bars)
        # Note: demand values are typically negative, so we show them below
        text_values = [f"{demand_label}{abs(val):{text_format}}" if val < 0 else "" 
                      for val in year_data['total']]
        
        fig.add_trace(
            go.Scatter(
                x=year_data['scen'],
                y=year_data['total'] - y_offset,
                mode='text',
                text=text_values,
                textposition='bottom center',
                textfont=dict(size=textfont_size, color=textfont_color, weight="bold"),
                showlegend=False,
                hoverinfo='skip',
                xaxis=f'x{col_num}' if col_num > 1 else 'x',
                yaxis=f'y{col_num}' if col_num > 1 else 'y'
            )
        )
    
    return fig


##### Plotting #####

def nice_title(title, subtitle):
    return f'{title}' + '<br>' +  f'<span style="font-size: 12px;">{subtitle}</span>'

my_template = go.layout.Template(
    layout=dict(
        font_color='#000000',
        #font_family="Open Sans",
        plot_bgcolor = "#ffffff",  #rgba(212,218,220,255)
        paper_bgcolor = "#ffffff", 
        legend=dict(bgcolor='rgba(0,0,0,0)'), #"#ffffff"
        title={'y': 0.95, 'x': .06},
        font_size=14,
        uniformtext_minsize=10, 
        uniformtext_mode='hide',
        margin=dict(l=0, r=50, t=50, b=50),
        # margin=dict(l=15, r=0, t=80, b=0),
))

pio.renderers.default = 'plotly_mimetype+notebook_connected'
pio.templates["my"] = my_template
pio.templates.default = "simple_white+xgridoff+ygridoff+my"
#pio.kaleido.scope.mathjax= None

def save_plotly_fig(df, fig, output_dir, fig_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir/f"{fig_name}.csv", index=False)
    
    # Extract width and height from figure layout to preserve dimensions
    width = fig.layout.width
    height = fig.layout.height
    
    fig.write_image(output_dir/f"{fig_name}.png", 
                   width=width, 
                   height=height, 
                   engine="kaleido")


##### Renaming and consolidation #####
#%%

def rename_techs(label):
    """Consolidated technology renaming function.
    
    Applies base cleanup (prefix removal, substring matching, exact matching)
    followed by specific technology mappings.
    """

    # Exact rename mappings
    rename = {
        "solar": "Solar PV",
        "solar rooftop": "Solar PV",
        "offwind": "Offshore wind",
        "offwind-ac": "Wind offshore (AC)",
        "offwind-dc": "Wind offshore (DC)",
        "onwind": "Wind onshore",
        "ror": "Hydro",
        "hydro": "Hydro",
        "PHS": "Hydro",
        "NH3": "Ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "Residential",
        "DC": "Transmission lines",
        "B2B": "Transmission lines",
        "Combined-Cycle Gas": "Gas turbines",
        "battery": "Battery",
        "Load_Shedding": "Backup/load-shedding",
        "helmeth": "CH4 Synthesis",
        "H2 Fuel Cell": "H2 Fuel Cell",
        "OCGT": "Gas turbines",
        "CCGT": "Gas turbines",
        "OCGT (Diesel)": "Gas turbines",
        "sasol_gas": "Gas turbines",
        "ocgt_diesel": "Gas turbines",
        "CHP": "Gas turbines",
        "Fischer-Tropsch": "FT Synthesis",
        "Fischer-Tropsch -> oil": "FT Synthesis",
        "CH3OH Synthesis": "FT Synthesis",
        "solar PV": "Solar PV",
        "solar_pv": "Solar PV",
        "solar CSP": "Solar CSP",
        "solar_csp": "Solar CSP",
        "coal": "Coal",
        "Sasol_coal": "Coal",
        "Sabatier": "CH4 Synthesis",
    }
    for old, new in rename.items():
        if old == label:
            label = new
            break

    # Remove prefixes
    prefix_to_remove = [
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]
    for ptr in prefix_to_remove:
        if label[: len(ptr)] == ptr:
            label = label[len(ptr) :]

    # Rename if substring contains
    rename_if_contains = [
        "CHP",
        "Gas boiler",
        "Biogas",
        "Solar thermal",
        "Air heat pump",
        "Ground heat pump",
        "Resistive heater",
    ]
    for rif in rename_if_contains:
        if rif in label:
            label = rif
            break

    # Rename if substring contains (with specific mappings)
    rename_if_contains_dict = {
        "water tanks": "Hot water storage",
        "retrofitting": "Building retrofitting",
        "battery": "Battery",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "Land transport fuel cell",
        "land transport oil": "Land transport oil",
        "oil shipping": "Shipping oil",
        "Hydro": "Hydro",
        "heat pump": "Power-to-heat",
        "resistive heater": "Power-to-heat",
        "H2 pipeline": "H2 pipeline",
        "V2G": "Vehicle-to-Grid",
        "Haber-Bosch": "NH3 Synthesis",
        "offshore wind": "Offshore wind",
        "DAC": "Direct air capture",
        "CC": "Carbon capture",
        "sequestration": "Carbon capture",
        "biomass": "Biomass",
        "nuclear": "Nuclear",
    }
    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new
            break

    
    # Step 5: Special case with substring replacement
    if "SMR" in label:
        return label.replace("SMR", "Steam reforming")
    
    return label
    

def rename_to_upper_case(tech):
    tech = tech[0].upper() + tech[1:]
    return tech


def rename_costs(tech):
    """
    """

    tech = rename_to_upper_case(tech)
    # Check patterns on original tech before renaming
    if tech in ["Fischer-Tropsch export"]:
        return "PtL export"    
    elif tech in ["Haber-Bosch"]:
        return "NH3 Synthesis"
    elif tech in ["NH3 export"]:
        return "Ammonia export"
    elif tech in ["H2 export", "H2"]:
        return "H2 export"
    elif tech in ["H2 Store Tank"]:
        return "H2 storage"
    elif tech in ["electricity distribution grid"]:
        return "Distribution grid"
    
    # Apply base renaming
    tech = rename_techs(tech)
    
    if tech in ["Solar PV", "Solar CSP"]:
        return "Solar"
    elif tech in ["Onshore Wind", "Offshore Wind (DC)", "Wind onshore"]:
        return "Wind"
    elif tech in ["H2 Electrolysis"]:
        return "H2 Electrolysis"
    elif tech in ["Hydro Power", "Hydroelectricity"]:
        return "Hydro"    
    elif tech in ["Battery"]:
        return "Battery"
    elif tech in ['Transmission lines']:
        return "Transmission line"
    elif tech in ["Coal"]:
        return "Coal"
    elif tech in ["Oil"]:
        return "Oil"
    elif tech in ["Gas"]:
        return "Gas"
    elif tech in ["Nuclear"]:
        return "Nuclear"
    elif tech in ["CH4 Synthesis"]:
        return "CH4 Synthesis"
    elif tech in ["FT Synthesis"]:
        return "FT Synthesis"
    elif tech in ["Power-to-heat"]:
        return "Power-to-heat"
    elif tech in ["Process emissions CC", "Carbon capture", "Direct air capture"]:
        return "Carbon capture"
    else:
        return tech
    
def rename_oil(tech):
    # Check patterns on original tech before renaming
    if "rail transport" in tech:
        return "Rail transport"
    elif "shipping" in tech:
        return "Navigation"
    elif "industry" in tech:
        return "Industry"
    elif "land transport" in tech:
        return "Land transport"
    elif "aviation" in tech:
        return "Aviation"
    elif "services" in tech:
        return "Commerce"
    elif "agriculture" in tech:
        return "Agriculture"
    elif "residential" in tech:
        return "Residential"
    elif tech in ["oil"]:
        return "Fossil fuel import"
    elif tech in ["Fischer-Tropsch export"]:
        return "FT Synthesis export"
    
    # Apply base renaming for other cases
    tech = rename_techs(tech)
    if tech in ["FT Synthesis"]:
        return "FT Synthesis"
    else:
        return tech
    
def rename_gas(tech):
    # Check patterns on original tech before renaming
    if tech in ["gas"]:
        return "Gas import"
    elif "methanation" in tech:
        return "CH4 Synthesis"
    elif "residential" in tech:
         return "Residential"
    elif "services" in tech:
        return "Commerce"
    elif tech == "gas for industry CC":
        return "Industry with CC"
    elif tech == "gas for industry":
        return "Industry"    
    
    # Apply base renaming
    tech = rename_techs(tech)
    if tech in ["OCGT", "CCGT", "Gas turbines"]:
        return "Gas turbines"
    elif tech == "Biogas":
        return "Biogas"
    elif tech == "Gas boiler":
        return "Gas boiler"
    else:
        return tech
    
def rename_h2(tech):
    # Check patterns on original tech before renaming
    if "methanolisation" in tech:
        return "CH3OH Synthesis"
    elif "shipping" in tech:
        return "Navigation"
    elif "industry" in tech:
        return "Other industry"
    elif "land transport" in tech:
        return "H2 Fuel Cell"
    elif tech == "H2":
        return "H2 export"
    elif tech == "DRI":
        return "H2 DRI"
    
    # Apply base renaming for other cases
    tech = rename_techs(tech)
    if tech == "H2 Fuel Cell":
        return "H2 Fuel Cell"
    else:
        return tech

def rename_electricity(tech):
    # Remove suffix first
    suffix_to_remove = [" electricity"]
    for sfx in suffix_to_remove:
        tech = tech.removesuffix(sfx)
    
    # Check patterns on original tech before renaming
    if tech == 'BEV charger':
        return 'Transport'
    if tech == 'rail transport':
        return 'Transport'
    elif tech == 'seawater desalination':
        return "Industry"
    elif tech == 'services':
        return "Commerce"
    
    # Apply base renaming
    tech = rename_techs(tech)
    
    if tech == 'Transmission lines':
        return "Residential"
    elif tech in [
        "Power-to-heat", 
        "FT Synthesis", "NH3 Synthesis",  
        "Direct air capture",
        "EAF", "DRI"]:
        return "Industry"
    else:
        return tech

def rename_co2(tech):
    # Check exact matches on original tech before renaming
    if tech == 'solid biomass for industry CC':
        return 'Biomass for industry with CC'        
    if tech == 'urban central solid biomass CHP CC':
        return 'Biomass for CHP with CC'   
    elif tech == 'gas for industry CC':
        return "Gas for industry with CC"
    elif tech == 'process emissions CC':
        return "Process emissions with CC"
    
    # Apply base renaming for other cases
    tech = rename_techs(tech)
    return tech


colors = {
    "electricity": {
        "Industry": "#a6bbc8", #'#f58220',
        "Commerce": "#008598", #"#b2d235",
        "Residential": "#39c1cd", #"#d3c7ae",
        "Transport": '#a8508c',
        #"Solar": '#fdb913',
        "Solar PV": '#fdb913',
        "Solar CSP": '#face61',
        #"Wind": '#005b7f',
        "Wind onshore": '#005b7f',
        "Wind offshore (DC)": "#39c1cd",
        "Coal": '#454545',
        "Hydro": '#a6bbc8',
        "Nuclear": "#bb0056",
        "Gas turbines": '#d3c7ae',
        "CHP": "#d6a67c",
        'NH3 Synthesis': '#fce356',
        "Agriculture": '#f08591',
        "Rail transport": '#008598',
        "Desalination": "#39c1cd",
        "Direct air capture": "#1c3f52",
        "Electric vehicles": '#a8508c',
        "Battery": '#836bad',
        "Battery charger": '#836bad',
        "Battery discharger": '#836bad',
        "Vehicle-to-Grid": '#a8508c',
        "Biomass": "#b2d235",
        "Other": "#a6bbc8",
        "Power-to-heat": "#FCD80E",
        "CH4 Synthesis": "#d3c7ae",
        'H2 Electrolysis': "#179c7d",
        "H2 Fuel Cell": "#179c7d",
        "H2 turbine": '#f08591',
        "CH3OH Synthesis": "#bb0056",
        "FT Synthesis": "#7c154d",
        "Oil": "#1c3f52",
        "Geothermal": '#f08591',
    },
    "hydrogen": {
        'H2 Electrolysis': "#179c7d",
        'Other industry': '#a6bbc8',
        'Land transport': '#669db2',
        "H2 Fuel Cell": "#179c7d",
        "H2 turbine": '#f08591',
        'NH3 Synthesis': '#fce356',
        'CH4 Synthesis': '#d3c7ae',
        'FT Synthesis': '#a8508c',
        "CH3OH Synthesis": "#836bad",
        "H2 DRI": '#1c3f52',
        "H2 export": "#4CC2A6",
        "Navigation": "#39c1cd",
        "Methanol steam reforming": "#39c1cd",
        'Steam reforming': '#d3c7ae',
    },
    "oil": {
        "Fuel import or CtL": "#1c3f52",
        "Fossil fuel import": "#1c3f52",
        'FT Synthesis': '#a8508c',
        'Land transport': '#669db2',
        "Navigation": "#39c1cd",
        "Aviation": "#fce356",
        "Industry": '#f58220',
        "Commerce": "#b2d235",
        "Residential": "#d3c7ae",
        "Agriculture": '#f08591',
        "Rail transport": '#008598',
        'CH4 Synthesis': '#d3c7ae',
        "FT Synthesis export": "#7c154d",
    },
    "gas": {
        'CH4 Synthesis': '#d3c7ae',
        "Commerce": "#b2d235",
        'Gas import': "#fce356",
        'Gas turbines': '#d3c7ae',
        'Residential': "#f58220",
        'Industry': '#a6bbc8',
        'Industry with CC': "#005b7f",
        "Biogas": "#b2d235",
        "Gas boiler": "#d6a67c",
    },
    "co2": {
        'Process emissions with CC': '#1c3f52',
        'Biomass for industry with CC': '#b2d235',
        "Biomass for CHP with CC": "#C2D05C",
        'Gas for industry with CC': '#005b7f',
        'Direct air capture': '#7c154d',
        'FT Synthesis': '#bb0056',
        "CH4 Synthesis": '#d3c7ae',
    },
    "costs": {
        "Ammonia export": '#bb0056',
        "PtL export": '#7c154d',
        "Solar": '#fdb913',
        "Wind": '#005b7f',
        "H2 Electrolysis": "#179c7d",
        "H2 export": "#7c154d",
        "NH3 Synthesis": '#fce356',
        "Transmission line": '#4cc2a6',
        "Battery": '#836bad',
        "Oil": '#C0C0C0',
        "Coal": '#454545',
        "Gas": "#d3c7ae",
        "Hydro": '#a6bbc8',
        "Nuclear": "#bb0056",
        'Carbon capture': '#7c154d',
        "Other": "#a6bbc8",
    }
}

def set_scen_col_DKS(df, index_levels_to_drop=[]):
    """
    Set combined scenario string for DKS experiments (for plots).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with index levels including 'demand', 'eopts', 'sopts'
    index_levels_to_drop : list
        List of index level names to drop after creating scen column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with new 'scen' column and specified index levels dropped
    """
    df = df.copy()
    
    # Set base scen value based on 'demand'
    scen = df["demand"].map({"RF": "AB", "EL": "AT"})
    
    # Add export info based on 'eopts', handling NaN and string values
    export_info = df["eopts"].apply(
        lambda x: "50EX" if isinstance(x, str) and "v0.5" in x
        else ("0EX" if pd.isna(x) or str(x) == "" else "100EX")
    )
    
    # Combine scenario components
    df["scen"] = scen + export_info.replace("", "", regex=False).apply(lambda x: f"-{x}" if x else "") + df["sopts"].replace("1H", "").replace("3H", "")
    df["scen"] = df["scen"].str.replace("AT-100EX","AT").str.replace("AB-50EX","AB")
    df = df.drop(columns=index_levels_to_drop)
    
    return df


def get_scen_col_function(name: str):
    """
    Get scenario column function by name.
    
    Parameters
    ----------
    name : str
        Function name: "DKS" or "MAPaper"
        
    Returns
    -------
    Callable
        Scenario column function
        
    Raises
    ------
    ValueError
        If function name is not recognized
    """
    functions = {
        "H2G": set_scen_col_H2G,
    }
    
    if name not in functions:
        raise ValueError(f"Unknown scenario column function: {name}. Available: {list(functions.keys())}")
    
    return functions[name]


def apply_standard_styling(fig, chart_type: str, config: dict = None):
    """
    Apply standard styling to plotly figures based on chart type.
    
    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to style
    chart_type : str
        Type of chart: "bar", "bar_with_negatives", or "bar_single"
    config : dict, optional
        Additional configuration options to override defaults
        
    Returns
    -------
    plotly.graph_objects.Figure
        Styled figure
    """
    import plotly.graph_objects as go
    
    config = config or {}
    
    if chart_type == "bar":
        # Standard bar chart styling
        fig.update_traces(
            textposition=config.get('textposition', 'inside'),
            textangle=config.get('textangle', 0),
            selector=dict(type='bar')
        )
        fig.update_layout(
            legend_traceorder=config.get('legend_traceorder', 'reversed')
        )
        
        # Clean up facet annotations (remove "year=" prefix)
        if config.get('facet_annotation_cleanup', True):
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    elif chart_type == "bar_with_negatives":
        # Bar chart with positive and negative values
        fig.update_traces(
            textposition=config.get('textposition', 'inside'),
            textangle=config.get('textangle', 0),
            selector=dict(type='bar')
        )
        fig.update_layout(
            legend_traceorder=config.get('legend_traceorder', 'normal')
        )
        
        if config.get('facet_annotation_cleanup', True):
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    elif chart_type == "bar_single":
        # Single-variable bar chart (e.g., hydrogen capacity)
        fig.update_traces(
            textposition=config.get('textposition', 'auto'),
            selector=dict(type='bar')
        )
        fig.update_layout(
            legend_traceorder=config.get('legend_traceorder', 'normal')
        )
        
        if config.get('facet_annotation_cleanup', True):
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    else:
        raise ValueError(f"Unknown chart_type: {chart_type}. Use 'bar', 'bar_with_negatives', or 'bar_single'")
    
    return fig


def plot_energy_balance(
    carrier: str,
    balance_dict: dict,
    config: dict,
    rename_function=None,
    post_process_function=None,
    show_supply: bool = None,
    save_dir=None
):
    """
    Create energy balance plot for a specific carrier.
    
    Parameters
    ----------
    carrier : str
        Carrier name (key in balance_dict)
    balance_dict : dict
        Dictionary of balance DataFrames by carrier
    config : dict
        Configuration dictionary from YAML with chart-specific settings
    rename_function : Callable, optional
        Function to rename variables
    post_process_function : Callable, optional
        Function to post-process variable names (e.g., rename_to_upper_case)
    show_supply : bool, optional
        If True, show only supply; if False, show only demand; if None, show both
    save_dir : Path, optional
        Directory to save the figure
        
    Returns
    -------
    plotly.graph_objects.Figure
        Created figure
    """
    import plotly.express as px
    from pathlib import Path
    
    # Get chart-specific config
    chart_config = config.get('chart_config', {})
    idx_group = config['idx_group']
    index_levels_to_drop = config['index_levels_to_drop']
    set_scen_col = config['set_scen_col_func']
    scen_filter = config.get('scen_filter', False)
    scen_order = config.get('scen_order', [])
    fig_kwargs = config['fig_kwargs']
    idx_group_name = config['idx_group_name']
    
    # Prepare dataframe
    df = prepare_dataframe(
        balance_dict[carrier],
        idx_group,
        index_levels_to_drop,
        set_scen_col,
        rename_function=rename_function
    )
    
    # Apply post-processing
    if post_process_function:
        df.variable = df.variable.map(post_process_function)
    
    # Filter zeros if specified
    if chart_config.get('filter_zeros', False):
        df = df[df["value"] != 0]
    
    # Apply scenario filter
    if scen_filter:
        df = df.query("scen in @scen_filter")
    
    # Split supply/demand if requested
    if chart_config.get('split_supply_demand', False):
        supply_df, supply_sum_df, demand_df, demand_sum_df = get_supply_demand_from_balance(df)
        
        if show_supply:
            plot_df = supply_df
            totals_df = supply_sum_df
        else:
            plot_df = demand_df
            totals_df = demand_sum_df
    else:
        plot_df = df
        supply_df, supply_sum_df, demand_df, demand_sum_df = get_supply_demand_from_balance(df)
        totals_df = None
    
    # Get colors
    carrier_key = carrier if carrier in colors else "electricity" if carrier == "AC" else "hydrogen"
    color_map = chart_config.get('color_discrete_map', colors.get(carrier_key, {}))
    
    # Override fig_kwargs with chart-specific settings
    fig_kwargs_custom = fig_kwargs.copy()
    if 'height' in chart_config:
        height_key = chart_config['height']
        fig_kwargs_custom['height'] = config['heights'].get(height_key, 500)
    if 'text_auto' in chart_config:
        fig_kwargs_custom['text_auto'] = chart_config['text_auto']
    
    # Create figure
    fig = px.bar(
        plot_df,
        **fig_kwargs_custom,
        labels=chart_config.get('labels', {}),
        color_discrete_map=color_map,
        category_orders={
            "variable": list(color_map.keys()),
            "scen": scen_order
        }
    )
    
    # Apply styling
    chart_type = chart_config.get('chart_type', 'bar')
    fig = apply_standard_styling(fig, chart_type, chart_config)
    
    # Apply standard layout updates
    update_layout(fig)
    
    # Add totals if appropriate
    if chart_config.get('split_supply_demand', False) and totals_df is not None:
        fig = add_totals_to_plot(fig, totals_df)
    elif chart_config.get('add_legend_groups', False):
        fig = add_production_consumption_legend_groups(fig, plot_df)
        fig = add_balance_totals_to_plot(
            fig, supply_sum_df, demand_sum_df,
            y_offset=chart_config.get('y_offset', 5),
            supply_label='',
            demand_label=''
        )
    
    # Save figure
    if save_dir:
        fig_name = chart_config.get('fig_name', f"{carrier}_balance")
        save_plotly_fig(plot_df, fig, save_dir, f"{idx_group_name}_{fig_name}")
    
    return fig, df


def plot_capacity(
    carrier: str,
    capacity_dict: dict,
    config: dict,
    rename_function=None,
    post_process_function=None,
    save_dir=None
):
    """
    Create installed capacity plot for a specific carrier.
    
    Parameters
    ----------
    carrier : str
        Carrier name (key in capacity_dict)
    capacity_dict : dict
        Dictionary of capacity DataFrames by carrier
    config : dict
        Configuration dictionary from YAML with chart-specific settings
    rename_function : Callable, optional
        Function to rename variables
    post_process_function : Callable, optional
        Function to post-process variable names
    save_dir : Path, optional
        Directory to save the figure
        
    Returns
    -------
    plotly.graph_objects.Figure
        Created figure
    """
    import plotly.express as px
    from pathlib import Path
    
    # Get configuration
    chart_config = config.get('chart_config', {})
    idx_group = config['idx_group']
    index_levels_to_drop = config['index_levels_to_drop']
    set_scen_col = config['set_scen_col_func']
    scen_filter = config.get('scen_filter', False)
    scen_order = config.get('scen_order', [])
    fig_kwargs = config['fig_kwargs']
    idx_group_name = config['idx_group_name']
    
    # Prepare dataframe
    df = prepare_dataframe(
        capacity_dict[carrier],
        idx_group,
        index_levels_to_drop,
        set_scen_col,
        rename_function=rename_function
    )
    
    # Apply post-processing
    if post_process_function:
        df.variable = df.variable.map(post_process_function)
    
    # Filter by threshold if specified (e.g., drop loadshedding capacity)
    if 'filter_threshold' in chart_config:
        df = df[df["value"] < chart_config['filter_threshold']]
    
    # Apply scenario filter
    if scen_filter:
        df = df.query("scen in @scen_filter")
    
    # Get supply/demand split (for capacity, we just use supply as positive values)
    supply_df, supply_sum_df, demand_df, demand_sum_df = get_supply_demand_from_balance(df)
    
    # Get colors
    carrier_key = carrier if carrier in colors else "electricity" if carrier == "AC" else "hydrogen"
    color_map = chart_config.get('color_discrete_map', colors.get(carrier_key, {}))
    
    # Override fig_kwargs with chart-specific settings
    fig_kwargs_custom = fig_kwargs.copy()
    if 'height' in chart_config:
        height_key = chart_config['height']
        fig_kwargs_custom['height'] = config['heights'].get(height_key, 500)
    if 'text_auto' in chart_config:
        fig_kwargs_custom['text_auto'] = chart_config['text_auto']
    
    # Create figure
    fig = px.bar(
        df,
        **fig_kwargs_custom,
        labels=chart_config.get('labels', {}),
        color_discrete_map=color_map,
        category_orders={
            "variable": list(color_map.keys()),
            "scen": scen_order
        }
    )
    
    # Apply styling
    chart_type = chart_config.get('chart_type', 'bar')
    fig = apply_standard_styling(fig, chart_type, chart_config)
    
    # Apply standard layout updates
    update_layout(fig)
    
    # Add totals
    if supply_sum_df is not None and not supply_sum_df.empty:
        textfont_size = chart_config.get('totals_font_size', 14)
        fig = add_totals_to_plot(fig, supply_sum_df, textfont_size=textfont_size)
    
    # Save figure
    if save_dir:
        fig_name = chart_config.get('fig_name', f"{carrier}_capacity")
        save_plotly_fig(df, fig, save_dir, f"{idx_group_name}_{fig_name}")
    
    return fig, df


def load_plot_config(config_path: str = None):
    """
    Load plotting configuration from YAML file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration YAML file. If None, looks for plot_summary.yaml
        in the post_processing directory.
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    import yaml
    from pathlib import Path
    
    if config_path is None:
        config_path = Path(__file__).parent / "plot_summary.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config