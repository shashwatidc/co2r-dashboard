
################################################################################
################################### IMPORTS ####################################
################################################################################

### Import packages

import streamlit as st
# from streamlit_super_slider import st_slider as superslider

import pandas as pd
import numpy as np

import matplotlib as mp
# from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib import ticker
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import animation
# from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import threading

# import csv

from datetime import datetime
# import timeit

# from functools import reduce

# from os.path import exists
# import os

# import openpyxl
# from openpyxl.worksheet.dimensions import ColumnDimension, DimensionHolder
# import openpyxl.utils.cell
# from openpyxl.styles import PatternFill, Border, Side, Alignment, Protection, Font

# from scipy import optimize

# from IPython.display import display, HTML, clear_output

# import import_ipynb
# from testernb import single_run
# from testerscript import single_run

from ElectrolyzerModel import *
from DownstreamProcessModel import *
from ProcessEconomics import *
from TEA_SingleRun import *

# Cache single run of model
@st.cache_data(ttl = "1h")
def cached_single_run(product_name,
        product_rate_kg_day,
        df_products,
        FE_CO2R_0,
        FE_product_specified,
        j_total_mA_cm2,
        SPC,
        crossover_ratio,
        P,
        T_streams,
        R_ohmcm2,
        an_E_eqm,
        an_eta_ref,
        an_Tafel_slope,
        an_j_ref,
        cathode_outlet_humidity,
        excess_water_ratio,   
        electrolyte_conc,  
        density_kgm3,
        PSA_second_law_efficiency,
        carbon_capture_efficiency,
        T_sep,         
        electricity_cost_USD_kWh,
        heat_cost_USD_kWh,
        electricity_emissions_kgCO2_kWh,
        heat_emissions_kgCO2_kWh,
        product_cost_USD_kgprod,
        H2_cost_USD_kgH2,
        water_cost_USD_kg,
        CO2_cost_USD_tCO2,
        electrolyzer_capex_USD_m2,       
        lifetime_years,
        stack_lifetime_years,
        capacity_factor,
        battery_capex_USD_kWh,               
        battery_capacity,
        is_additional_capex,
        is_additional_opex,
        additional_capex_USD,
        additional_opex_USD_kg,
        model_FE,
        overridden_vbl,
        overridden_value,
        overridden_unit,
        override_optimization,
        exponent,
        scaling,
        MW_CO2,
        MW_H2O,
        MW_O2,
        MW_MX,
        R, 
        F,
        K_to_C = 273.15,
        kJ_per_kWh = 3.60E+03,
        ):
    # Execute a single run of the model. The actual function is imported from TEA_SingleRun.ipynb.
    # This is useful becuase of the caching - it will only actually rerun the model if there is a change in the inputs
    return single_run(product_name=product_name,
        product_rate_kg_day=product_rate_kg_day,
        df_products=df_products,
        FE_CO2R_0=FE_CO2R_0,
        FE_product_specified=FE_product_specified,
        j_total_mA_cm2=j_total_mA_cm2,
        SPC=SPC,
        crossover_ratio=crossover_ratio,
        P=P,
        T_streams = T_streams,
        R_ohmcm2 = R_ohmcm2,
        an_E_eqm = an_E_eqm,
        an_eta_ref = an_eta_ref,
        an_Tafel_slope = an_Tafel_slope,
        an_j_ref = an_j_ref,
        cathode_outlet_humidity = cathode_outlet_humidity,
        excess_water_ratio = excess_water_ratio,   
        electrolyte_conc = electrolyte_conc,  
        density_kgm3 = density_kgm3,
        PSA_second_law_efficiency = PSA_second_law_efficiency,
        carbon_capture_efficiency = carbon_capture_efficiency,
        T_sep = T_sep,         
        electricity_cost_USD_kWh = electricity_cost_USD_kWh,
        heat_cost_USD_kWh = heat_cost_USD_kWh,
        electricity_emissions_kgCO2_kWh = electricity_emissions_kgCO2_kWh,
        heat_emissions_kgCO2_kWh = heat_emissions_kgCO2_kWh,
        product_cost_USD_kgprod = product_cost_USD_kgprod,
        H2_cost_USD_kgH2 = H2_cost_USD_kgH2,
        water_cost_USD_kg = water_cost_USD_kg,
        CO2_cost_USD_tCO2 = CO2_cost_USD_tCO2,
        electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2,       
        lifetime_years = lifetime_years,
        stack_lifetime_years = stack_lifetime_years,
        capacity_factor = capacity_factor,
        battery_capex_USD_kWh = battery_capex_USD_kWh,               
        battery_capacity = battery_capacity,
        model_FE = model_FE,
        is_additional_capex=is_additional_capex,
        is_additional_opex=is_additional_opex,
        additional_opex_USD_kg = additional_opex_USD_kg,
        additional_capex_USD = additional_capex_USD,
        overridden_vbl = overridden_vbl,
        overridden_value = overridden_value,
        overridden_unit = overridden_unit,
        override_optimization = override_optimization,
        exponent = exponent,
        scaling = scaling,
        MW_CO2 = MW_CO2,
        MW_H2O = MW_H2O,
        MW_O2 = MW_O2,
        MW_MX = MW_MX,
        R = R, 
        F = F,
        K_to_C = K_to_C,
        kJ_per_kWh = kJ_per_kWh,
        )

# Cache default model run 
# @st.cache_data(ttl = "1h")
# def default_single_run(product_name,
#         product_rate_kg_day,
#         df_products,
#         FE_CO2R_0,
#         FE_product_specified,
#         j_total_mA_cm2,
#         SPC,
#         crossover_ratio,
#         P,
#         T_streams,
#         R_ohmcm2,
#         an_E_eqm,
#         an_eta_ref,
#         an_Tafel_slope,
#         an_j_ref,
#         cathode_outlet_humidity,
#         excess_water_ratio,   
#         electrolyte_conc,  
#         density_kgm3,
#         PSA_second_law_efficiency,
#         carbon_capture_efficiency,
#         T_sep,         
#         electricity_cost_USD_kWh,
#         heat_cost_USD_kWh,
#         electricity_emissions_kgCO2_kWh,
#         heat_emissions_kgCO2_kWh,
#         product_cost_USD_kgprod,
#         H2_cost_USD_kgH2,
#         water_cost_USD_kg,
#         CO2_cost_USD_tCO2,
#         electrolyzer_capex_USD_m2,       
#         lifetime_years,
#         stack_lifetime_years,
#         is_additional_capex, is_additional_opex,
#         additional_capex_USD, additional_opex_USD_kg,
#         capacity_factor,
#         battery_capex_USD_kWh,               
#         battery_capacity,
#         model_FE,
#         overridden_vbl,
#         overridden_value,
#         overridden_unit,
#         override_optimization,
#         exponent,
#         scaling,
#         MW_CO2,
#         MW_H2O,
#         MW_O2,
#         MW_MX,
#         R, 
#         F,
#         K_to_C = 273.15,
#         kJ_per_kWh = 3.60E+03,
#         ):
#     __, df_capex_totals_default, __, __, __, __,\
#                 df_energy_default, __, __, __, __, __, df_opex_totals_default, __,\
#                 __, df_potentials_default, __, __, __, __, __ = cached_single_run(product_name,
#         product_rate_kg_day,
#         df_products,
#         FE_CO2R_0,
#         FE_product_specified,
#         j_total_mA_cm2,
#         SPC,
#         crossover_ratio,
#         P,
#         T_streams,
#         R_ohmcm2,
#         an_E_eqm,
#         an_eta_ref,
#         an_Tafel_slope,
#         an_j_ref,
#         cathode_outlet_humidity,
#         excess_water_ratio,   
#         electrolyte_conc,  
#         density_kgm3,
#         PSA_second_law_efficiency,
#         carbon_capture_efficiency,
#         T_sep,         
#         electricity_cost_USD_kWh,
#         heat_cost_USD_kWh,
#         electricity_emissions_kgCO2_kWh,
#         heat_emissions_kgCO2_kWh,
#         product_cost_USD_kgprod,
#         H2_cost_USD_kgH2,
#         water_cost_USD_kg,
#         CO2_cost_USD_tCO2,
#         electrolyzer_capex_USD_m2,       
#         lifetime_years,
#         stack_lifetime_years,
#         capacity_factor,
#         is_additional_capex, is_additional_opex,
#         additional_capex_USD, additional_opex_USD_kg,
#         battery_capex_USD_kWh,               
#         battery_capacity,
#         model_FE,
#         overridden_vbl,
#         overridden_value,
#         overridden_unit,
#         override_optimization,
#         exponent,
#         scaling,
#         MW_CO2,
#         MW_H2O,
#         MW_O2,
#         MW_MX,
#         R, 
#         F,
#         K_to_C,
#         kJ_per_kWh,
#         )
#     capex_default = df_capex_totals_default.loc['Total permanent investment', 'Cost ($)']
#     opex_default = df_opex_totals_default.loc['Production cost', 'Cost ($/kg {})'.format(product_name)]
#     levelized_default = df_opex_totals_default.loc['Levelized cost', 'Cost ($/kg {})'.format(product_name)]
#     potential_default = df_potentials_default.loc['Cell potential', 'Value'] 
#     energy_default = df_energy_default.loc['Total', 'Energy (kJ/kg {})'.format(product_name)]
#     emissions_default = sum(df_energy_default.fillna(0).iloc[:-2].loc[:, 'Emissions (kg CO2/kg {})'.format(product_name)])
#     return capex_default, opex_default, levelized_default, potential_default, energy_default, emissions_default

_render_lock = threading.RLock()

###################################################################################
################################### FORMATTING ####################################
###################################################################################

# Streamlit page formatting
st.set_page_config(page_title = 'CO2R Costing Dashboard - Ethylene bar charts', 
                   page_icon = ":test_tube:",
                   initial_sidebar_state= 'expanded',
                   layout="wide")

# Plot formatting for Matplotlib - rcParams. All options at https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams

# All fonts and font sizes
SMALL_SIZE = 20 # set smallest font size
MEDIUM_SIZE = 24 # set medium font size
BIGGER_SIZE = 27 # set
mp.rcParams["font.family"] = "sans-serif"
mp.rcParams["font.sans-serif"] = "Liberation Sans"
mp.rc('font', size=MEDIUM_SIZE)     # controls default text sizes if unspecified
mp.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title; I think this is for subplots 
mp.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
mp.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
mp.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
mp.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mp.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# text padding
mp.rc('axes', labelpad= 18) # padding between axis label and axis
mp.rc('axes', titlepad= 22) # padding between axis title and labels; might be for subplots
mp.rc('legend', handletextpad = 0.3) # padding between each legend entry's icon and text
mp.rc('legend', borderaxespad = 1) # padding between legend border and axes
TICK_PADDING = 8
mp.rcParams['xtick.major.pad'] = TICK_PADDING
mp.rcParams['xtick.minor.pad'] = TICK_PADDING
mp.rcParams['ytick.major.pad'] = TICK_PADDING
mp.rcParams['ytick.minor.pad'] = TICK_PADDING
mp.rcParams['axes.xmargin'] = 1
mp.rcParams['axes.ymargin'] = 1

# figure settings
aspect_ratio = 1/1 # 3/4
mp.rc('figure', figsize = (5, 5*aspect_ratio)) # figure size
mp.rc('figure', dpi = 250) # figure dpi/ pix per inch
mp.rcParams['axes.spines.right'] = True # right border
mp.rcParams['axes.spines.top'] = True # top border

# legend
mp.rc('legend', loc = 'upper left') # legend location 
mp.rc('legend', frameon = False) # legend border - yes or no?
mp.rc('legend', markerscale = 1) # scale up markers in legend

# axes
mp.rc('axes', linewidth = 2) # linewidth of axes

# default axes
mp.rc('axes', autolimit_mode = 'round_numbers') # set default axis limits to be "round" numbers rather than data

# major x-ticks
mp.rcParams['xtick.top'] = False  # top or bottom of plot
mp.rcParams['xtick.direction'] = 'out' # ticks in or out of plot
mp.rcParams['xtick.major.width'] = 2 # linewidth of ticks
mp.rcParams['xtick.major.size'] = 12 # length of ticks

# minor x-ticks
mp.rcParams['xtick.minor.visible'] = True
mp.rcParams['xtick.minor.width'] = 2  # linewidth of ticks
mp.rcParams['xtick.minor.size'] = 6   # length of ticks

# major y-ticks
mp.rcParams['ytick.right'] = False  # right or left of plot
mp.rcParams['ytick.direction'] = 'out' # ticks in or out of plot
mp.rcParams['ytick.major.width'] = 2   # linewidth of ticks
mp.rcParams['ytick.major.size'] = 12  # length of ticks

# minor y-ticks
mp.rcParams['ytick.minor.visible'] = True
mp.rcParams['ytick.minor.right'] = False # right or left of plot
mp.rcParams['ytick.minor.width'] = 2 # linewidth of ticks
mp.rcParams['ytick.minor.size'] = 6  # length of ticks

# format for saving figures
mp.rcParams['savefig.format'] = 'tiff'
mp.rcParams['savefig.bbox'] = 'tight' # 'tight' # or standard; tight may break ffmpeg
# If using standard, be sure to use bbox_inches="tight" argument in savefig

# defaults for scatterplots
mp.rcParams['scatter.marker'] = 'o' # round markers unless otherwise specified
mp.rcParams['lines.markersize'] = 6 # sets the scatter/line marker size; roughly equivalent to s = 40 in my experience

# defaults for lines
mp.rcParams['lines.linestyle'] = '-' # solid lines unless otherwise specified
mp.rcParams['lines.linewidth'] = 2 # default linewidth

# defaults for errorbars
mp.rcParams['errorbar.capsize'] = 4

### Fix random state for reproducibility
np.random.seed(19680801)

### Some options for ticks:
# np.arange(min, max, step): returns a list of step-spaced entries between min and max EXCLUDING max
# np.linspace(min, max, n): returns a list of n linearly spaced entries between min and max, including max
# np.logspace(min, max, n, base=10.0): returns a list of n log-spaced entries between min and max
# axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(n)): sets axis ticks to be multiples of 
                                                            #n within the data range

## Theme colors 
theme_colors = ['#bf5700',  '#ffc919', '#8f275d', '#73a3b3', '#193770', '#e35555', '#191f24' ] #ffffff (white)

## Import colormaps
# summer = mp.colormaps['summer']
# summer_r = mp.colormaps['summer_r']
# PuOr = mp.colormaps['PuOr']
viridis = mp.colormaps['viridis']
# viridis_r = mp.colormaps['viridis_r']
# wistia = mp.colormaps['Wistia']
# greys = mp.colormaps['gist_yarg'] # 'Gray'
# RdBu = mp.colormaps['RdBu'] # seismic
RdYlBu = mp.colormaps['RdYlBu']
# inferno = mp.colormaps['inferno_r']
# Blues = mp.colormaps['Blues']
# winter = mp.colormaps['winter_r']
# cool = mp.colormaps['cool_r']

## Custom colormaps
# Endpoint colors
colors = [ '#fff01f', '#00503d']  # gold to sherwood green
bright_summer_r = LinearSegmentedColormap.from_list('custom_cmap', colors) # create colormap

colors = ['#abd5e2', '#190033', '#a60027', theme_colors[1]  ] #  
diverging = LinearSegmentedColormap.from_list('diverging_cmap', colors) # create colormap

# colors = ['#a60027', theme_colors[1], theme_colors[3], '#012469'  ] #  
# RdYlBu = LinearSegmentedColormap.from_list('diverging_cmap', colors)

colors = ['#a60027', '#ffefdc', '#012469'  ] #  
RdBu = LinearSegmentedColormap.from_list('diverging_cmap', colors)

# st.markdown(
#     """
#     <style>
#         section[data-testid="stSidebar"] {
#             width: 750px !important; # Set the width to your desired value
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

###################################################################################
################################## STREAMLIT APP ##################################
###################################################################################

#__________________________________________________________________________________

################################## TITLE AND INTRO ################################
# from matplotlib.backends.backend_agg import RendererAgg
# _lock = RendererAgg.lock # Lock figures so that concurrent users/threads can coexist independently

st.title("CO$_2$R Costing Dashboard: Bar charts for CO$_2$R to Ethylene")
st.write("*Developed by [Shashwati da Cunha](https://shashwatidc.github.io/) in the [Resasco Catalysis Lab](https://www.resascolab.com/)*")
st.write('''Visualize how the capex and opex respond to a change in a single process parameter for CO₂R to ethylene.
         Pick a parameter and modify the settings on the left to see how the results change. 
         ''')

st.write('''**Update (March 12, 2025):** The CO₂R Dashboard has been updated! Capital costs are now adjusted to the approximate average CEPCI for 2024 (800). 
         Industrial electricity prices are now the average for 2024 (\$0.082/kWh). The base case single-pass conversion and total current density have been adjusted to the optimal in the Hawks model at these new costs.
         The market price of ethylene is updated to the 2024 global average. Pure CO is a difficult chemical to price since it is rarely sold, usually used within a facility where it is generated.
         The base price for CO (\$0.6/kg in 2001) has also been updated with an arbitrary 1% inflation rate. Note that it may be more likely to track natural gas prices, which slightly dipped from 2001 to 2024 on the Henry Hub.   
         ''')

st.write('**:red[Known issues]:** Currently, there is no warning if you enter a numeric value in any text box that is out of  \
         physical range, e.g. capacity factor > 1, single-pass conversion > 1. The displayed results will be physically unreasonable. \
         User is responsible for checking that their inputs are reasonable.')

# st.write(
#     '<img width=100 src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/squid_1f991.png" style="margin-left: 5px; filter: hue-rotate(230deg) brightness(1.1);">',
#     unsafe_allow_html=True,
# )
# st.write(
#     """
#     [![Github](https://img.shields.io/github/stars/jrieke/year-on-github.svg?logo=github&style=social)](https://gitHub.com/jrieke/year-on-github)
#     [![Follow](https://img.shields.io/twitter/follow/jrieke?style=social)](https://www.twitter.com/jrieke)
#     """
# )

with st.expander("**Help**", expanded = False):
    st.write("""Select the variable which you want to see the costs respond to, and adjust the costing assumptions and model in the left sidebar.
        Select how many points you want to generate, the range between them, and whether they are linearly or log-spaced. The bar charts breaking down 
        cost will update dynamically every time you change the model. This model and its assumptions are based on our [2024 paper](https://pubs.acs.org/doi/10.1021/acsenergylett.4c02647).
        Plot formatting options to change the range and number of ticks (axis labels) can be found in **Plot formatting** in the sidebar.
            Please note that it can take some time to run the model for a large number of datapoints. If possible, start with a few points and add more once you have the settings you want. 
           \n By default, the cell voltage will be modeled using Tafel equations, and the Faradaic efficiency based on the single-pass conversion and the maximum Faradaic efficiency.
        Mouse over the :grey[**?**] next to each input to see the default values for each parameter. Refresh the page to reset all values to their defaults.      
        This dashboard will not exactly match the paper, since costs have been updated since its publication. You can get close to reproducing it
            by adjusting the electrolyzer capex to \$5000/m$^2$, electricity price to \$0.076/kWh, single-pass conversion for CO
            to 11.5%, total current density for CO to 472 mA/cm$^2$, single-pass conversion for ethylene to 2.7%, 
            and total current density for ethylene to 436 mA/cm$^2$, CO market price to \$0.6/kg, and ethylene market price to \$0.96. 
            However, the PSA capex and labor costs will still differ from the original, so the resulting price will still be slightly higher. 
        """)

st.write("**Cite this work**: [10.1021/acsenergylett.4c02647](https://pubs.acs.org/doi/10.1021/acsenergylett.4c02647)")
st.write("**Questions, collaborations, requests?** Contact [shashwati.dc@utexas.edu](mailto:shashwati.dc@utexas.edu).")

#__________________________________________________________________________________

###############################  IMPORT SOURCE DATA ###############################

file_imports = r"Supplementary Workbook.xlsx"
sheet_utility_imports = 'Utilities'
sheet_constants =  'Constants and assumptions'
sheet_products =  'Products'

# Get current date and time to name files
time_now = datetime.now().time()
date_now = datetime.now().date()
current_date = date_now.strftime("%Y%m%d") # format string
current_time = time_now.strftime("%I-%M%p") # format string

# Cache Excel sheet reading
@st.cache_data(ttl = "1h")
def import_data(file_imports):
    df_constants = pd.DataFrame # Create dataframe for constants
    xlsx = pd.ExcelFile(file_imports) # Read each data Excel file
    df_constants = xlsx.parse(sheet_name = sheet_constants) # Read the sheet with the constants
    df_constants.set_index('Variable name', drop = True, inplace = True) # Reset index to variable name
    xlsx.close() # close xlsx file

    df_products = pd.DataFrame # Create dataframe for product data
    xlsx = pd.ExcelFile(file_imports) # Read each data Excel file
    df_products = xlsx.parse(sheet_name = sheet_products) # Read the sheet with the product data
    df_products.set_index('Product', drop = True, inplace = True) # reset index to product name
    xlsx.close() # close xlsx file
        
    df_utility_imports = pd.DataFrame # Create dataframe for costs
    xlsx = pd.ExcelFile(file_imports) # Read each data Excel file
    df_utility_imports = xlsx.parse(sheet_name = sheet_utility_imports) # Read the sheet with the costing
    df_utility_imports.set_index('Utility', drop = True, inplace = True) # reset index to utility name
    xlsx.close() # close xlsx file

    return(df_constants, df_products, df_utility_imports)

df_constants, df_products, df_utility_imports = import_data(file_imports)

#_________________________________________________________________________________

############################# CONSTANTS AND PARAMETERS ############################

### Extract constants to use for costing and emissions calculations

## Update constants as variables 
## NOTE: Modifying globals() is a very frowned on practice in Python - be VERY careful implementing this, it's easy to say 
# overwrite variables that are carelessly named. Consider indexing directly from df_constants (set its index to 'variable name')
# and df_utilities instead
for index, row in df_constants.iterrows(): # for each row in df_constants
        globals()[index] = row['Value'] # save its 'value' to a new variable called the 'variable name' column
        globals()['default_'+index] = row['Value']
# Utility costs
electricity_cost_USD_kWh = df_utility_imports.loc['Electric utility chosen', 'Cost ($/kWh)'] 
default_electricity_cost_USD_kWh = electricity_cost_USD_kWh
heat_cost_USD_kWh = df_utility_imports.loc['Heat utility chosen', 'Cost ($/kWh)']
default_heat_cost_USD_kWh = heat_cost_USD_kWh
battery_capex_USD_kWh = df_utility_imports.loc['Battery storage', 'Cost ($/kWh)']
default_battery_capex_USD_kWh = battery_capex_USD_kWh

# Utility emissions
electricity_emissions_kgCO2_kWh = 0.001 * df_utility_imports.loc['Electric utility chosen', 'CO2 emissions (g CO2/kWh)'] # convert g CO2 to kg CO2
default_electricity_emissions_kgCO2_kWh = electricity_emissions_kgCO2_kWh
heat_emissions_kgCO2_kWh = 0.001 * df_utility_imports.loc['Heat utility chosen', 'CO2 emissions (g CO2/kWh)'] # convert g CO2 to kg CO2
default_heat_emissions_kgCO2_kWh = heat_emissions_kgCO2_kWh

######## DEFAULTS
override_one = False
override_onebyone = False
overridde_multivbl = False
override_optimization = False
override_animation = False
override_single = False
model_FE = 'Kas'
is_battery = False
is_additional_capex = False
is_additional_opex = False
st.session_state.is_active_error_ethylene = False
product_name = 'Ethylene' # default
range_selection = 'Linear' # default
override_vbl_selection = 'Total current density' # default
vbl_name = 'Current density' # default
vbl_unit = 'mA/cm$^2$' # default
if 'minimum_value_input_ethylene' not in st.session_state:
    st.session_state.minimum_value_input_ethylene = str(0.001)
if 'maximum_value_input_ethylene' not in st.session_state:
    st.session_state.maximum_value_input_ethylene = str(1500)

########## OTHER FIXED VALUES
# PRODUCT COSTS
product_cost_USD_kgprod = df_products.loc[product_name, 'Cost ($/kg product)']
default_product_cost_USD_kgprod = product_cost_USD_kgprod
H2_cost_USD_kgH2 = float(df_products.loc['H2', 'Cost ($/kg product)']) # assume H2 is not sold
default_H2_cost_USD_kgH2 = H2_cost_USD_kgH2

# RAW INPUTS
crossover_ratio = crossover_neutral
default_crossover_ratio = crossover_ratio
FE_product_specified = df_products.loc[product_name, 'FECO2R at SPC = 0']  # 0.9 # 0.90 # %/100
default_FE_product_specified = FE_product_specified
j_total_mA_cm2 = float(df_products.loc[product_name, 'Optimal j @ 8.2 c/kWh, Hawks model']) # 300 # mA/cm2
default_j_total_mA_cm2 = j_total_mA_cm2
cell_E_V = 3.0  # default cell voltage
default_cell_E_V = cell_E_V
BV_eta_cat_V = -0.6
default_BV_eta_cat_V = BV_eta_cat_V
BV_eta_an_V = 0.25
default_BV_eta_an_V = BV_eta_an_V
FE_CO2R_0 = df_products.loc[product_name, 'FECO2R at SPC = 0']
default_FE_CO2R_0  = FE_CO2R_0
SPC = df_products.loc[product_name, 'Optimal SPC @ 8.2 c/kWh, Hawks model']  #0.3 # 0.5 # %/100
default_SPC = SPC
cat_Tafel_slope = df_products.loc[product_name, 'Tafel slope (mV/dec)']
default_cat_Tafel_slope = cat_Tafel_slope

# Save variables that may be adjusted
scaling = 4.7306 ## TODO: Move this option for modeling elsewhere
default_scaling = scaling
exponent = 5.4936 ## TODO: Move this option for modeling elsewhere
default_exponent = exponent

additional_capex_USD = 0.0
additional_opex_USD_kg = 0.0

##### CHOICE OF x-AXIS VARIABLE
options_list  = ['Cell voltage (V)', 
                'Cathodic overpotential (V)',
                'Anodic overpotential (V)',
                'Membrane resistance ($\Omega$.cm$^2$)',
                'Total current density (mA/cm$^2$)',
                '$FE_{CO_2R, \: 0}$',
                'Single-pass conversion',
                'Crossover ratio (mol CO$_2$ per mol e$^-$)',
                'Production rate (kg/day)',
                'Capacity factor',
                'Lifetime (yrs)',
                'Stack lifetime (yrs)',
                'Separation efficiency',
                'Electricity cost ($/kWh)',
                'CO$_2$ cost ($/t)',
                'H$_2$ cost ($/kg)',
                'Water cost ($/kg)',
                'Electrolyzer capex (\$/m$^2$)',
                # 'Renewables capacity factor'
                ] # Order must exactly match df_flags

middle_column, right_column = st.columns(2, gap = 'large')
st.sidebar.header('x-axis variable' )
middle_column.header('Results')
right_column.header('_')

# Cache creation of flags dataframe
@st.cache_data(ttl = "1h")
def flags(product_name):
    # Create flags for selecting variable
    dict_flags = {   # Formatted as 'override_parameter': 'parameter name', 'unit', 'variable name', 'default value', 'minimum value', 'maximum value', 
        'override_cell_voltage': [ 'Cell voltage', 'V', 'cell_E_V',                                   cell_E_V, 1.34, 5 ],
        'override_eta_cat': ['Cathodic overpotential', 'V', 'BV_eta_cat_V',                          BV_eta_cat_V, 0, -2.5],
        'override_eta_an': ['Anodic overpotential', 'V', 'BV_eta_an_V',                              BV_eta_an_V, 0, 2.5 ],
        'override_ohmic' : ['Specific resistance', '$\Omega$.cm$^2$', 'R_ohmcm2',               R_ohmcm2, 0, 25],

        'override_j': ['Current density', 'mA/cm$^2$', 'j_total_mA_cm2',                             df_products.loc[product_name, 'Optimal j @ 8.2 c/kWh, Hawks model'], 25, 1275],
        # 'override_FE': ['FE$_{{{}}}$'.format(product_name), '', 'FE_product_specified',              df_products.loc[product_name, 'FECO2R at SPC = 0'], 1e-3, 1 ],
        'override_FE_CO2R_0': ['FE$_{CO_2R,0}$', '', 'FE_CO2R_0',                                    df_products.loc[product_name, 'FECO2R at SPC = 0'], 1e-3, 1 ],
        'override_SPC':['Single-pass conversion' , '', 'SPC',                                        df_products.loc[product_name, 'Optimal SPC @ 8.2 c/kWh, Hawks model'], 1e-4, 0.5],
        'override_crossover': ['Crossover' , 'mol CO$_2$ per mol e$^-$', 'crossover_ratio',          crossover_ratio, 1e-4, 0.5],
        
        'override_rate': ['{} production rate'.format(product_name), 'kg/day', 'product_rate_kg_day', product_rate_kg_day, 1e-3, 1.25e6],
        'override_capacity': ['Capacity factor' , '', 'capacity_factor',                             capacity_factor, 1e-4, 1 ],
        'override_lifetime': ['Plant lifetime' , 'years', 'lifetime_years',                          lifetime_years, 1e-3, 50],
        'override_stack_lifetime': ['Stack lifetime' , 'years', 'stack_lifetime_years',              stack_lifetime_years, 1e-3, 30],
        'override_separation': ['Second-law separation efficiency', '', 'PSA_second_law_efficiency',   PSA_second_law_efficiency, 0.01, 0.1],
        
        'override_electricity_cost':[ 'Electricity cost' , '$/kWh', 'electricity_cost_USD_kWh',      electricity_cost_USD_kWh, 0, 0.1 ],
        'override_CO2_cost': ['CO$_2$ cost'  , '\$/t CO$_2$', 'CO2_cost_USD_tCO2',                   CO2_cost_USD_tCO2, 0, 200],
        'override_H2_cost': ['H$_2$ cost'  , '\$/kg H$_2$', 'H2_cost_USD_kgH2',                   H2_cost_USD_kgH2, 0, 10],
        'override_water_cost': ['Water cost' , '\$/kg', 'water_cost_USD_kg',                      water_cost_USD_kg, 0, 0.30],
        'override_electrolyzer_capex': ['Electrolyzer capital cost' , '\$/m$^2$', 'electrolyzer_capex_USD_m2', electrolyzer_capex_USD_m2, 3000, 10000],
        'override_carbon_intensity': ['Grid CO$_2$ intensity', 'kg CO$_2$/kWh', 'electricity_emissions_kgCO2_kWh',electricity_emissions_kgCO2_kWh, 0, 0.5],
        
        # 'override_battery_capacity': ['Renewables capacity factor' , '', 'avbl_renewables',        avbl_renewables, 1e-4, 1 ],
        }
    # Note that percentages here are expressed directly as decimals. E.g. entering 0.01 above for default FE will result in default FE = 1%

    df_flags = pd.DataFrame(dict_flags).T
    df_flags.reset_index(inplace = True, drop = False)
    df_flags.set_index(0, inplace = True, drop = True) # Set independent variable name as index
    df_flags.index.name = 'Independent variable'
    df_flags.columns = ['Old flag name', 'Unit', 'Python variable', 'Default value', 'Range min', 'Range max']
    
    # Clear flags
    df_flags['T/F?'] = False # add column for truth value of given override
    return df_flags

@st.cache_data(ttl = "1h")
def generate_range(df_flags, override_vbl_selection, vbl_min, vbl_max, vbl_num):
    vbl_row = options_list.index(override_vbl_selection) # convert input into integer
    
    df_flags['T/F?'] = False # clear column
    df_flags.iloc[vbl_row, 6] = True # set that flag to be True
    vbl_name = df_flags.index[vbl_row] # set vbl_name from that row
    vbl_unit = df_flags['Unit'].iloc[vbl_row] # set vbl_unit from that row

    # Reorder potential limits defined and generate a range of the chosen independent variable
    vbl_limits = [vbl_min, vbl_max]
    vbl_min = min(vbl_limits)
    vbl_max = max(vbl_limits)

    # Generate range
    
    # Linearly space points based on step size
    if range_selection == 'Linear':
        vbl_range = np.linspace(start = vbl_min, stop = vbl_max, num = vbl_num) # include the last point as close as possible

    # Log space points
    else:
        vbl_range = np.logspace(start = np.log10(vbl_min), stop = np.log10(vbl_max), num = vbl_num, base = 10, endpoint = True)

    vbl_range_text = ['{} {}'.format(x, vbl_unit) for x in vbl_range]
    
    return vbl_name, vbl_unit, vbl_range, vbl_range_text, vbl_min, vbl_max

# Cache creation of axis variables
@st.cache_data(ttl = "1h")
def x_axis_formatting(x_axis_min, x_axis_max, x_axis_num):    
    if range_selection == 'Linear':
        x_axis_major_ticks = np.linspace(start = x_axis_min, stop = x_axis_max, num = x_axis_num)
    else:
        x_axis_major_ticks = np.logspace(np.log10(x_axis_min), np.log10(x_axis_max), num = x_axis_num, endpoint = True)

    # Some options for ticks:
    # np.arange(min, max, step): returns a list of step-spaced entries between min and max EXCLUDING max
    # np.linspace(min, max, n): returns a list of n linearly spaced entries between min and max, including max
    # np.logspace(min, max, n, base=10.0): returns a list of n log-spaced entries between min and max
    # axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(n)): sets axis ticks to be multiples of 
                                                                    #n within the data range

    return(x_axis_major_ticks)

@st.cache_data(ttl = "1h")
def y_axis_formatting(y_axis_min, y_axis_max, y_axis_num):    
    if range_selection == 'Linear':
        y_axis_major_ticks = np.linspace(start = y_axis_min, stop = y_axis_max, num = y_axis_num)
    else:
        y_axis_major_ticks = np.logspace(np.log10(y_axis_min), np.log10(y_axis_max), num = y_axis_num, endpoint = True)

    # Some options for ticks:
    # np.arange(min, max, step): returns a list of step-spaced entries between min and max EXCLUDING max
    # np.linspace(min, max, n): returns a list of n linearly spaced entries between min and max, including max
    # np.logspace(min, max, n, base=10.0): returns a list of n log-spaced entries between min and max
    # axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(n)): sets axis ticks to be multiples of 
                                                                    #n within the data range

    return(y_axis_major_ticks)

def updated_radio_state(df_flags):
    vbl_row = options_list.index(st.session_state['overridden_vbl_radio_ethylene']) # convert input into integer
    vbl_name = df_flags.index[vbl_row] # set vbl_name from that row
    vbl_unit = df_flags.loc[vbl_name, 'Unit']
    st.session_state.minimum_value_input_ethylene = str(df_flags.loc[vbl_name, 'Range min'])
    st.session_state.maximum_value_input_ethylene = str(df_flags.loc[vbl_name, 'Range max'])

df_flags = flags(product_name)

with st.sidebar:
    ## Initialize overridden_vbl_radio_ethylene widget

    override_vbl_selection = st.radio(label = 'Select variable to see cost sensitivity ', key='overridden_vbl_radio_ethylene',
                          options= options_list, 
                    index = 4, # default option
                    label_visibility='visible',
                    help = '''Choose a variable as the x-axis (category) for the bar charts. 
                      \n Then choose its range below by defining the minimum, maximum, and number of bars to generate between them.
                      \n The actual range on the bar charts can be specified in the **Plot formatting** section below.
                      \n Default option: Total current density''',
                      on_change= updated_radio_state, args = (df_flags, )
                    )
    try:
        st.write('Minimum value')
        vbl_min = float(st.text_input(label = 'Minimum value',
                    key = 'minimum_value_input_ethylene', # value = str(df_flags.loc[vbl_name, 'Range min']),  
                    label_visibility='collapsed'))
        st.write('Maximum value')
        vbl_max = float(st.text_input(label = 'Maximum value',
                    key = 'maximum_value_input_ethylene',#  value = str(df_flags.loc[vbl_name, 'Range max']),
                    label_visibility='collapsed'))
        
        st.write('Number of points (integer)')
        vbl_num = int(st.text_input(label = 'Number of points',
                    value = 11, 
                    label_visibility='collapsed'))
        st.session_state.is_active_error_ethylene = False
    except:
        st.error('One of your x-variable values is invalid. Please check that they are all numbers and \
                  that the number of points is an integer.')
        st.session_state.is_active_error_ethylene = True
        st.header('*:red[There is an error in the model inputs.]*')

    st.write('Spacing for points')
    range_selection = st.radio(label = 'Spacing for points', 
                          options= ['Linear', 
                                    'Logspace'], 
                    index = 0, # default option
                    label_visibility='collapsed',
                    help = '''Check the variable that is the x-axis ('category') in the bar charts. 
                      \n Then choose its range below.
                      \n Default option: Total current density'''
    )

    # st.subheader('CO$_2$R product')
    # ######## PRODUCT SELECTION
    # # Choose a product
    # product_name = st.radio(label = 'Reduction product', options= ['CO', 'Ethylene'], 
    #                 index = 0, # default option
    #                 label_visibility='collapsed',
    #                 help = '''Choose the product that CO$_2$ is reduced into. 
    #                 The only byproduct made is hydrogen. 
    #                   \n Default product: CO'''
    # )

vbl_name, vbl_unit, vbl_range, vbl_range_text, vbl_min, vbl_max = generate_range(df_flags, override_vbl_selection, vbl_min, vbl_max, vbl_num)
default_y_axis_max_opex = 2.0 if product_name == 'CO' else 12.0

## Define axis limits and ticks - see note below for options
with st.sidebar:
    st.header('Plot formatting')
    with st.expander('**x-axis formatting**', expanded=False):
        st.write('x-axis minimum')
        x_axis_min = st.text_input(label = 'x-axis minimum',
                            value = vbl_min, label_visibility='collapsed',)
        st.write('x-axis maximum')
        x_axis_max = st.text_input(label = 'x-axis maximum',
                            value = vbl_max, label_visibility='collapsed')
        st.write('Number of x-ticks, including endpoints (integer)')
        x_axis_num = st.text_input(label = 'x-axis ticks',
                            value = 4, label_visibility='collapsed')
        try:
            x_axis_min = float(x_axis_min)
            x_axis_max = float(x_axis_max)
            x_axis_num = int(x_axis_num)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your x-axis values is invalid. Please check that they are all numbers and \
                    that the number of x-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs.]*')

    with st.expander('**y-axes formatting**', expanded = False):

        st.write('**Capex**')
        st.write('Capex y-axis minimum (millions)')
        y_axis_min_capex = st.text_input(label = 'capex y-axis minimum',
                            value = 0, label_visibility='collapsed',)
        st.write('Capex y-axis maximum (millions)')
        y_axis_max_capex = st.text_input(label = 'capex y-axis maximum',
                            value = 300, label_visibility='collapsed')
        st.write('Number of capex y-ticks, including endpoints (integer)')
        y_axis_num_capex = st.text_input(label = 'capex y-axis ticks',
                            value = 6, label_visibility='collapsed')
        try:
            y_axis_min_capex = float(y_axis_min_capex)
            y_axis_max_capex = float(y_axis_max_capex)
            y_axis_num_capex = int(y_axis_num_capex)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your capex y-axis values is invalid. Please check that they are all numbers and \
                    that the number of y-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs..]*')

        st.divider()
        st.write('**Opex**')
        st.write('Opex y-axis minimum ($/kg {})'.format(product_name))
        y_axis_min_opex = st.text_input(label = 'opex y-axis minimum',
                            value = 0, label_visibility='collapsed',)
        st.write('Opex y-axis maximum ($/kg {})'.format(product_name))
        y_axis_max_opex = st.text_input(label = 'opex y-axis maximum',
                            value = default_y_axis_max_opex, label_visibility='collapsed')
        st.write('Number of opex y-ticks, including endpoints (integer)')
        y_axis_num_opex = st.text_input(label = 'opex y-axis ticks',
                            value = 6, label_visibility='collapsed')
        try:
            y_axis_min_opex = float(y_axis_min_opex)
            y_axis_max_opex = float(y_axis_max_opex)
            y_axis_num_opex = int(y_axis_num_opex)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your opex y-axis values is invalid. Please check that they are all numbers and \
                    that the number of y-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs..]*')
    
        st.divider()
        st.write('**Levelized cost**')
        st.write('Levelized cost y-axis minimum')
        y_axis_min_levelized = st.text_input(label = 'levelized y-axis minimum',
                            value = 0, label_visibility='collapsed',)
        st.write('Levelized cost y-axis maximum')
        y_axis_max_levelized = st.text_input(label = 'levelized y-axis maximum',
                            value = default_y_axis_max_opex, label_visibility='collapsed')
        st.write('Number of levelized cost y-ticks, including endpoints (integer)')
        y_axis_num_levelized = st.text_input(label = 'levelized x-axis ticks',
                            value = 6, label_visibility='collapsed')
        try:
            y_axis_min_levelized = float(y_axis_min_levelized)
            y_axis_max_levelized = float(y_axis_max_levelized)
            y_axis_num_levelized = int(y_axis_num_levelized)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your levelized cost y-axis values is invalid. Please check that they are all numbers and \
                    that the number of y-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs..]*')

        st.write('**Cell potential**')
        st.write('Cell potential y-axis minimum')
        y_axis_min_potential = st.text_input(label = 'E y-axis minimum',
                            value = 0, label_visibility='collapsed',)
        st.write('Cell potential y-axis maximum')
        y_axis_max_potential = st.text_input(label = 'E y-axis maximum',
                            value = 5, label_visibility='collapsed')
        st.write('Number of potential y-ticks, including endpoints (integer)')
        y_axis_num_potential = st.text_input(label = 'E y-axis ticks',
                            value = 6, label_visibility='collapsed')
        try:
            y_axis_min_potential = float(y_axis_min_potential)
            y_axis_max_potential = float(y_axis_max_potential)
            y_axis_num_potential = int(y_axis_num_potential)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your opex y-axis values is invalid. Please check that they are all numbers and \
                    that the number of y-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs..]*')

        st.divider()
        st.write('**Energy**')
        st.write('Energy y-axis minimum')
        y_axis_min_energy = st.text_input(label = 'energy y-axis minimum',
                            value = 0, label_visibility='collapsed',)
        st.write('Energy y-axis maximum')
        y_axis_max_energy = st.text_input(label = 'energy y-axis maximum',
                            value = 15000, label_visibility='collapsed')
        st.write('Number of energy y-ticks, including endpoints (integer)')
        y_axis_num_energy = st.text_input(label = 'energy y-axis ticks',
                            value = 4, label_visibility='collapsed')
        try:
            y_axis_min_energy = float(y_axis_min_energy)
            y_axis_max_energy = float(y_axis_max_energy)
            y_axis_num_energy = int(y_axis_num_energy)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your energy y-axis values is invalid. Please check that they are all numbers and \
                    that the number of y-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs..]*')
    
        st.divider()
        st.write('**Emissions**')
        st.write('Emissions y-axis minimum')
        y_axis_min_emissions = st.text_input(label = 'emissions y-axis minimum',
                            value = 0, label_visibility='collapsed',)
        st.write('Emissions y-axis maximum')
        y_axis_max_emissions = st.text_input(label = 'emissions y-axis maximum',
                            value = 50, label_visibility='collapsed')
        st.write('Number of emissions y-ticks, including endpoints (integer)')
        y_axis_num_emissions = st.text_input(label = 'emissions y-axis ticks',
                            value = 6, label_visibility='collapsed')
        try:
            y_axis_min_emissions = float(y_axis_min_emissions)
            y_axis_max_emissions = float(y_axis_max_emissions)
            y_axis_num_emissions = int(y_axis_num_emissions)
            st.session_state.is_active_error_ethylene = False
        except:
            st.error('One of your emissions y-axis values is invalid. Please check that they are all numbers and \
                    that the number of y-ticks is an integer.')
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs..]*')


# SLIDERS 
st.sidebar.header('Model inputs' )

with st.sidebar:
    st.subheader('Cell potential model')
    with st.expander(label = '**Simplified Butler-Volmer model assumptions**', expanded = False):
    #     override_cell_voltage = st.checkbox('Manually specify full-cell voltage', value = False)
    #     cell_E_V = st.slider(label = 'Cell voltage',
    #                         min_value = 0.001, 
    #                         max_value = 10.0, 
    #                         step = 0.1, value = cell_E_V,
    #                         format = '%.1f',
    #                 help = '''Check the box above to set the full-cell voltage. No underlying voltage model will be used. 
    #                 This means that current and voltage have no physical relationship.
    #                   \n Default cell voltage: {} V'''.format(default_cell_E_V),
    #                 disabled = not override_cell_voltage)
    #     if override_cell_voltage:
    #         st.write('*Using specified cell voltage*')
    #     else:
    #         st.write('*Modeling cell voltage from simplified Butler-Volmer model*')
    #         override_eta_cat = st.checkbox('Specify cathode (CO$_2$R) overpotential', value = False)
    #         BV_eta_cat_V = st.slider(label = 'Cathodic overpotential (V)',
    #                         min_value = -10.0, 
    #                         max_value = 0.0, 
    #                         step = 0.1, value = BV_eta_cat_V,
    #                         format = '%.1f',
    #                         disabled = not override_eta_cat,
    #                         help = '''Check the box above to set the cathodic overpotential. 
    #                         Thermodynamics, cell resistance and anodic overpotential will be modeled. 
    #                         Note that more negative overpotentials indicate slower kinetics.
    #                           \n Default cathodic overpotential: {} V'''.format(default_BV_eta_cat_V),)
    #         if override_eta_cat:
    #             st.write('*Using manually specified cathodic overpotential*')
    #         else:
    #             st.write('*Modeling cathodic overpotential from Tafel slope, {:.0f} mV/dec*'.format(cat_Tafel_slope))

    #         override_eta_an = st.checkbox('Specify anode (oxidation reaction) overpotential', value = False)
    #         BV_eta_an_V = st.slider(label = 'Anodic overpotential (V)',
    #                         min_value = 0.0, 
    #                         max_value = 10.0, 
    #                         step = 0.1, value = BV_eta_an_V,
    #                         format = '%.1f',
    #                         disabled = not override_eta_an,
    #                         help = '''Check the box above to set the anodic overpotential. Thermodynamics, cell resistance and cathodic overpotential will be modeled. 
    #                           \n Default anodic overpotential: {} V'''.format(default_BV_eta_an_V),) 
    #         if override_eta_an:
    #             st.write('*Using manually specified anodic overpotential*')
    #         else:
    #             st.write('*Modeling anodic overpotential from Tafel slope, {:.0f} mV/dec*'.format(an_Tafel_slope))
            
            override_ohmic = st.checkbox('Specify full-cell area-specific resistance', value = False)
            R_ohmcm2 = st.slider(label = 'Area-specific resistance ($ \Omega \cdot$ cm$^2$)',
                            min_value = 0.0, 
                            max_value = 25.0, 
                            step = 0.1, value = R_ohmcm2,
                            format = '%.2f',
                            disabled= not override_ohmic,
                            help = '''Check the box above to set the area-specific ohmic resistance of the cell. Thermodynamics and kinetic overpotentials will be modeled. 
                              \n Default specific resistance: {} $ \Omega \cdot$ cm$^2$'''.format(default_R_ohmcm2),)
            if override_ohmic:
                st.write('*Using manually specified cell specific resistance*')
            else:
                st.write('*Using default cell resistance for anion exchange membrane electrode assembly, {} $ \Omega \cdot$ cm$^2$*'.format(R_ohmcm2))

with st.sidebar:
    st.subheader('Electrolyzer operation')
    with st.expander(label = '**Reactor model**', expanded = False):
        j_total_mA_cm2 = st.slider(label = 'Total current density (mA/cm$^2$)',
                            min_value = 0.0001, 
                            max_value = 1500.0, 
                            step = 1.0, value = j_total_mA_cm2,
                            format = '%.0f',
                            help = '''Total current density of the cell. This will determine the size and voltage of the cell.
                              \n Default total current density: {} mA/cm$^2$'''.format(default_j_total_mA_cm2),)

        ##### FE-SPC TRADEOFF  
        option_1 = 'Plug flow in gas channel'
        option_2 = 'Carbonate electrolyte supports CO$_2$ availability'
        option_3 = 'Manually specify $ FE_{{{}}}$ and single-pass conversion'.format(product_name)
        st.write('Model for selectivity tradeoff versus single-pass conversion')
        answer = st.radio(label = 'FE-SPC model',
                options = [option_1,
                        option_2, 
                        option_3 ],
                index = 0,
                label_visibility= 'collapsed')
        if answer == option_1:
            model_FE = 'Hawks'
            st.write('*Using [Hawks and Baker model](https://doi.org/10.1021/acsenergylett.2c01106) for $ FE_{}$ - $ X_{CO_2}$ tradeoff*')
            st.latex(r"""
                    \scriptsize \implies \displaystyle \frac{FE_{CO_2R}}{FE_{CO_2R, \: 0}} + \displaystyle \frac{-X_{CO_2} \cdot (1 + \frac{c \cdot n_{i}}{z_i \cdot FE_{CO_2R}})}{\ln(1-X_{CO_2} \cdot (1 + \frac{c \cdot n_{i}}{z_i \cdot FE_{CO_2R}}))} = 0
                    """)
        elif answer == option_2:
            model_FE = 'Kas'
            st.write('*Using [Kas and Smith model](https://doi.org/10.1021/acssuschemeng.0c07694) for $ FE_{}$ - $ X_{CO_2}$ tradeoff*')
            st.latex(r"""
                    \footnotesize \implies FE_{CO_2R} = FE_{CO_2R, \: 0} - 4.7306 \cdot {X_{CO_2}}^{5.4936}
                    """)
        elif answer == option_3: # TODO I don't think this works rn?
            model_FE = None
            FE_product_specified = st.slider(label = 'FE$_{{{}}}$'.format(product_name),
                    min_value = 0.0001,
                    max_value = 1.0,
                    step = 0.01, value = FE_CO2R_0,
                    help = 'Faradaic efficiency, independent of any other variables. \
                        This is not a recommended or default option, since it neglects electrolyzer geometry.\
                        Therefore, it artificially lowers costs.')

        FE_CO2R_0 = st.slider(label = '$ FE_{CO_2R, \: 0}$, maximum Faradaic efficiency',
                            min_value = 0.0001, 
                            max_value = 1.0, 
                            step = 0.01, value = FE_CO2R_0,
                            format = '%.2f',
                            help = r'''Maximum Faradaic efficiency achieved in the limit of 0 single-pass conversion or vast excess of CO$_2$,
                            $$$
                            lim_{{X_{{CO_2}} → 0}} FE_{{CO_2R}}
                            $$$
                            ''' +   '\n  Default $ FE_{{CO_2R, \: 0}}$: {}'.format(default_FE_CO2R_0),
                            disabled = answer == option_3)
        SPC = st.slider(label = 'Single-pass conversion',
                            min_value = 0.0001, 
                            max_value = 1.0, 
                            step = 0.01, value = SPC,
                            format = '%.2f')        
        crossover_ratio = st.slider(label = 'Crossover ratio (mol CO$_2$/mol e$^-$)',
                            min_value = 0.0001, 
                            max_value = 1.0, 
                            step = 0.01, value = crossover_ratio,
                            help = """The amount of CO$_2$ converted into carbonate ions that then crosses the membrane into the anode gas stream. 
                              \n Default crossover ratio: 0.5
                              \n This is based on the carbonate equilibrium: 
                            $$$
                            \\\  CO_2 + H_2O + 2e^- → CO + 2OH^-  
                            \\\  2CO_2 + 8H_2O + 12e^- → C_2H_4 + 12OH^-
                            \\\  2H_2O + 2e^- → H_2 + 2OH^-
                            \\\   CO_{{2}} + 2OH^{{-}} → HCO_{{3}}^{{-}} + OH^- ⇌ CO_{{3}}^{{2-}} + H_2O 
                            $$$
                            """,
                            format = '%.2f')
        FE_product_checked, __ = SPC_check(FE_product_specified=FE_product_specified, 
                exponent= exponent,
                scaling = scaling,
                SPC = SPC,
                j_total= j_total_mA_cm2,
                FE_CO2R_0= FE_CO2R_0,
                product_name=product_name,
                model_FE= model_FE,
                df_products=df_products,
                crossover_ratio=crossover_ratio)
        if not np.isnan(FE_product_checked): 
            st.session_state.is_active_error_ethylene = False
        else:
            st.session_state.is_active_error_ethylene = True
            st.header(':red[Model is physically unviable. Please check $ FE_{CO_2R, \: 0}$,  $ X_{CO_2}$ and crossover ratio.]')    
        st.latex(r'''
                 \footnotesize \implies  FE_{{{}}} = {:.2f}
                 '''.format(product_name, FE_product_checked))

with st.sidebar:
    st.subheader('Process design')
    with st.expander(label = '**Plant and separation parameters**', expanded = False):
        product_rate_kg_day = 1000 * st.slider(label = '{} production rate (ton/day)'.format(product_name),
                            min_value = 0.0001 / 1000, 
                            max_value = 1.5e6 / 1000, 
                            step = 10.0, value = product_rate_kg_day / 1000,
                            format = '%.0f',
                            help = '''Daily production rate. This is fixed for the entire plant lifetime and sets the total CO$_2$R current required.
                              \n Default value: {} kg$ _{{{}}}$/day
                            '''.format(product_rate_kg_day, product_name))
        capacity_factor = st.slider(label = 'Capacity factor (days per 365 days)',
                            min_value = 0.0001, 
                            max_value = 1.0, 
                            step = 0.01, value = capacity_factor,
                            format = '%.2f',
                            help = '''Fraction of time per year that the plant is operational.
                              \n Default value: {:.2f}, based on 350/365 days per year
                            '''.format(default_capacity_factor))
        lifetime_years = st.slider(label = 'Plant lifetime (years)',
                            min_value = 0.0001, 
                            max_value = 100.0, 
                            step = 1.0, value = lifetime_years,
                            format = '%.0f',
                            help = '''Plant lifetime in years. The process operates to produce {} kg/day of product for this many years.
                              \n Default value: {} years
                            '''.format(product_rate_kg_day, lifetime_years))
        stack_lifetime_years = st.slider(label = 'Stack lifetime (years)',
                            min_value = 0.0001, 
                            max_value = 30.0, 
                            step = 1.0, value = stack_lifetime_years,
                            format = '%.0f',
                            help = '''Stack replacement time in years. The entire electrolyzer must be replacemed at this interval.
                              \n Default value: {} years
                            '''.format(stack_lifetime_years))
        PSA_second_law_efficiency = st.slider(label = 'Second-law separation efficiency',
                            min_value = 0.0001, 
                            max_value = 1.0, 
                            step = 0.01, value = PSA_second_law_efficiency,
                            format = '%.2f',
                            help = '''Second-law efficiency of gas separations between CO$_2$/O$_2$, CO$_2$/CO, and CO/H$_2$.
                            This adjusts the ideal work of binary separation, 
                            $$$
                             \\\  W_{{sep \: (j)}}^{{ideal}} = R \cdot T \cdot (\sum_i x_i\cdot ln(x_i)) \cdot \displaystyle \dot{{N}} \\
                              \\\  \implies W = R \cdot T \cdot (x_i\cdot ln(x_i)) + (1-x_i)\cdot ln(1-x_i)) \cdot \displaystyle \dot{{N}} \\
                              \\\  W_{{sep \: (j)}}^{{real}} = \displaystyle \\frac{{W_{{sep\: (j)}}^{{ideal}}}}{{\zeta}}
                            $$$
                              \n Default value: {}
                            '''.format(default_PSA_second_law_efficiency))

    ##### BATTERY  
    answer = st.toggle('Include energy storage', value = False,
                         help = 'Check this box to include utility-scale batteries. The cost of electricity should also be reduced to account for cheap, intermittent renewables like wind and solar. Battery capex can be adjusted in the Market variables section.')

    if answer:            
        # Handle battery to flatten curve and maximize capacity
        is_battery = True
        st.write('*Including utility-scale batteries*')
        avbl_renewables = st.slider(label = 'Minimum fraction of time when renewables power the electrolyzer',
                                    min_value = 0.0001, max_value = 1.000, 
                                    step = 0.01, value = 0.236,
                                    format = '%.2f',
                                    help = '''Fraction of time per day that renewable power is available. 
                                    The battery size will be large enough to make the plant capacity factor as listed below, i.e. battery size depends on the difference
                                    between the capacity factor of available electricity versus the capacity factor of the plant.
                                    \n Default value: {:.2f}, based on 5.6/24 hours per day
                                    '''.format(0.236))
        avbl_renewables = float(avbl_renewables)
    else:
        is_battery = False
        battery_capacity = 0

with st.sidebar:
    st.subheader('Market variables')
    with st.expander(label = '**Capital and operating costs**', expanded = False):
        electricity_cost_USD_kWh = st.slider(label = 'Electricity cost ($/kWh)',
                            min_value = 0.0, 
                            max_value = 0.25, 
                            step = 0.01, value = electricity_cost_USD_kWh,
                            format = '%.2f',
                            help = '''Electricity cost.
                            \n Default value: \${}/kWh, based on average retail industrial electricity cost in April 2023 in the United States.
                            '''.format(default_electricity_cost_USD_kWh))
        CO2_cost_USD_tCO2 = st.slider(label = 'CO$_2$ cost (\$/t CO$_2$)',
                            min_value = 0.0, 
                            max_value = 500.0, 
                            step = 1.0, value = CO2_cost_USD_tCO2,
                            format = '%.0f',
                            help = '''Default value: \${}/t$_{{CO_2}}$
                            '''.format(default_CO2_cost_USD_tCO2))
        H2_cost_USD_tCO2 = st.slider(label = 'H$_2$ cost (\$/kg H$_2$)',
                            min_value = 0.0, 
                            max_value = 5.0, 
                            step = 0.1, value = H2_cost_USD_kgH2,
                            format = '%.1f',
                            help = '''Default value: \${}/kg
                            '''.format(default_H2_cost_USD_kgH2))
        product_cost_USD_kgprod = st.slider(label = '{} market price (\$/kg {})'.format(product_name, product_name),
                            min_value = 0.0, 
                            max_value = 10.0, 
                            step = 0.1, value = product_cost_USD_kgprod,
                            format = '%.1f',
                            help = '''Default value: \${}/kg
                            '''.format(default_product_cost_USD_kgprod))
        water_cost_USD_kg = st.slider(label = 'Water cost (\$/kg)',
                            min_value = 0.0, 
                            max_value = 1.0, 
                            step = 0.1, value = water_cost_USD_kg,
                            format = '%.1f',
                            help = '''Default value: \${}/kg
                            '''.format(default_water_cost_USD_kg))
        electrolyzer_capex_USD_m2 = st.slider(label = 'Electrolyzer capital cost (\$/m$^2$)' , 
                            min_value = 0.0, 
                            max_value = 15000.0, 
                            step = 100.0, value = electrolyzer_capex_USD_m2,
                            format = '%.0f',
                            help = '''Default value: \${}/m$^2$
                            '''.format(default_electrolyzer_capex_USD_m2))
        battery_capex_USD_kWh = st.slider(label = 'Battery capital cost (\$/kWh)' , 
                            min_value = 0.0, 
                            max_value = 500.0, 
                            step = 1.0, value = battery_capex_USD_kWh,
                            format = '%.0f', disabled = not is_battery,
                            help = '''Default value: \${}/kWh, based on 4-hour storage.
                            '''.format(default_battery_capex_USD_kWh))
        
    ##### CUSTOM CAPEX OR OPEX  
    answer = st.toggle('Add custom capex', value = False,
                         help = 'Optional bare-module capital cost for any custom units')
    if answer:            
        # Handle battery to flatten curve and maximize capacity
        is_additional_capex = True
        additional_capex_USD = st.slider(label = 'Additional capital cost (\$ million)' ,
                            min_value = 0.0, 
                            max_value = 100.0, 
                            step = 1.0, value = 0.0,
                            format = '%.0f', disabled = not is_additional_capex,
                            help = '''Optional additional capex. Default value: \${} million.
                            '''.format(0)) *1e6
    else:
        is_additional_capex = False
        additional_capex_USD = 0
        
    answer = st.toggle('Add custom opex', value = False,
                         help = 'Optional operating cost')
    if answer:            
        # Handle battery to flatten curve and maximize capacity
        is_additional_opex = True
        additional_opex_USD_kg = st.slider(label = 'Additional operating cost (\$/kg {})'.format(product_name),  
                            min_value = 0.0, 
                            max_value = 5.0, 
                            step = 0.1, value = 0.0,
                            format = '%.1f', disabled = not is_additional_opex,
                            help = '''Optional operating cost for any custom expenses. Convert daily costs to \$/kg product as follows:
                            $$$
                            \\\ \\frac{{\$ opex}}{{year}} = \\frac{{\$ opex}}{{day}} \cdot CF \cdot 365 \cdot plant lifetime
                            $$$
                            ''')    
    else:
        is_additional_opex = False
        additional_opex_USD_kg = 0 
        
with st.sidebar:
    st.subheader('Emissions assessment')
    electricity_emissions_kgCO2_kWh = st.slider(label = 'Grid CO$_2$ intensity (kg$_{CO_2}$/kWh)',
                        min_value = 0.0, 
                        max_value = 1.0, 
                        step = 0.01, value = electricity_emissions_kgCO2_kWh,
                        format = '%.2f',
                        help = '''Electricity emissions for partial life-cycle assessment.
                        \n Default value: {:.2f} kg$_{{CO_2}}$/kWh, based on the United States average.
                        '''.format(default_electricity_emissions_kgCO2_kWh))


#___________________________________________________________________________________

##########################  RUN MODEL AT DEFAULT VALUES  ###########################

# capex_default, opex_default, levelized_default, potential_default, energy_default, emissions_default = default_single_run(product_name = product_name, 
#                         product_rate_kg_day = default_product_rate_kg_day, 
#                         df_products = df_products, FE_CO2R_0 = default_FE_CO2R_0, 
#                         FE_product_specified = default_FE_product_specified, 
#                         j_total_mA_cm2 = default_j_total_mA_cm2,SPC = default_SPC, 
#                         crossover_ratio = default_crossover_ratio, model_FE = 'Hawks',  
#                         overridden_vbl = '', overridden_value = np.NaN, overridden_unit = '', 
#                         override_optimization =  override_optimization, P = default_P, T_streams = default_T_streams, 
#                         R_ohmcm2 = default_R_ohmcm2, an_E_eqm = default_an_E_eqm, MW_CO2 = MW_CO2, 
#                         MW_H2O = MW_H2O, MW_O2 = MW_O2,  MW_MX = MW_K2CO3,
#                         cathode_outlet_humidity = default_cathode_outlet_humidity,
#                         excess_water_ratio = default_excess_water_ratio,
#                         an_eta_ref = default_an_eta_ref, 
#                         an_Tafel_slope = default_an_Tafel_slope, 
#                         an_j_ref = default_an_j_ref, 
#                         electricity_emissions_kgCO2_kWh = default_electricity_emissions_kgCO2_kWh,
#                         heat_emissions_kgCO2_kWh = default_heat_emissions_kgCO2_kWh,
#                         electrolyte_conc = default_electrolyte_conc, 
#                         density_kgm3 = default_density_kgm3,
#                         PSA_second_law_efficiency = default_PSA_second_law_efficiency, 
#                         T_sep = T_sep, electricity_cost_USD_kWh = default_electricity_cost_USD_kWh, 
#                         heat_cost_USD_kWh = default_heat_cost_USD_kWh,product_cost_USD_kgprod = default_product_cost_USD_kgprod,
#                         H2_cost_USD_kgH2 = default_H2_cost_USD_kgH2,water_cost_USD_kg = default_water_cost_USD_kg,
#                         CO2_cost_USD_tCO2 = default_CO2_cost_USD_tCO2,lifetime_years = default_lifetime_years,
#                         stack_lifetime_years = stack_lifetime_years,
#                         is_additional_capex = False, is_additional_opex = False,
#                         additional_capex_USD = 0, additional_opex_USD_kg = 0,
#                         electrolyzer_capex_USD_m2 = default_electrolyzer_capex_USD_m2,
#                         capacity_factor = default_capacity_factor,battery_capex_USD_kWh = default_battery_capex_USD_kWh,               
#                         battery_capacity = default_battery_capacity, exponent=default_exponent, scaling=default_scaling,
#                         carbon_capture_efficiency = default_carbon_capture_efficiency,
#                         R = R,
#                         F = F) 

#___________________________________________________________________________________
    
##########################  GENERATE MODEL OVER RANGE  ##########################
with middle_column:
    if not st.session_state.is_active_error_ethylene:
        progress_bar = st.progress(0, text= "Running model over range. Please wait.")

        ### Generate modeling results for variable range 

        ### Create empty variables
        # # Storage for individual runs
        # dict_stream_tables = {}
        # dict_econ_tables = {}

        # Storage for summary results across varied independent variable
        df_potentials_vs_vbl = pd.DataFrame()
        df_energy_vs_vbl = pd.DataFrame()
        df_emissions_vs_vbl = pd.DataFrame()
        df_electrolyzer_assumptions_vs_vbl = pd.DataFrame()
        df_outlet_assumptions_vs_vbl = pd.DataFrame()
        df_opex_vs_vbl = pd.DataFrame()
        df_opex_totals_vs_vbl = pd.DataFrame()
        df_capex_BM_vs_vbl = pd.DataFrame()
        df_capex_totals_vs_vbl = pd.DataFrame()
        df_costing_assumptions_vs_vbl = pd.DataFrame()
        df_sales_vs_vbl = pd.DataFrame()

        #### Loop through variable
        for i, vbl in enumerate(vbl_range):
            progress_bar.progress(i/(len(vbl_range)+1), text = "Running model at {} = {} {}  \n Please wait.".format(vbl_name, vbl, vbl_unit))

            ### Update variable in its place
            if vbl_name != 'Cell voltage' and vbl_name != 'Cathodic overpotential' and vbl_name != 'Anodic overpotential':
                value_original = globals()[df_flags.loc[vbl_name,'Python variable']] # Save the original value of the adjusted variable
                globals()[df_flags.loc[vbl_name,'Python variable']] = vbl # Overwrite the global variable whose override flag is True

            ### Generate physical and costing model
            
            # Handle battery to flatten curve
            if is_battery:
                battery_capacity = 1 - avbl_renewables # assumes daily storage battery
                capacity_factor = 350/365 # capacity is re-maximized
            else:
                battery_capacity = 0
            
            ### Generate physical and costing model
            df_capex_BM, df_capex_totals, df_costing_assumptions, df_depreciation, df_electrolyzer_assumptions, df_electrolyzer_streams_mol_s,\
                df_energy, df_feedstocks, df_general, df_maintenance, df_operations, df_opex, df_opex_totals, df_outlet_assumptions,\
                df_overhead, df_potentials, df_sales, df_streams, df_streams_formatted, df_taxes, df_utilities = cached_single_run(product_name = product_name, 
                                                                            product_rate_kg_day = product_rate_kg_day, 
                                                                            df_products = df_products, FE_CO2R_0 = FE_CO2R_0, 
                                                                            FE_product_specified = FE_product_specified, 
                                                                            j_total_mA_cm2 = j_total_mA_cm2,SPC = SPC, 
                                                                            crossover_ratio = crossover_ratio, 
                                                                            model_FE = model_FE, 
                                                                            is_additional_capex = is_additional_capex,
                                                                            is_additional_opex = is_additional_opex,
                                                                            additional_opex_USD_kg = additional_opex_USD_kg,
                                                                            additional_capex_USD = additional_capex_USD,
                                                                            overridden_vbl = vbl_name, 
                                                                            overridden_value = vbl, overridden_unit = vbl_unit, 
                                                                            override_optimization =  override_optimization, 
                                                                            P = P, T_streams = T_streams, R_ohmcm2 = R_ohmcm2, 
                                                                            an_E_eqm = an_E_eqm,
                                                                            an_eta_ref = an_eta_ref,
                                                                            an_Tafel_slope = an_Tafel_slope,
                                                                            an_j_ref = an_j_ref,
                                                                            MW_CO2 = MW_CO2, MW_H2O = MW_H2O, 
                                                                            MW_O2 = MW_O2,  MW_MX = MW_K2CO3,
                                                                            cathode_outlet_humidity = cathode_outlet_humidity,
                                                                            excess_water_ratio = excess_water_ratio,
                                                                            electrolyte_conc = electrolyte_conc, 
                                                                            density_kgm3 = density_kgm3,
                                                                            PSA_second_law_efficiency = PSA_second_law_efficiency, 
                                                                            T_sep = T_sep, 
                                                                            electricity_cost_USD_kWh = electricity_cost_USD_kWh, 
                                                                            heat_cost_USD_kWh = heat_cost_USD_kWh,
                                                                            electricity_emissions_kgCO2_kWh = electricity_emissions_kgCO2_kWh,
                                                                            heat_emissions_kgCO2_kWh = heat_emissions_kgCO2_kWh,
                                                                            product_cost_USD_kgprod = product_cost_USD_kgprod,
                                                                            H2_cost_USD_kgH2 = H2_cost_USD_kgH2,
                                                                            water_cost_USD_kg = water_cost_USD_kg,
                                                                            CO2_cost_USD_tCO2 = CO2_cost_USD_tCO2,
                                                                            electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2,
                                                                            lifetime_years = lifetime_years,
                                                                            stack_lifetime_years = stack_lifetime_years,
                                                                            capacity_factor = capacity_factor,
                                                                            battery_capex_USD_kWh = battery_capex_USD_kWh,               
                                                                            battery_capacity = battery_capacity,
                                                                            carbon_capture_efficiency = carbon_capture_efficiency,
                                                                            exponent=exponent, 
                                                                            scaling=scaling,
                                                                            R = R,
                                                                            F = F)

            ### Store results of models                             
            # dict_stream_tables[vbl] = {
            #     df_streams_formatted.index.name: df_streams_formatted, 
            #     df_electrolyzer_assumptions.index.name: df_electrolyzer_assumptions, 
            #     df_outlet_assumptions.index.name: df_outlet_assumptions, 
            #     df_energy.index.name: df_energy,
            #     df_potentials.index.name: df_potentials
            #     }
            
            # dict_econ_tables[vbl] = {
            #         df_costing_assumptions.index.name: df_costing_assumptions, 
            #         df_capex_BM.index.name: df_capex_BM,
            #         df_capex_totals.index.name: df_capex_totals, 
            #         df_opex.index.name: df_opex, 
            #         df_opex_totals.index.name: df_opex_totals, 
            #         df_sales.index.name: df_sales, 
            #         df_feedstocks.index.name: df_feedstocks, 
            #         df_utilities.index.name: df_utilities, 
            #         df_operations.index.name: df_operations, 
            #         df_maintenance.index.name: df_maintenance, 
            #         df_overhead.index.name: df_overhead,
            #         df_taxes.index.name: df_taxes, 
            #         df_depreciation.index.name: df_depreciation, 
            #         df_general.index.name: df_general,
            #     }
            
            df_potentials_vs_vbl = pd.concat([df_potentials_vs_vbl, 
                                                df_potentials['Value']], axis = 1)  # Store cell voltages
            df_energy_vs_vbl = pd.concat([df_energy_vs_vbl, 
                                        df_energy['Energy (kJ/kg {})'.format(product_name)]], axis = 1) # Store energy utility for plotting
            df_emissions_vs_vbl = pd.concat([df_emissions_vs_vbl, 
                                                pd.concat([df_energy['Emissions (kg CO2/kg {})'.format(product_name)], pd.Series(df_outlet_assumptions.loc['Carbon capture loss', 'Value']) ] ) ], 
                                            axis = 1) # Store emissions for plotting
            df_electrolyzer_assumptions_vs_vbl = pd.concat([df_electrolyzer_assumptions_vs_vbl, 
                                                            df_electrolyzer_assumptions['Value']], axis = 1) # Store assumptions
            df_outlet_assumptions_vs_vbl = pd.concat([df_outlet_assumptions_vs_vbl, 
                                                        df_outlet_assumptions['Value']], axis = 1) # Store assumptions
            df_opex_vs_vbl = pd.concat([df_opex_vs_vbl, 
                                        df_opex['Cost ($/kg {})'.format(product_name)]], axis = 1) # Store opex for plotting
            df_opex_totals_vs_vbl = pd.concat([df_opex_totals_vs_vbl, 
                                        df_opex_totals['Cost ($/kg {})'.format(product_name)]], axis = 1) # Store opex for plotting
            df_capex_BM_vs_vbl = pd.concat([df_capex_BM_vs_vbl, 
                                        df_capex_BM['Cost ($)']], axis = 1) # Store capex for plotting
            df_capex_totals_vs_vbl = pd.concat([df_capex_totals_vs_vbl, 
                                        df_capex_totals['Cost ($)']], axis = 1) # Store capex for plotting
            df_costing_assumptions_vs_vbl = pd.concat([df_costing_assumptions_vs_vbl, 
                                        df_costing_assumptions['Cost']], axis = 1) # Store costing assumptions for plotting
            df_sales_vs_vbl = pd.concat([df_sales_vs_vbl, 
                                    df_sales['Earnings ($/yr)']], axis = 1) # Store costing assumptions for plotting
  
            ### Adjust FE_product, SPC, capacity_factor and variable back to their original values in globals()
            if vbl_name != 'Cell voltage' and vbl_name != 'Cathodic overpotential' and vbl_name != 'Anodic overpotential':
                globals()[df_flags.loc[vbl_name,'Python variable']] = value_original

            ### End loop through variable value

        progress_bar.empty()

        # Format completed "summary" dataframes

        for df in [df_energy_vs_vbl, df_potentials_vs_vbl,  df_emissions_vs_vbl,
                        df_electrolyzer_assumptions_vs_vbl, df_outlet_assumptions_vs_vbl, df_costing_assumptions_vs_vbl, 
                        df_capex_BM_vs_vbl, df_capex_totals_vs_vbl, df_opex_vs_vbl, df_opex_totals_vs_vbl, df_sales_vs_vbl,
                        ]:
            df.columns = vbl_range_text # rename columns

        ## Rename index (rows) for collected dataframes; columns will be renamed in the next section
        ## Add in units column but only after duplicating the df, otherwise indexing for plots is very complicated
        df_electrolyzer_assumptions_vs_vbl.index = df_electrolyzer_assumptions.index
        df_electrolyzer_assumptions_vs_vbl_2 = df_electrolyzer_assumptions_vs_vbl.copy()
        df_electrolyzer_assumptions_vs_vbl_2.insert(0, 'Units', df_electrolyzer_assumptions['Units'])

        df_outlet_assumptions_vs_vbl.index = df_outlet_assumptions.index
        df_outlet_assumptions_vs_vbl_2 = df_outlet_assumptions_vs_vbl.copy()
        df_outlet_assumptions_vs_vbl_2.insert(0, 'Units', df_outlet_assumptions['Units'])

        df_costing_assumptions_vs_vbl.index = df_costing_assumptions.index
        df_costing_assumptions_vs_vbl_2 = df_costing_assumptions_vs_vbl.copy()
        df_costing_assumptions_vs_vbl_2.insert(0, 'Units', df_costing_assumptions['Units'])

        df_potentials_vs_vbl.index = df_potentials.index
        df_potentials_vs_vbl_2 = df_potentials_vs_vbl.copy()
        df_potentials_vs_vbl_2.insert(0, 'Units', df_potentials['Units'])

        df_opex_vs_vbl.index = df_opex.index
        df_opex_vs_vbl_2 = df_opex_vs_vbl.copy()
        df_opex_vs_vbl_2.insert(0, 'Units', '$/kg {}'.format(product_name))

        df_sales_vs_vbl.index = df_sales.index
        df_sales_vs_vbl_2 = df_sales_vs_vbl.copy()
        df_sales_vs_vbl_2.insert(0, 'Units', '$/yr')

        df_capex_BM_vs_vbl.index = df_capex_BM.index
        df_capex_BM_vs_vbl_2 = df_capex_BM_vs_vbl.copy()
        df_capex_BM_vs_vbl_2.insert(0, 'Units', '$')

        df_opex_totals_vs_vbl.index = df_opex_totals.index
        df_opex_totals_vs_vbl_2 = df_opex_totals_vs_vbl.copy()
        df_opex_totals_vs_vbl_2.insert(0, 'Units', '$')

        df_capex_totals_vs_vbl.index = df_capex_totals.index
        df_capex_totals_vs_vbl_2 = df_capex_totals_vs_vbl.copy()
        df_capex_totals_vs_vbl_2.insert(0, 'Units', '$')

        df_energy_vs_vbl.index = df_energy_vs_vbl.index    
        df_energy_vs_vbl_2 = df_energy_vs_vbl.copy()    
        df_energy_vs_vbl_2.index.name  = 'Energy'
        df_energy_vs_vbl_2.insert(0, 'Units', 'kJ/kg {}'.format(product_name))

        df_emissions_vs_vbl.index = np.append(df_energy.index, 'Carbon capture')
        df_emissions_vs_vbl_2 = df_emissions_vs_vbl.copy()  
        df_emissions_vs_vbl_2.index.name  = 'Emissions'
        df_emissions_vs_vbl_2.insert(0, 'Units', 'kg CO2/kg {}'.format(product_name))

        if df_capex_BM_vs_vbl.isnull().values.all():
            st.session_state.is_active_error_ethylene = True
            st.header('*:red[There is an error in the model inputs. Please check the sidebar for error messages.]*')
        else:
            st.session_state.is_active_error_ethylene = False

    else:
        st.header('*:red There is an error in the model inputs. Please check the sidebar for error messages.*')

#___________________________________________________________________________________

############################### FIGURE OUTPUTS #####################################

if not st.session_state.is_active_error_ethylene:
    ###### BAR CHART FORMATTING
    flag = {True: 1,
            False: 0}
    ### Define colors
    BM_capex_colors = [bright_summer_r(i) for i in np.linspace(0, 1, len(df_capex_BM_vs_vbl.index)- flag[is_battery] - flag[is_additional_capex])] # battery gets its own color, so 1 less than capex length for other units
    if is_battery:
        BM_capex_colors.append((0.65, 0.65, 0.65, 1)) # add in battery         
    if is_additional_capex:
        BM_capex_colors.append((0.85, 0.85, 0.85, 1)) # add in battery 
        
    # Blues from 0.2 to 1 not bad but low contrast; YlGnbu not but looks jank with PuOr; winter_r is best
    # Opex colors
    opex_colors = [diverging(i) for i in np.linspace(0, 0.85, len(df_opex_vs_vbl.index) - flag[is_additional_opex]) ]
    if is_additional_opex:
        opex_colors.append((0.85, 0.85, 0.85, 1)) # add in battery 

    # Emissions colors
    emissions_colors = [RdYlBu(i) for i in np.linspace(0, 1, sum(~df_emissions_vs_vbl.T.isnull().all()) - 1 )  ] # len(df_energy_vs_vbl.index) - 2)] # last rows are totals

    # Potentials colors
    potentials_colors = [RdYlBu(i) for i in np.linspace(0, 1, np.shape(df_potentials_vs_vbl.iloc[2:7])[0] )  ] # last rows are totals

    # Energy colors
    energy_colors = emissions_colors # [RdYlBu(i) for i in np.linspace(0, 1, sum(~df_energy_vs_vbl.iloc[:-3].T.isnull().all())  )  ] # len(df_energy_vs_vbl.index) - 2)] # last rows are totals

    
    x_axis_major_ticks = x_axis_formatting(x_axis_min, x_axis_max, x_axis_num)

    if vbl_unit == '':
        x_axis_label = vbl_name
    else:
        x_axis_label = '{} ({})'.format(vbl_name, vbl_unit)

    barwidth = 1/(1.5*len(vbl_range)) * (x_axis_max - x_axis_min)
        # To get a target of 1/(2.5) spacing, calculate linewidth, which will be drawn on top of bars
        #linewidth_calc = (0.2*barwidth) * (0.8 / 1) * (mp.rcParams['figure.figsize'][0]/(x_axis_max - x_axis_min)) * 72 # convert barwidth fraction in x-data units into points
        # Fix linewidth to be uniform in all plots
    linewidth_calc = 1.4 # 1.4769 

    ###### CAPEX BAR CHART
    with middle_column.container(height = 300, border = False):  
        with _render_lock:
            capex_bar_fig, axs = plt.subplots() # Set up plot
            #fig.subplots_adjust(left=0.9, bottom=0.9, right=1, top=1, wspace=None, hspace=None)

            y_axis_major_ticks = y_axis_formatting(y_axis_min_capex, y_axis_max_capex, y_axis_num_capex)

            ## Axis labels
            axs.set_ylabel('Capital cost (million \$)')
            axs.set_xlabel(x_axis_label)

            ## Draw axis ticks
            axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            axs.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            plt.xticks(x_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])
            plt.yticks(y_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])

            ## Apply axis limits
            axs.set_xlim([x_axis_min,x_axis_max])
            axs.set_ylim([y_axis_min_capex,y_axis_max_capex])

            ## Plot series
            if df_flags.loc['Capacity factor', 'T/F?'] :
                axs.plot([0.23625,0.23625], [y_axis_min_capex, y_axis_max_capex], alpha = 1,
                    c = theme_colors[6]) # Plot line for cost 
                axs.text(0.23625, y_axis_min_capex + (y_axis_max_capex - y_axis_min_capex)*0.025, 'Solar capacity', ha='right', va='bottom', #  (5.67 h/day)
                    fontsize = SMALL_SIZE, rotation = 90)
            if df_flags.loc['{} production rate'.format(product_name), 'T/F?'] == True:
                axs.text(x_axis_max*0.1, y_axis_max_capex*0.975, 'Lifetime sales', #.format(product_cost_USD_kgprod*product_rate_kg_day*capacity_factor*365*20/1e6), 
                        ha='left', va='top', 
                    fontsize = SMALL_SIZE)
                axs.plot(vbl_range, df_sales_vs_vbl.loc['Total']*lifetime_years,
                    alpha = 1, c = theme_colors[6]) # Plot line for cost 
                plt.xticks(rotation=35)  # Rotate text labels
            else:
                axs.text(x_axis_max*0.975, y_axis_max_capex*0.975, 'Lifetime sales > ${:.0f} million'.format(min(df_sales_vs_vbl.loc['Total', df_sales_vs_vbl.loc['Total'] > 0]*lifetime_years)/1e6), ha='right', va='top', 
                    fontsize = SMALL_SIZE)
            
            cumsum = 0
            for i, category in enumerate(df_capex_BM_vs_vbl.index):
                axs.bar(vbl_range, df_capex_BM_vs_vbl.fillna(0).loc[category]/1e6, label=category , bottom = cumsum, width = barwidth, color = BM_capex_colors[i],
                    edgecolor = 'w', linewidth = linewidth_calc)
                cumsum += df_capex_BM_vs_vbl.fillna(0).loc[category]/1e6
            axs.plot(vbl_range[df_capex_totals_vs_vbl.loc['Total permanent investment'] > 0], 
                    df_capex_totals_vs_vbl.loc['Total permanent investment', df_capex_totals_vs_vbl.loc['Total permanent investment'] > 0]/1e6, # All capex except working capital, which is recovered during operation
                    label = 'Total permanent investment', alpha = 1, c = theme_colors[6]) # Plot line for Total permanent investment cost 

            ## Legend
            axs.legend(bbox_to_anchor=(1, 1), loc='upper left', reverse = True) # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position

            st.pyplot(capex_bar_fig, transparent = True, use_container_width = True)

    ###### OPEX BAR CHART 
    with right_column.container(height = 300, border = False): 
        with _render_lock:
            opex_bar_fig, axs = plt.subplots() # Set up plot

            y_axis_major_ticks = y_axis_formatting(y_axis_min_opex, y_axis_max_opex, y_axis_num_opex)

            ## Axis labels
            axs.set_ylabel('Operating cost \n (\$/kg$_{{{}}}$)'.format(product_name))
            axs.set_xlabel(x_axis_label)

            ## Draw axis ticks
            axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            axs.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            plt.xticks(x_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])
            plt.yticks(y_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])

            ## Apply axis limits
            axs.set_xlim([x_axis_min,x_axis_max])
            axs.set_ylim([y_axis_min_opex,y_axis_max_opex])
            
            ## Plot series
            axs.plot([x_axis_min, x_axis_max], 
                    [df_costing_assumptions_vs_vbl.loc[product_name][0],df_costing_assumptions_vs_vbl.loc[product_name][-1]], 
                    alpha = 1, c = theme_colors[6]) # Plot line for cost 
            axs.text(x_axis_max * 1.025, product_cost_USD_kgprod, 'Market price', ha='left', va='center', 
                    fontsize = SMALL_SIZE)
            
            cumsum = 0
            for i, category in enumerate(df_opex.index):
                axs.bar(vbl_range, df_opex_vs_vbl.loc[category], label=category , bottom = cumsum, width = barwidth, color = opex_colors[i],
                    edgecolor = 'w', linewidth = linewidth_calc)
                cumsum += df_opex_vs_vbl.loc[category]

            if df_flags.loc['Electricity cost', 'T/F?'] == True:
                axs.text(df_utility_imports.loc['Electricity - current US mix','Cost ($/kWh)'], 
                        y_axis_max_opex*0.975, 'U.S. average', ha='right', va='top', 
                    fontsize = SMALL_SIZE, rotation = 90)
                axs.plot([df_utility_imports.loc['Electricity - current US mix', 'Cost ($/kWh)'], df_utility_imports.loc['Electricity - current US mix', 'Cost ($/kWh)']], 
                        [y_axis_min_opex, y_axis_max_opex], alpha = 1,
                    c = theme_colors[6]) # Plot line for cost 
            
            if df_flags.loc['{} production rate'.format(product_name), 'T/F?'] == True:
                plt.xticks(rotation=35)  # Rotate text labels

            ## Legend
            axs.legend(bbox_to_anchor=(1.4, 1.1), loc='upper left', reverse = True) # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position
            
            st.pyplot(opex_bar_fig, transparent = True, use_container_width = True)   

    ###### LEVELIZED BAR CHART
    with middle_column.container(height = 300, border = False): 
        with _render_lock:
            levelized_bar_fig, axs = plt.subplots() # Set up plot
            #fig.subplots_adjust(left=0.9, bottom=0.9, right=1, top=1, wspace=None, hspace=None)

            y_axis_major_ticks = y_axis_formatting(y_axis_min_levelized, y_axis_max_levelized, y_axis_num_levelized)

            ## Axis labels
            axs.set_ylabel('Levelized cost \n (\$/kg$_{{{}}}$)'.format(product_name))
            axs.set_xlabel(x_axis_label)
            
            ## Draw axis ticks
            axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            axs.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            plt.xticks(x_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])
            plt.yticks(y_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])

            ## Apply axis limits
            axs.set_xlim([x_axis_min,x_axis_max])
            axs.set_ylim([y_axis_min_levelized,y_axis_max_levelized])

            ## Plot series
            axs.plot([x_axis_min, x_axis_max],
                    [df_costing_assumptions_vs_vbl.loc[product_name][0],df_costing_assumptions_vs_vbl.loc[product_name][-1]],
                    alpha = 1, 
                    c = theme_colors[6]) # Plot line for cost 
            axs.text(x_axis_max * 1.025, product_cost_USD_kgprod, 'Market price', ha='left', va='center', 
                    fontsize = SMALL_SIZE)
            
            # Additional lines
            if df_flags.loc['Capacity factor', 'T/F?'] == True: # or df_flags.loc['Renewables capacity factor', 'T/F?'] == True:
                axs.plot([0.23625,0.23625], [y_axis_min_levelized, y_axis_max_levelized], alpha = 1,
                    c = theme_colors[6]) # Plot line for solar 
                axs.text(0.23625, y_axis_max_levelized*0.975, 'Average solar', ha='right', va='top', 
                    fontsize = SMALL_SIZE, rotation = 90)
        #     axs.plot([0.0677,0.0677], [y_axis_min_levelized, y_axis_max_levelized], alpha = 1,
        #              c = theme_colors[6]) # Plot line for solar 
        #     axs.text(0.0677, y_axis_max_levelized*0.975, 'Texas (EIA, Jul 2023)', ha='right', va='top', 
        #               fontsize = SMALL_SIZE, rotation = 90)

            axs.plot(vbl_range[df_opex_totals_vs_vbl.loc['Levelized cost'] > 0], 
                    df_opex_totals_vs_vbl.loc['Levelized cost', df_opex_totals_vs_vbl.loc['Levelized cost'] > 0],
                    label = 'Levelized cost', alpha = 1, c = theme_colors[6]) # Plot line for total levelized cost
                    # Levelized cost includes all capex except working capital, which is recovered during operation
            
            cumsum = 0
            for i, category in enumerate(df_opex.index):
                axs.bar(vbl_range, df_opex_vs_vbl.fillna(0).loc[category], label=category , bottom = cumsum, width = barwidth, color = opex_colors[i],
                    edgecolor = 'w', linewidth = linewidth_calc)
                cumsum += df_opex_vs_vbl.fillna(0).loc[category]
                
            for i, category in enumerate(df_capex_BM_vs_vbl.index):      
                axs.bar(vbl_range, df_capex_BM_vs_vbl.fillna(0).loc[category]/(df_costing_assumptions_vs_vbl.loc['Plant lifetime']*365*df_costing_assumptions_vs_vbl.loc['Capacity factor']*df_electrolyzer_assumptions_vs_vbl.loc['Production rate']),
                        label=category , bottom = cumsum, width = barwidth, color = BM_capex_colors[i],
                    edgecolor = 'w', linewidth = linewidth_calc)
                cumsum += df_capex_BM_vs_vbl.fillna(0).loc[category]/(df_costing_assumptions_vs_vbl.loc['Plant lifetime']*365*df_costing_assumptions_vs_vbl.loc['Capacity factor']*df_electrolyzer_assumptions_vs_vbl.loc['Production rate'])
            
            if df_flags.loc['{} production rate'.format(product_name), 'T/F?'] == True:
                plt.xticks(rotation=35)  # Rotate text labels

            if df_flags.loc['Electricity cost', 'T/F?'] == True:
                axs.text(df_utility_imports.loc['Electricity - current US mix','Cost ($/kWh)'], 
                        y_axis_max_levelized*0.975, 'U.S. average industrial', ha='right', va='top', 
                    fontsize = SMALL_SIZE, rotation = 90)
                axs.plot([df_utility_imports.loc['Electricity - current US mix', 'Cost ($/kWh)'], df_utility_imports.loc['Electricity - current US mix', 'Cost ($/kWh)']], 
                        [y_axis_min_levelized, y_axis_max_levelized], alpha = 1,
                    c = theme_colors[6]) # Plot line for cost 

            ## Legend
            axs.legend(ncol = 1, bbox_to_anchor=(1.4, 1.15), loc='upper left', reverse = True, fontsize = 16) # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position
        #     axs.legend(bbox_to_anchor=(1, 1), loc='upper left') # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position
            st.pyplot(levelized_bar_fig, transparent = True, use_container_width = True)   

    with right_column.container(height = 300, border = False): 
        pass

    ###### POTENTIALS BAR CHART
    with middle_column.container(height = 300, border = False): 
        if not vbl_name == 'Cell voltage':
            with _render_lock:
                potentials_bar_fig, axs = plt.subplots() # Set up plot
                #fig.subplots_adjust(left=0.9, bottom=0.9, right=1, top=1, wspace=None, hspace=None)

                y_axis_major_ticks = y_axis_formatting(y_axis_min_potential, y_axis_max_potential, y_axis_num_potential)

                ## Axis labels
                axs.set_ylabel('Cell potential (V)')
                axs.set_xlabel(x_axis_label)

                ## Hide or show plot borders 
                axs.spines['right'].set_visible(True)
                axs.spines['top'].set_visible(True)

                ## Draw axis ticks
                plt.minorticks_on()
                axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
                axs.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
                plt.xticks(x_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])
                plt.yticks(y_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])

                ## Apply axis limits
                axs.set_xlim([x_axis_min,x_axis_max])
                axs.set_ylim([y_axis_min_potential,y_axis_max_potential])
                #axs.set_ylim([i_.iloc[:,0].min(axis=0),i_.iloc[:,0].max(axis=0)])

                ## Plot title
                # axs.set_title('Chronoamperometry at {:.3f} V for {:.2f} hrs'.format(vbl_we_avg, expt_time) )

                ## Plot series
                cumsum = 0
                counter = 0
                for i, category in enumerate(df_potentials_vs_vbl.iloc[2:7].index):
                    if not df_potentials_vs_vbl.loc[category].isnull().all():
                        axs.bar(vbl_range, abs(df_potentials_vs_vbl.fillna(0).loc[category]), label=category , bottom = cumsum, 
                                width = barwidth, color = potentials_colors[counter],
                        edgecolor = 'w', linewidth = linewidth_calc)
                        cumsum += abs(df_potentials_vs_vbl.fillna(0).loc[category])
                        counter += 1
                        
                ## Legend
                axs.legend(bbox_to_anchor=(1, 1), loc='upper left', reverse = True) # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position
                st.pyplot(potentials_bar_fig, transparent = True, use_container_width = True)

    ###### FE-SPC SCATTERPLOT
    with right_column.container(height = 300, border = False): 
        with _render_lock:
            FE_SPC_bar_fig, axs = plt.subplots() # Set up plot
            
            ## Axis labels
            axs.set_ylabel('FE$_{{{}}}$'.format(product_name))
            axs.set_xlabel('Single-pass conversion') 

            ## Draw axis ticks
            axs.xaxis.set_major_locator(mp.ticker.MultipleLocator(0.2))
            axs.yaxis.set_major_locator(mp.ticker.MultipleLocator(0.2))
            axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
            axs.yaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)

            ## Apply axis limits
            axs.set_xlim([0,1])
            axs.set_ylim([0,1])

            # Plot FE vs SPC
            axs.scatter(df_outlet_assumptions_vs_vbl.loc['Single-pass conversion'],
                        df_outlet_assumptions_vs_vbl.loc['FE {}'.format(product_name)],
                                    color = theme_colors[3], 
                                    label = 'FE$_{{CO_2R, 0}}$ > {}'.format(min(df_electrolyzer_assumptions_vs_vbl.loc['FE {} at 0% SPC'.format(product_name)])),
                                    s = 200, 
                                    alpha = 1, marker = 'o') 
        #     axs.scatter(df_products.loc[product_name, 'Typical SPC']*100, df_products.loc[product_name, 'FECO2R at SPC = 0']*100, 
        #                 marker = 'X', c = 'k', s = 200, alpha = 1, label = 'Limits')
            
            axs.legend(bbox_to_anchor = (1, 1), loc= 'upper left') # bbox_to_anchor = (1,1)
            st.pyplot(FE_SPC_bar_fig, transparent = True, use_container_width = True,)    

    ###### ENERGY BAR CHART 
    with middle_column.container(height = 300, border = False): 
        if not vbl_name == 'Cell voltage':   
            with _render_lock:
                energy_bar_fig, axs = plt.subplots() # Set up plot
                #fig.subplots_adjust(left=0.9, bottom=0.9, right=1, top=1, wspace=None, hspace=None)

                y_axis_major_ticks = y_axis_formatting(y_axis_min_energy, y_axis_max_energy, y_axis_num_energy)

                ## Axis labels
                axs.set_ylabel('Energy (kJ/mol$_{{{}}}$)'.format(product_name))
                axs.set_xlabel(x_axis_label)

                ## Draw axis ticks
                axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
                axs.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
                plt.xticks(x_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])
                plt.yticks(y_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])

                ## Apply axis limits
                axs.set_xlim([x_axis_min,x_axis_max])
                axs.set_ylim([y_axis_min_energy,y_axis_max_energy])

                ## Plot series
                cumsum = 0
                counter = 0
                for i, category in enumerate(df_energy_vs_vbl.iloc[:-3].index):
                    if not df_energy_vs_vbl.loc[category].isnull().all():
                        axs.bar(vbl_range, 
                                (abs(df_energy_vs_vbl.fillna(0).loc[category])/1000)*df_products.loc[product_name, 'Molecular weight (g/mol)'], # energy kJ/kg * 0.001 g/kg * MW g/mol
                                label=category , 
                                bottom = cumsum, width = barwidth, color = energy_colors[counter],
                                edgecolor = 'w', linewidth = linewidth_calc)
                        cumsum += (abs(df_energy_vs_vbl.fillna(0).loc[category])/1000)*df_products.loc[product_name, 'Molecular weight (g/mol)'] # energy kJ/kg * 0.001 g/kg * MW g/mol
                        counter += 1
                        
                ## Legend
                axs.legend(bbox_to_anchor=(1, 1), loc='upper left', reverse = True) 
                # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position

                st.pyplot(energy_bar_fig, transparent = True, use_container_width = True)   

    ###### EMISSIONS BAR CHART
    with right_column.container(height = 300, border = False): 
        if not vbl_name == 'Cell voltage':
            with _render_lock:
                emissions_bar_fig, axs = plt.subplots() # Set up plot
                #fig.subplots_adjust(left=0.9, bottom=0.9, right=1, top=1, wspace=None, hspace=None)

                y_axis_major_ticks = y_axis_formatting(y_axis_min_emissions, y_axis_max_emissions, y_axis_num_emissions)

                ## Axis labels
                axs.set_ylabel('Emissions \n (kg$_{{CO_2}}$/kg$_{{{}}}$)'.format(product_name))
                axs.set_xlabel(x_axis_label)

                ## Draw axis ticks
                axs.xaxis.set_minor_locator(AutoMinorLocator(2)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
                axs.yaxis.set_minor_locator(AutoMinorLocator(5)) # subdivide major ticks into this many minor divisions (eg. major step = 5 V, then autominorlocator(5) will mark off every 1 V)
                plt.xticks(x_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])
            #     plt.yticks(y_axis_major_ticks) # tick locations as a list, eg. plt.xticks([0,10,20])

                ## Apply axis limits
                axs.set_xlim([x_axis_min,x_axis_max])
                axs.set_ylim([y_axis_min_emissions,y_axis_max_emissions])

                ## Plot series
                axs.plot([x_axis_min, x_axis_max], [MW_CO2/df_products.loc[product_name, 'Molecular weight (g/mol)'],MW_CO2/df_products.loc[product_name, 'Molecular weight (g/mol)']], alpha = 1,
                        c = theme_colors[6]) # Plot line for negative emissions
                axs.text(x_axis_max * 0.975, MW_CO2/df_products.loc[product_name, 'Molecular weight (g/mol)']*0.95, 'Negative emissions', ha='right', va='top', 
                        fontsize = SMALL_SIZE)
                if df_flags.loc['Grid CO$_2$ intensity', 'T/F?'] == True:
                    axs.plot([df_utility_imports.loc['Electricity - current US mix', 'CO2 emissions (g CO2/kWh)']/1000,df_utility_imports.loc['Electricity - current US mix', 'CO2 emissions (g CO2/kWh)']/1000], 
                            [y_axis_min_emissions, y_axis_max_emissions], alpha = 1,
                        c = theme_colors[6]) # Plot line for cost 
                    axs.text(df_utility_imports.loc['Electricity - current US mix','CO2 emissions (g CO2/kWh)']/1000, 
                            y_axis_max_emissions*0.975, 'U.S. mix', ha='right', va='top', 
                        fontsize = SMALL_SIZE, rotation = 90)
                    
                    axs.plot([df_utility_imports.loc['Electricity - current California mix', 'CO2 emissions (g CO2/kWh)']/1000,df_utility_imports.loc['Electricity - current California mix', 'CO2 emissions (g CO2/kWh)']/1000], 
                            [y_axis_min_emissions, y_axis_max_emissions], alpha = 1,
                        c = theme_colors[6]) # Plot line for cost 
                    axs.text(df_utility_imports.loc['Electricity - current California mix', 'CO2 emissions (g CO2/kWh)']/1000, 
                            y_axis_max_emissions*0.975, 'California mix', ha='right', va='top', 
                        fontsize = SMALL_SIZE, rotation = 90)
                    
                    axs.plot([df_utility_imports.loc['Electricity - solar', 'CO2 emissions (g CO2/kWh)']/1000,df_utility_imports.loc['Electricity - solar', 'CO2 emissions (g CO2/kWh)']/1000], 
                            [y_axis_min_emissions, y_axis_max_emissions], alpha = 1,
                        c = theme_colors[6]) # Plot line for cost 
                    axs.text(df_utility_imports.loc['Electricity - solar', 'CO2 emissions (g CO2/kWh)']/1000, 
                                    y_axis_max_emissions*0.975, 'Solar', ha='right', va='top', 
                        fontsize = SMALL_SIZE, rotation = 90)

                cumsum = 0
                counter = 0
                for i, category in enumerate(df_emissions_vs_vbl.drop(['Total', 'Cell potential', 'Efficiency vs LHV'], inplace = False, errors = 'ignore').index):
                    if not df_emissions_vs_vbl.loc[category].isnull().all():
                        axs.bar(vbl_range, abs(df_emissions_vs_vbl.fillna(0).loc[category]), label=category , 
                                bottom = cumsum, width = barwidth, color = emissions_colors[counter],
                        edgecolor = 'w', linewidth = linewidth_calc)
                        cumsum += abs(df_emissions_vs_vbl.fillna(0).loc[category])
                        counter += 1
                        
                ## Legend
                axs.legend(bbox_to_anchor=(1, 1), loc='upper left', reverse = True) # -> bbox_to_anchor docks the legend to a position, loc specifies which corner of legend is that position

                st.pyplot(emissions_bar_fig, transparent = True, use_container_width = True)   
                
    #___________________________________________________________________________________

    st.divider()

    ############################### RAW RESULTS #####################################

    st.header('Raw model results')

    st.subheader('Capex')                                                                               
    df_capex_BM_vs_vbl_2
    df_capex_totals_vs_vbl_2

    st.subheader('Opex')
    df_opex_vs_vbl_2
    df_opex_totals_vs_vbl_2
    
    st.subheader('Sales')
    df_sales_vs_vbl_2
    
    st.subheader('Electrolyzer model')
    df_potentials_vs_vbl_2    

    st.subheader('Energy')
    df_energy_vs_vbl_2
    df_emissions_vs_vbl_2                                                                         

    st.subheader('Assumptions')
    df_costing_assumptions_vs_vbl_2
    df_electrolyzer_assumptions_vs_vbl_2
    df_outlet_assumptions_vs_vbl_2
    df_constants
    df_products
    df_utility_imports

st.write('Copyright © {} Shashwati C da Cunha. All rights reserved.'.format(datetime.now().date().strftime("%Y")))