
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
# from matplotlib.ticker import AutoMinorLocator
import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib import ticker
# from matplotlib.patches import Patch
# from matplotlib.lines import Line2D
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import animation
# from matplotlib import font_manager
import threading

# import csv

from datetime import datetime
# import timeit

# from functools import reduce

# from os.path import exists
# import os

# from io import BytesIO

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

@st.cache_data
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
        capacity_factor,
        battery_capex_USD_kWh,               
        battery_capacity,
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
    return single_run(product_name,
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
        capacity_factor,
        battery_capex_USD_kWh,               
        battery_capacity,
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
        K_to_C,
        kJ_per_kWh,
        )

@st.cache_data
def default_single_run(product_name,
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
        capacity_factor,
        battery_capex_USD_kWh,               
        battery_capacity,
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
    __, df_capex_totals_default, __, __, __, __,\
                df_energy_default, __, __, __, __, __, df_opex_totals_default, __,\
                __, df_potentials_default, __, __, __, __, __ = cached_single_run(product_name,
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
        capacity_factor,
        battery_capex_USD_kWh,               
        battery_capacity,
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
        K_to_C,
        kJ_per_kWh,
        )
    capex_default = df_capex_totals_default.loc['Total permanent investment', 'Cost ($)']
    opex_default = df_opex_totals_default.loc['Production cost', 'Cost ($/kg {})'.format(product_name)]
    levelized_default = df_opex_totals_default.loc['Levelized cost', 'Cost ($/kg {})'.format(product_name)]
    potential_default = df_potentials_default.loc['Cell potential', 'Value'] 
    energy_default = df_energy_default.loc['Total', 'Energy (kJ/kg {})'.format(product_name)]
    emissions_default = sum(df_energy_default.fillna(0).iloc[:-2].loc[:, 'Emissions (kg CO2/kg {})'.format(product_name)])
    return capex_default, opex_default, levelized_default, potential_default, energy_default, emissions_default

_render_lock = threading.RLock()

###################################################################################
################################### FORMATTING ####################################
###################################################################################

# Streamlit page formatting
st.set_page_config(page_title = 'CO2R Costing Dashboard - Home', 
                   page_icon = ":test_tube:",
                   initial_sidebar_state= 'expanded',
                   menu_items= {'Report a bug': 'mailto:shashwatidc@utexas.edu', },
                   layout="wide")

# Plot formatting for Matplotlib - rcParams. All options at https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams

# All fonts and font sizes
SMALL_SIZE = 20 # set smallest font size
MEDIUM_SIZE = 24 # set medium font size
BIGGER_SIZE = 27 # set
# font_dir = Path(mp.get_data_path(), r'/.streamlit/Arial/Arial.ttf')
# for font in font_manager.findSystemFonts(font_dir):
#     font_manager.fontManager.addfont(fontpaths = font)
# mp.rc('font', family = 'sans-serif') # 'Arial' # font group is sans-serif
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
summer = mp.colormaps['summer']
summer_r = mp.colormaps['summer_r']
PuOr = mp.colormaps['PuOr']
viridis = mp.colormaps['viridis']
viridis_r = mp.colormaps['viridis_r']
# wistia = mp.colormaps['Wistia']
# greys = mp.colormaps['gist_yarg'] # 'Gray'
RdBu = mp.colormaps['RdBu'] # seismic
RdYlBu = mp.colormaps['RdYlBu']
# inferno = mp.colormaps['inferno_r']
Blues = mp.colormaps['Blues']
# winter = mp.colormaps['winter_r']
# cool = mp.colormaps['cool_r']

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

st.title("CO₂R Costing Dashboard: Home")
st.write('''This tool generates the capital and operating cost for a CO₂ reduction process converting captured CO₂
         into either CO or ethylene. :red[It is based on the model in our paper [__]()]. 
        ''')
st.write("**Cite this work: __**")
st.write('Copyright © {} Shashwati C da Cunha. All rights reserved.'.format(datetime.now().date().strftime("%Y")))
st.write("Questions, collaborations, requests? Contact Shashwati da Cunha ([shashwatidc@utexas.edu](mailto:shashwatidc@utexas.edu)).")
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
    st.write("""Adjust the costing assumptions and model in the sidebar. The cost breakdowns will update dynamically every time you change the model.
           \n By default, the cell voltage will be modeled using Tafel equations, and the Faradaic efficiency based on the single-pass conversion and the maximum Faradaic efficiency.
         Mouse over the :grey[**?**] next to each input to see the default values for each parameter. Refresh the page to reset all values to their defaults.      
         """)

#__________________________________________________________________________________

###############################  IMPORT SOURCE DATA ###############################

file_imports = r"DataForTEA.xlsx"
sheet_utility_imports = 'Utilities'
sheet_constants =  'Constants and assumptions'
sheet_products =  'Products'

# Get current date and time to name files
time_now = datetime.now().time()
date_now = datetime.now().date()
current_date = date_now.strftime("%Y%m%d") # format string
current_time = time_now.strftime("%I-%M%p") # format string

# Cache Excel sheet reading
@st.cache_data
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
override_optimization = False
override_cell_voltage = False
override_eta_an = False
override_eta_cat = False
override_ohmic = False
overridden_vbl = ''
overridden_value = np.NaN
overridden_unit = ''
model_FE = 'Kas'
is_battery = False

middle_column, right_column = st.columns(2, gap = 'large')
st.sidebar.header('Model inputs' )
middle_column.header('Results')
right_column.header('_')

with st.sidebar:
    st.subheader('CO₂R product')
    ######## PRODUCT SELECTION
    # Choose a product
    product_name = st.radio(label = 'Reduction product', options= ['CO', 'Ethylene'], 
                    index = 0, # default option
                    label_visibility='collapsed',
                    help = '''Choose the product that CO₂ is reduced into. 
                    The only byproduct made is hydrogen. 
                      \n Default product: CO'''
    )
    # product_name = ['CO', 'Ethylene'][int(answer)-1] # fix indexing from 0 to extract product name from this list

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
j_total_mA_cm2 = float(df_products.loc[product_name, 'Optimal j @ 7.6 c/kWh, Hawks model']) # 300 # mA/cm2
default_j_total_mA_cm2 = j_total_mA_cm2
cell_E_V = 3.0  # default cell voltage
default_cell_E_V = cell_E_V
BV_eta_cat_V = -0.6
default_BV_eta_cat_V = BV_eta_cat_V
BV_eta_an_V = 0.25
default_BV_eta_an_V = BV_eta_an_V
FE_CO2R_0 = df_products.loc[product_name, 'FECO2R at SPC = 0']
default_FE_CO2R_0  = FE_CO2R_0
SPC = df_products.loc[product_name, 'Optimal SPC @ 7.6 c/kWh, Hawks model']  #0.3 # 0.5 # %/100
default_SPC = SPC
cat_Tafel_slope = df_products.loc[product_name, 'Tafel slope (mV/dec)']
default_cat_Tafel_slope = cat_Tafel_slope

# Save variables that may be adjusted
SPC_original = SPC

scaling = 4.7306 ## TODO: Move this option for modeling elsewhere
default_scaling = scaling
exponent = 5.4936 ## TODO: Move this option for modeling elsewhere
default_exponent = exponent

# SLIDERS 

with st.sidebar:
    st.subheader('Cell potential model')
    with st.expander(label = '**Simplified Butler-Volmer model assumptions**', expanded = False):
        override_cell_voltage = st.checkbox('Manually specify full-cell voltage', value = False,
                                            disabled = override_eta_an or override_eta_cat or override_ohmic)
        cell_E_V = st.slider(label = 'Cell voltage',
                            min_value = 0.001, 
                            max_value = 10.0, 
                            step = 0.1, value = cell_E_V,
                            format = '%.1f',
                    help = '''Check the box above to set the full-cell voltage. No underlying voltage model will be used. 
                    This means that current and voltage have no physical relationship.
                      \n Default cell voltage: {} V'''.format(default_cell_E_V),
                    disabled = not override_cell_voltage)
        if override_cell_voltage:
            st.write('*Using specified cell voltage*')
            overridden_vbl = 'Cell voltage'
            overridden_value = cell_E_V
            overridden_unit = 'V'
        else:
            st.write('*Modeling cell voltage from simplified Butler-Volmer model*')
            override_eta_cat = st.checkbox('Specify cathode (CO₂R) overpotential', value = False,
                                           disabled = override_cell_voltage)
            BV_eta_cat_V = st.slider(label = 'Cathodic overpotential (V)',
                            min_value = -10.0, 
                            max_value = 0.0, 
                            step = 0.1, value = BV_eta_cat_V,
                            format = '%.1f',
                            disabled = not override_eta_cat or override_cell_voltage,
                            help = '''Check the box above to set the cathodic overpotential. 
                            Thermodynamics, cell resistance and anodic overpotential will be modeled. 
                            Note that more negative overpotentials indicate slower kinetics.
                              \n Default cathodic overpotential: {} V'''.format(default_BV_eta_cat_V),)
            if override_eta_cat:
                st.write('*Using manually specified cathodic overpotential*')
                overridden_vbl = 'Cathodic overpotential'
                overridden_value = BV_eta_cat_V
                overridden_unit = 'V'            
            else:
                st.write('*Modeling cathodic overpotential from Tafel slope, {:.0f} mV/dec*'.format(cat_Tafel_slope))

            override_eta_an = st.checkbox('Specify anode (oxidation reaction) overpotential', value = False)
            BV_eta_an_V = st.slider(label = 'Anodic overpotential (V)',
                            min_value = 0.0, 
                            max_value = 10.0, 
                            step = 0.1, value = BV_eta_an_V,
                            format = '%.1f',
                            disabled = not override_eta_an or override_cell_voltage,
                            help = '''Check the box above to set the anodic overpotential. Thermodynamics, cell resistance and cathodic overpotential will be modeled. 
                              \n Default anodic overpotential: {} V'''.format(default_BV_eta_an_V),) 
            if override_eta_an:
                st.write('*Using manually specified anodic overpotential*')
                overridden_vbl = 'Anodic overpotential'
                overridden_value = BV_eta_an_V
                overridden_unit = 'V'              
            else:
                st.write('*Modeling anodic overpotential from Tafel slope, {:.0f} mV/dec*'.format(an_Tafel_slope))
            
            override_ohmic = st.checkbox('Specify full-cell area-specific resistance', value = False)
            R_ohmcm2 = st.slider(label = 'Area-specific resistance ($ \Omega \cdot$ cm$^2$',
                            min_value = 0.0, 
                            max_value = 25.0, 
                            step = 0.01, value = R_ohmcm2,
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
                            min_value = 0.001, 
                            max_value = 2000.0, 
                            step = 1.0, value = j_total_mA_cm2,
                            format = '%i',
                            help = '''Total current density of the cell. This will determine the size and voltage of the cell.
                              \n Default total current density: {} mA/cm$^2$'''.format(default_j_total_mA_cm2),)

        ##### FE-SPC TRADEOFF  
        option_1 = 'Plug flow in gas channel'
        option_2 = 'Carbonate electrolyte supports CO₂ availability'
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
            FE_product_specified = st.slider(label = 'FE_{}'.format(product_name),
                    min_value = 0.001,
                    max_value = 1.0,
                    step = 0.01, value = FE_CO2R_0)

        FE_CO2R_0 = st.slider(label = '$ FE_{CO_2R, \: 0}$, maximum Faradaic efficiency',
                            min_value = 0.001, 
                            max_value = 1.0, 
                            step = 0.01, value = FE_CO2R_0,
                            format = '%.2f',
                            help = r'''Maximum Faradaic efficiency achieved in the limit of 0 single-pass conversion or vast excess of CO₂,
                            $$$
                            lim_{\big X_{CO_2} → 0} FE_{\scriptsize CO_2R}
                            $$$
                            ''' +   '\n  Default $ FE_{{CO_2R, \: 0}}$: {}'.format(default_FE_CO2R_0),
                            disabled = not answer==option_3)
        SPC = st.slider(label = 'Single-pass conversion',
                            min_value = 0.001, 
                            max_value = 1.0, 
                            step = 0.01, value = SPC,
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
        st.latex(r'''
                 \footnotesize \implies  FE_{{{}}} = {:.2f}
                 '''.format(product_name, FE_product_checked))
        crossover_ratio = st.slider(label = 'Crossover ratio (mol CO₂/mol e$^-$)',
                            min_value = 0.001, 
                            max_value = 1.0, 
                            step = 0.01, value = crossover_ratio,
                            help = """The amount of CO₂ converted into carbonate ions that then crosses the membrane into the anode gas stream. 
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

with st.sidebar:
    st.subheader('Process design')
    with st.expander(label = '**Plant and separation parameters**', expanded = False):
        product_rate_kg_day = 1000 * st.slider(label = '{} production rate (ton/day)'.format(product_name),
                            min_value = 0.001 / 1000, 
                            max_value = 1.5e6 / 1000, 
                            step = 10.0, value = product_rate_kg_day / 1000,
                            format = '%i',
                            help = '''Daily production rate. This is fixed for the entire plant lifetime and sets the total CO₂R current required.
                              \n Default value: {} kg$ _{{{}}}$/day
                            '''.format(product_rate_kg_day, product_name))
        capacity_factor = st.slider(label = 'Capacity factor (days per 365 days)',
                            min_value = 0.001, 
                            max_value = 1.0, 
                            step = 0.01, value = capacity_factor,
                            format = '%.2f',
                            help = '''Fraction of time per year that the plant is operational.
                              \n Default value: {:.2f}, based on 350/365 days per year
                            '''.format(default_capacity_factor))
        lifetime_years = st.slider(label = 'Plant lifetime (years)',
                            min_value = 0.001, 
                            max_value = 100.0, 
                            step = 1.0, value = lifetime_years,
                            format = '%i',
                            help = '''Plant lifetime in years. The process operates to produce {} kg/day of product for this many years.
                              \n Default value: {} years
                            '''.format(product_rate_kg_day, default_capacity_factor))
        PSA_second_law_efficiency = st.slider(label = 'Second-law separation efficiency',
                            min_value = 0.001, 
                            max_value = 1.0, 
                            step = 0.01, value = PSA_second_law_efficiency,
                            format = '%.2f',
                            help = '''Second-law efficiency of gas separations between CO₂/O₂, CO₂/CO, and CO/H₂.
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
                                    min_value = 0.001, max_value = 1.000, 
                                    step = 0.01, value = 0.236,
                                    format = '%.2f',
                                    help = '''Fraction of time per day that renewable power is available.
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
        CO2_cost_USD_tCO2 = st.slider(label = 'CO₂ cost (\$/t CO₂)',
                            min_value = 0.0, 
                            max_value = 500.0, 
                            step = 1.0, value = CO2_cost_USD_tCO2,
                            format = '%i',
                            help = '''Default value: \${}/t$_{{CO_2}}$
                            '''.format(default_CO2_cost_USD_tCO2))
        H2_cost_USD_tCO2 = st.slider(label = 'H₂ cost (\$/kg H₂)',
                            min_value = 0.0, 
                            max_value = 5.0, 
                            step = 0.1, value = H2_cost_USD_kgH2,
                            format = '%.1f',
                            help = '''Default value: \${}/kg
                            '''.format(default_H2_cost_USD_kgH2))
        product_cost_USD_kgprod = st.slider(label = '{} cost (\$/kg$_{{{}}}$)'.format(product_name, product_name),
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
                            max_value = 12500.0, 
                            step = 100.0, value = electrolyzer_capex_USD_m2,
                            format = '%i',
                            help = '''Default value: \${}/m$^2$
                            '''.format(default_electrolyzer_capex_USD_m2))
        battery_capex_USD_kWh = st.slider(label = 'Battery capital cost (\$/kWh)' , 
                            min_value = 0.0, 
                            max_value = 500.0, 
                            step = 1.0, value = battery_capex_USD_kWh,
                            format = '%i', disabled = not is_battery,
                            help = '''Default value: \${}/kWh, based on 4-hour storage.
                            '''.format(default_battery_capex_USD_kWh))
        
    
with st.sidebar:
    st.subheader('Emissions assessment')
    electricity_emissions_kgCO2_kWh = st.slider(label = 'Grid CO₂ intensity (kg$_{CO_2}$/kWh)',
                        min_value = 0.0, 
                        max_value = 1.0, 
                        step = 0.01, value = electricity_emissions_kgCO2_kWh,
                        format = '%.2f',
                        help = '''Electricity emissions for partial life-cycle assessment.
                        \n Default value: {:.2f} kg$_{{CO_2}}$/kWh, based on the United States average.
                        '''.format(default_electricity_emissions_kgCO2_kWh))

#___________________________________________________________________________________

##########################  RUN MODEL AT DEFAULT VALUES  ###########################

capex_default, opex_default, levelized_default, potential_default, energy_default, emissions_default = default_single_run(product_name = product_name, 
                        product_rate_kg_day = default_product_rate_kg_day, 
                        df_products = df_products, FE_CO2R_0 = default_FE_CO2R_0, 
                        FE_product_specified = default_FE_product_specified, 
                        j_total_mA_cm2 = default_j_total_mA_cm2,SPC = default_SPC, 
                        crossover_ratio = default_crossover_ratio, model_FE = 'Hawks',  
                        overridden_vbl = '', overridden_value = np.NaN, overridden_unit = '', 
                        override_optimization =  override_optimization, P = default_P, T_streams = default_T_streams, 
                        R_ohmcm2 = default_R_ohmcm2, an_E_eqm = default_an_E_eqm, MW_CO2 = MW_CO2, 
                        MW_H2O = MW_H2O, MW_O2 = MW_O2,  MW_MX = MW_K2CO3,
                        cathode_outlet_humidity = default_cathode_outlet_humidity,
                        excess_water_ratio = default_excess_water_ratio,
                        an_eta_ref = default_an_eta_ref, 
                        an_Tafel_slope = default_an_Tafel_slope, 
                        an_j_ref = default_an_j_ref, 
                        electricity_emissions_kgCO2_kWh = default_electricity_emissions_kgCO2_kWh,
                        heat_emissions_kgCO2_kWh = default_heat_emissions_kgCO2_kWh,
                        electrolyte_conc = default_electrolyte_conc, 
                        density_kgm3 = default_density_kgm3,
                        PSA_second_law_efficiency = default_PSA_second_law_efficiency, 
                        T_sep = T_sep, electricity_cost_USD_kWh = default_electricity_cost_USD_kWh, 
                        heat_cost_USD_kWh = default_heat_cost_USD_kWh,product_cost_USD_kgprod = default_product_cost_USD_kgprod,
                        H2_cost_USD_kgH2 = default_H2_cost_USD_kgH2,water_cost_USD_kg = default_water_cost_USD_kg,
                        CO2_cost_USD_tCO2 = default_CO2_cost_USD_tCO2,lifetime_years = default_lifetime_years,
                        electrolyzer_capex_USD_m2 = default_electrolyzer_capex_USD_m2,
                        capacity_factor = default_capacity_factor,battery_capex_USD_kWh = default_battery_capex_USD_kWh,               
                        battery_capacity = default_battery_capacity, exponent=default_exponent, scaling=default_scaling,
                        carbon_capture_efficiency = default_carbon_capture_efficiency,
                        R = R,
                        F = F)

#___________________________________________________________________________________
    
##########################  GENERATE SINGLE RUN OF MODEL  ##########################

### Handle battery to flatten curve
if is_battery:
    battery_capacity = 1 - avbl_renewables # assumes daily storage battery
    capacity_factor = 350/365 # capacity is re-maximized
else:
    battery_capacity = 0

# ### Generate physical and costing model
if not np.isnan(FE_product_checked): 
    df_capex_BM, df_capex_totals, df_costing_assumptions, df_depreciation, df_electrolyzer_assumptions, df_electrolyzer_streams_mol_s,\
            df_energy, df_feedstocks, df_general, df_maintenance, df_operations, df_opex, df_opex_totals, df_outlet_assumptions,\
            df_overhead, df_potentials, df_sales, df_streams, df_streams_formatted, df_taxes, df_utilities = cached_single_run(product_name = product_name, 
                        product_rate_kg_day = product_rate_kg_day, 
                        df_products = df_products, FE_CO2R_0 = FE_CO2R_0, 
                        FE_product_specified = FE_product_specified, 
                        j_total_mA_cm2 = j_total_mA_cm2,SPC = SPC, 
                        crossover_ratio = crossover_ratio, model_FE = model_FE,  
                        overridden_vbl = overridden_vbl, overridden_value = overridden_value, overridden_unit = overridden_unit, 
                        override_optimization =  override_optimization, P = P, T_streams = T_streams, 
                        R_ohmcm2 = R_ohmcm2, an_E_eqm = an_E_eqm,MW_CO2 = MW_CO2, 
                        MW_H2O = MW_H2O, MW_O2 = MW_O2,  MW_MX = MW_K2CO3,
                        cathode_outlet_humidity = cathode_outlet_humidity,
                        excess_water_ratio = excess_water_ratio,
                        an_eta_ref = an_eta_ref, 
                        an_Tafel_slope = an_Tafel_slope, 
                        an_j_ref = an_j_ref, 
                        electricity_emissions_kgCO2_kWh = electricity_emissions_kgCO2_kWh,
                        heat_emissions_kgCO2_kWh = heat_emissions_kgCO2_kWh,
                        electrolyte_conc = electrolyte_conc, 
                        density_kgm3 = density_kgm3,
                        PSA_second_law_efficiency = PSA_second_law_efficiency, 
                        T_sep = T_sep, electricity_cost_USD_kWh = electricity_cost_USD_kWh, 
                        heat_cost_USD_kWh = heat_cost_USD_kWh,product_cost_USD_kgprod = product_cost_USD_kgprod,
                        H2_cost_USD_kgH2 = H2_cost_USD_kgH2,water_cost_USD_kg = water_cost_USD_kg,
                        CO2_cost_USD_tCO2 = CO2_cost_USD_tCO2,lifetime_years = lifetime_years,
                        electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2,
                        capacity_factor = capacity_factor,battery_capex_USD_kWh = battery_capex_USD_kWh,               
                        battery_capacity = battery_capacity, exponent=exponent, scaling=scaling,
                        carbon_capture_efficiency = carbon_capture_efficiency,
                        R = R,
                        F = F)
    df_emissions = pd.concat([pd.Series(df_outlet_assumptions.loc['Carbon capture loss', 'Value']), df_energy['Emissions (kg CO2/kg {})'.format(product_name)]])
    df_emissions.index = np.append('Carbon capture', df_energy.index)

    #___________________________________________________________________________________

    ############################### FIGURE OUTPUTS #####################################

    ###### PIE CHART FORMATTING
    ### Define colors for pie charts
    if is_battery:
        # Capex colors for bare modules
        BM_capex_colors = [summer_r(i) for i in np.linspace(0, 1, len(df_capex_BM.index)-1)] # battery gets its own color, so 1 less than capex length for other units
        BM_capex_colors.append((0.65, 0.65, 0.65, 1)) # add in battery 
    else:
        # Capex colors
        BM_capex_colors = [summer_r(i) for i in np.linspace(0, 1, len(df_capex_BM.index))]
        
    # Opex colors
    opex_colors = [PuOr(i) for i in np.linspace(0, 0.85, len(df_opex.index))]
    levelized_colors = opex_colors + BM_capex_colors

    # Potentials colors
    potentials_colors = [RdYlBu(i) for i in np.linspace(0, 1, np.shape(df_potentials.iloc[2:7])[0] )  ] # last rows are totals

    # Energy colors
    energy_colors = [RdYlBu(i) for i in np.linspace(0, 1,  sum(~df_energy.T.isnull().all()) - 2)  ] # last rows are totals

    # Emissions colors
    emissions_colors = energy_colors
    # emissions_colors = [RdYlBu(i) for i in np.linspace(0, 1, sum(~df_emissions.T.isnull().all()) - 2)  ] # len(df_energy_vs_vbl.index) - 2)] # last rows are totals
    
    @st.cache_data
    def delta_color_checker(df_capex_totals):
        if np.isclose(df_capex_totals.loc['Total permanent investment', 'Cost ($)'], capex_default, rtol = 1e-6, equal_nan = True) and np.isclose(df_opex_totals.loc['Production cost', 'Cost ($/kg {})'.format(product_name)], opex_default, rtol = 1e-6, equal_nan = True):
            delta_color = 'off'
        else:
            delta_color = 'inverse'       
        return delta_color

    delta_color = delta_color_checker(df_capex_totals = df_capex_totals)    

    alternating = 1
    flag = False
    far_near = {1: 3.5, -1: 4.5}

    ###### CAPEX PIE CHART
    with middle_column.container(height = 455, border = False): 
        st.subheader('Capital cost')
        # st.write('Capex: \${:.2f} million'.format(df_capex_totals.loc['Total permanent investment', 'Cost ($)']/1e6) )
        st.metric(label = 'Capex', value = '${:.2f} million'.format(df_capex_totals.loc['Total permanent investment', 'Cost ($)']/1e6), 
                delta = '{:.2f}%'.format(100*(df_capex_totals.loc['Total permanent investment', 'Cost ($)'] - capex_default)/capex_default),
                delta_color = delta_color, label_visibility='collapsed') 
        with _render_lock:
            capex_pie_fig, axs = plt.subplots(figsize = (5, 5*aspect_ratio)) # Set up plot
            axs.pie(df_capex_BM.loc[:, 'Cost ($)'], 
                    labels = df_capex_BM.index, labeldistance = 1.1,
                    autopct = lambda val: '{:.1f}%'.format(val) if val > 2 else '', 
                    pctdistance = 0.7,
                    colors = BM_capex_colors, startangle = 90, 
                    textprops = {'fontsize' : SMALL_SIZE}, 
                    radius = 2, wedgeprops= {'width' : 1}, # donut
                    counterclock = False,
                        )   
            axs.text(0, 0,  
                'Capex: \n ${:.2f} million'.format(df_capex_totals.loc[ 'Total permanent investment', 'Cost ($)']/1e6 ), # All capex except working capital, which is recovered during operation
                ha='center', va='center', 
                fontsize = MEDIUM_SIZE)  
            # buffer = BytesIO()
            # capex_pie_fig.savefig(buffer, format="png")
            # st.image(buffer, width = 400)
            st.pyplot(capex_pie_fig, transparent = True, use_container_width = True)

    ###### OPEX PIE CHART 
    with right_column.container(height = 455, border = False): 
        st.subheader('Operating cost')
        # st.write('Opex: \${:.2f}/kg$_{{{}}}$'.format(df_opex_totals.loc['Production cost', 'Cost ($/kg {})'.format(product_name)], product_name) )
        st.metric(label = 'Opex', value = r'${:.2f}/kg {} '.format(df_opex_totals.loc['Production cost', 'Cost ($/kg {})'.format(product_name)], product_name),
                delta = '{:.2f}%'.format(100*(df_opex_totals.loc['Production cost', 'Cost ($/kg {})'.format(product_name)] - opex_default)/opex_default),
                delta_color = delta_color, label_visibility = 'collapsed') 

        with _render_lock:
            opex_pie_fig, axs = plt.subplots(figsize = (5, 5*aspect_ratio)) # Set up plot
            axs.pie(df_opex.loc[:, 'Cost ($/kg {})'.format(product_name)], 
                    labels = df_opex.index, labeldistance = 1.1,
                    autopct = lambda val: '{:.1f}%'.format(val) if val > 2 else '', 
                    pctdistance = 0.8,
                    colors = opex_colors, startangle = 90, 
                    textprops = {'fontsize' : SMALL_SIZE}, 
                    radius = 2, wedgeprops= {'width' : 1}, # donut
                    counterclock = False,
                    # explode = 0.2*np.ones(len(df_opex.index),
                    )   
            axs.text(0, 0,  
            'Opex: \n \${:.2f}/kg$_{{{}}}$'.format(df_opex_totals.loc[ 'Production cost', 'Cost ($/kg {})'.format(product_name)], product_name), # All capex except working capital, which is recovered during operation
            ha='center', va='center', 
            fontsize = MEDIUM_SIZE)  
            axs.text(2.5, 0, 'x', color = 'white') # make figure bigger
            axs.text(-2.5, 0, 'x', color = 'white') # make figure bigger
            st.pyplot(opex_pie_fig, transparent = True, use_container_width = True)   

    ###### LEVELIZED PIE CHART
    with middle_column.container(height = 455, border = False): 
        st.subheader('Levelized cost')
        # st.write('Levelized cost: ${:.2f}/kg$_{{{}}}$'.format(df_opex_totals.loc['Levelized cost', 'Cost ($/kg {})'.format(product_name)], product_name ) )
        st.metric(label = 'Levelized', value = r'${:.2f}/kg {}'.format(df_opex_totals.loc['Levelized cost', 'Cost ($/kg {})'.format(product_name)], product_name),
                delta = '{:.2f}%'.format(100*(df_opex_totals.loc['Levelized cost', 'Cost ($/kg {})'.format(product_name)] - levelized_default)/levelized_default),
                delta_color = delta_color, label_visibility='collapsed') 
        levelized_pie_fig, axs = plt.subplots(figsize = (5, 5*aspect_ratio)) # Set up plot
        
        full_list_of_costs = pd.concat([df_opex.loc[:, 'Cost ($/kg {})'.format(product_name)],
                            df_capex_BM.loc[:,'Cost ($)']/(365*capacity_factor*lifetime_years*product_rate_kg_day)], axis = 0)

        with _render_lock:
            wedges, __, __ = axs.pie(full_list_of_costs, 
                            # labels = full_list_of_costs.index, 
                            # labeldistance = 1.1,
                            autopct = lambda val: '{:.1f}%'.format(val) if val > 2 else '', 
                            pctdistance = 0.8,
                            colors = levelized_colors, startangle = 0, 
                            textprops = {'fontsize' : SMALL_SIZE}, 
                            radius = 2, wedgeprops= {'width' : 1}, # donut
                            counterclock = False,
                            # explode = 0.2*np.ones(len(df_opex.index),
                            )   
            axs.text(0, 0,  
            'Levelized cost: \n \${:.2f}/kg$_{{{}}}$'.format(df_opex_totals.loc[ 'Levelized cost', 'Cost ($/kg {})'.format(product_name)], product_name), # All capex except working capital, which is recovered during operation
            ha='center', va='center', 
            fontsize = MEDIUM_SIZE)
            
            box_properties = dict(boxstyle="square,pad=0.3", fc="none", lw=0)
            label_properties_away = dict(arrowprops=dict(arrowstyle="-"),
                                bbox=box_properties, zorder=0, va="center")
            label_properties_near = dict(arrowprops=dict(arrowstyle="-",alpha = 0),
                                bbox=box_properties, zorder=0, va="center")
            for i, wedge in enumerate(wedges):
                middle_angle = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1 # in degrees
                y_posn = np.sin(np.deg2rad(middle_angle))
                x_posn = np.cos(np.deg2rad(middle_angle))
                verticalalignment = {-1: "bottom", 1: "top"}[int(np.sign(y_posn))]
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_posn))]
                if (wedge.theta2 - wedge.theta1) <22:
                    alternating = -alternating
                    connectionstyle = f"angle,angleA=0,angleB={middle_angle}"
                    label_properties_away["arrowprops"].update({"connectionstyle": connectionstyle})
                    axs.annotate(full_list_of_costs.index[i], xy=(x_posn, y_posn), 
                                xytext=(far_near[alternating]*1*x_posn, 3.7*y_posn),
                                horizontalalignment=horizontalalignment, 
                                **label_properties_away)
                else:                            
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_posn))]
                    axs.text(2.3*x_posn, 2.3*y_posn,
                            full_list_of_costs.index[i],
                            horizontalalignment=horizontalalignment, 
                            verticalalignment = verticalalignment)
            st.pyplot(levelized_pie_fig, transparent = True, use_container_width = True)   

    with right_column.container(height = 455, border = False): 
        pass

    ###### POTENTIALS PIE CHART
    with middle_column.container(height = 455, border = False):  
        st.subheader('Cell potential')
        # st.write('Full cell potential: {:.2f} V'.format(df_potentials.loc['Cell potential', 'Value']) )
        st.metric(label = 'Cell potential', value = '{:.2f} V'.format(df_potentials.loc['Cell potential', 'Value']),
                delta = '{:.2f}%'.format(100*(df_potentials.loc['Cell potential', 'Value'] - potential_default)/potential_default),
                delta_color = delta_color, label_visibility='collapsed') 
        if not override_cell_voltage:
            with _render_lock:
                potentials_pie_fig, axs = plt.subplots(figsize = (5, 5*aspect_ratio)) # Set up plot
                axs.pie(abs(df_potentials.iloc[2:7].loc[:,'Value']),
                        labels = df_potentials.iloc[2:7].index, labeldistance = 1.1,
                        autopct = lambda val: '{:.1f}%'.format(val) if val > 2 else '', 
                        pctdistance = 0.8,
                        colors = potentials_colors, startangle = 90, 
                        textprops = {'fontsize' : SMALL_SIZE}, 
                        radius = 2, wedgeprops= {'width' : 1}, # donut
                        counterclock = False,
                        # explode = 0.2*np.ones(len(df_opex.index),
                        )   
                axs.text(0, 0,  
                'Cell potential: \n {:.2f} V'.format(df_potentials.loc['Cell potential', 'Value']),
                ha='center', va='center', 
                fontsize = MEDIUM_SIZE)  
                st.pyplot(potentials_pie_fig, transparent = True, use_container_width = True)

    ###### ENERGY PIE CHART 
    with right_column.container(height = 455, border = False): 
        st.subheader('Process energy')
        # st.write('Energy required: {:.0f} kJ/kg$_{{{}}}}$'.format(df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)], product_name) )
        st.metric(label = 'Energy', value = r'{:.2f} MJ/kg {}'.format(df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)]/1000, product_name),
                delta = '{:.2f}%'.format(100*(df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)] - energy_default)/energy_default),
                delta_color = delta_color, label_visibility='collapsed') 
        with _render_lock:
            if not override_cell_voltage:
                energy_pie_fig, axs = plt.subplots(figsize = (5, 5*aspect_ratio)) # Set up plot
                axs.pie((abs(df_energy.iloc[2:-2].loc[:, 'Energy (kJ/kg {})'.format(product_name)])/1000)*df_products.loc[product_name, 'Molecular weight (g/mol)'],
                        labels = df_energy.iloc[2:-2].index, labeldistance = 1.1,
                        autopct = lambda val: '{:.1f}%'.format(val) if val > 2 else '', 
                        pctdistance = 0.8,
                        colors = energy_colors, startangle = 0, 
                        textprops = {'fontsize' : SMALL_SIZE}, 
                        radius = 2, wedgeprops= {'width' : 1}, # donut
                        counterclock = False,
                        # explode = 0.2*np.ones(len(df_opex.index),
                        )   
                axs.text(0, 0,  
                        'Energy: \n {:.0f} kJ/mol$_{{{}}}$'.format(sum((abs(df_energy.fillna(0).iloc[2:-2].loc[:, 'Energy (kJ/kg {})'.format(product_name)])/1000)*df_products.loc[product_name, 'Molecular weight (g/mol)']), product_name),
                        ha='center', va='center', 
                        fontsize = MEDIUM_SIZE)                  
                st.pyplot(energy_pie_fig, transparent = True, use_container_width = True)   

    ###### EMISSIONS PIE CHART
    with middle_column.container(height = 455, border = False): 
        st.subheader('Emissions')
        if electricity_emissions_kgCO2_kWh > 0:
            # st.write('Total emissions: {:.2f} kg$_{CO_2}$/kg$_{{{}}}$'.format(sum(df_energy.fillna(0).iloc[:-2].loc[:, 'Emissions (kg CO2/kg {})'.format(product_name)]), product_name ) )
            st.metric(label = 'Emissions', value = r'{:.2f} kg CO2/kg {}'.format(sum(df_energy.fillna(0).iloc[:-2].loc[:, 'Emissions (kg CO2/kg {})'.format(product_name)]), product_name ) ,
                delta = '{:.2f}%'.format(100*(sum(df_energy.fillna(0).iloc[:-2].loc[:, 'Emissions (kg CO2/kg {})'.format(product_name)])  - emissions_default)/emissions_default),
                delta_color = delta_color, label_visibility='collapsed') 
            if not override_cell_voltage:
                with _render_lock:
                    emissions_pie_fig, axs = plt.subplots(figsize = (5, 5*aspect_ratio)) # Set up plot
                    wedges, __, __ = axs.pie(df_emissions.loc[~np.isnan(df_emissions)].iloc[:-2], 
                        # labels = df_emissions.loc[~np.isnan(df_emissions)].iloc[:-2].index, labeldistance = 1.1,
                        autopct = lambda val: '{:.1f}%'.format(val) if val > 2 else '', 
                        pctdistance = 0.8,
                        colors = emissions_colors, startangle = 0, 
                        textprops = {'fontsize' : SMALL_SIZE}, 
                        radius = 2, wedgeprops= {'width' : 1}, # donut
                        counterclock = False,
                        # explode = 0.2*np.ones(len(df_opex.index),
                        )   
                    axs.text(0, 0,  
                    'Emissions: \n {:.2f} kg$_{{CO_2}}$/kg$_{{{}}}$'.format(sum(df_emissions.fillna(0).iloc[:-2]), product_name), # All capex except working capital, which is recovered during operation
                    ha='center', va='center', 
                    fontsize = MEDIUM_SIZE)  
                            
                    # Label pie chart with arrows
                    box_properties = dict(boxstyle="square,pad=0.3", fc="none", lw=0)
                    label_properties_away = dict(arrowprops=dict(arrowstyle="-"),
                                        bbox=box_properties, zorder=0, va="center")
                    for i, wedge in enumerate(wedges):
                        middle_angle = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1 # in degrees
                        y_posn = np.sin(np.deg2rad(middle_angle))
                        x_posn = np.cos(np.deg2rad(middle_angle))
                        verticalalignment = {-1: "bottom", 1: "top"}[int(np.sign(y_posn))]
                        if (wedge.theta2 - wedge.theta1) < 15:
                            alternating = -alternating
                            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_posn))]
                            connectionstyle = f"angle,angleA=0,angleB={middle_angle}"
                            label_properties_away["arrowprops"].update({"connectionstyle": connectionstyle})
                            axs.annotate(df_emissions.loc[~np.isnan(df_emissions)].iloc[:-2].index[i], xy=(x_posn, y_posn), 
                                        xytext=(far_near[alternating]*0.7*np.sign(x_posn), 3.5*y_posn),
                                        horizontalalignment=horizontalalignment, verticalalignment = 'center',
                                        **label_properties_away)
                        else:                            
                            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_posn))]
                            axs.text(2.3*x_posn, 2.3*y_posn,
                                        df_emissions.loc[~np.isnan(df_emissions)].iloc[:-2].index[i],
                                        horizontalalignment=horizontalalignment,
                                        verticalalignment=verticalalignment)
                    st.pyplot(emissions_pie_fig, transparent = True, use_container_width = True)   

    st.divider()
            
    #___________________________________________________________________________________

    ############################### RAW RESULTS #####################################

    st.header('Raw model results')
    st.subheader('Capex')                                                                                  
    df_capex_BM
    df_capex_totals

    st.subheader('Opex')
    df_opex
    df_opex_totals
        
    st.subheader('Sales')
    df_sales

    st.subheader('Electrolyzer model')
    df_potentials    

    st.subheader('Stream table')
    df_streams_formatted     

    st.subheader('Energy')
    df_energy
    df_emissions                                                                           

    st.subheader('Assumptions')
    df_costing_assumptions
    df_electrolyzer_assumptions
    df_outlet_assumptions    
    df_constants
    df_products
    df_utility_imports
    
else:
    with middle_column:
        st.header(':red[Model is physically unviable. Please check $ FE_{CO_2R, \: 0}$,  $ X_{CO_2}$ and crossover ratio.]')

