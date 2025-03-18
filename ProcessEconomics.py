# %% [markdown]
# # Technoeconomic model: Notes
# Date updated: 2024/06/11 \
# Update notes: Strict function inputs\
# Contact: Shashwati da Cunha, [shashwati.dacunha@austin.utexas.edu](mailto:shashwati.dacunha@austin.utexas.edu)
# 
# ### Instructions
# 1. Not designed for standalone run - this is only a collection of functions. Other notebooks call it.
# 

# %% [markdown]
# ## 0. Imports and setup

# %%
# UNCOMMENT TO RUN STANDALONE
import streamlit as st
import pandas as pd
import numpy as np
from scipy import optimize

# %% [markdown]
# ### 0.1 Imports and styling

# %% [markdown]
# ### 0.2 Check for independent variables

# %%
### Save costs to dataframe

@st.cache_data(ttl = "1h")
def costing_assumptions(
    product_name,
    product_cost_USD_kgprod,
    H2_cost_USD_kgH2 ,
    electricity_cost_USD_kWh ,
    water_cost_USD_kg,
    CO2_cost_USD_tCO2,
    lifetime_years ,
    stack_lifetime_years,
    electrolyzer_capex_USD_m2,
    capacity_factor
): 
    """
    Store all costing assumptions

    Arguments: product name of choice, product selling price ($/kg product), H2 selling price ($/kg H2), electricity cost ($/kWh), water cost ($/kg water), 
    CO2 cost ($/ton CO2), lifetime of plant (years), stack lifetime (years), capex of electrolyzer ($/m^2), capacity factor of plant operation () 

    Returns: dataframe of costing assumptions
    """
    
    df_costing_assumptions = pd.DataFrame(
    {
        '{}'.format(product_name) : [product_cost_USD_kgprod , '$/kg'],
        'H2' : [H2_cost_USD_kgH2 , '$/kg'],
        'Electricity' : [electricity_cost_USD_kWh , '$/kWh'],
        'Water' : [water_cost_USD_kg, '$/kg'],
        'CO2' : [CO2_cost_USD_tCO2 , '$/ton CO2'],
        'Plant lifetime': [lifetime_years, 'years'],
        'Stack lifetime': [stack_lifetime_years, 'years'],
        'Electrolyzer capex': [electrolyzer_capex_USD_m2, '$/m^2'],
        "Capacity factor" : [capacity_factor, ''],
    },
    ).transpose()

    df_costing_assumptions.columns = ['Cost', 'Units']
    df_costing_assumptions = df_costing_assumptions.astype({'Cost':'float64', 'Units':'string'})
    df_costing_assumptions.index.name = 'Assumed variable - inputs and costing'
    
    return df_costing_assumptions

# %% [markdown]
# ## 1. Capital costs

# %% [markdown]
# ### 1.1 Electrolyzer capex

# %%
@st.cache_data(ttl = "1h")
def capex(
    area_m2 , 
    electricity_kJ_per_kg,
    df_streams,
    product_rate_kg_day,
    battery_capex_USD_kWh ,
    electrolyzer_capex_USD_m2, # electrolyzer_capex_USD_kW
    battery_capacity,
    kJ_per_kWh,
    is_additional_capex,
    additional_capex_USD,
):
    """
    Calculate unit (bare-module) capital costs, and cumulative capital costs mostly according to the method of Seider, Lewin et. al.

    Arguments: electrolyzer area (m^2), electricity required per kg product, dataframe of stream table, production basis (kg/day),
    battery capex ($/kWh), electrolyzer capex ($/m^2), battery pseudo-"capacity factor" or fraction of time when battery powers plant (), 
    kJ/kWh conversion

    Returns: dataframe of bare-module costs per unit, and summary capital costs
    """
    
    # Price index to adjust reference values
    CEPCI_2024 = 800
    CEPCI_2020 = 596.2
    
    ## Battery limits ("on-site") are the electrolysis and separations only. 
    # => There are no off-site units included, i.e. feedstocks (CO2, DI water) and utilities (electricity) come from external vendors 
    
    ## Bare module costs of onsite equipment (includes process equipment which is customized, like pressure vessels; and off-the-shelf process machinery, like pumps)
    dict_capex_BM = {
        'Electrolyzer' : ['Electrolysis', '$5174/m2 - Linear scaling, Badgett Cortright J Cleaner Prod 2022', np.NaN],
        #         'Deionization' : ['', '', 0] , # TODO: add deionization capex and opex
        'Balance of plant' : ['Pressure changes etc', '34% of electrolyzer bare module - Linear scaling, H2A model', np.NaN] ,
        'Cathode PSA - CO$_2$/products' : ['Separations', 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', np.NaN] ,
        'Cathode PSA - Products/H$_2$' : ['Separations', 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', np.NaN] ,
        'Anode PSA - CO$_2$/O$_2$' : ['Separations', 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', np.NaN] ,
    }

    # Create dataframe for bare-module costs
    df_capex_BM = pd.DataFrame(dict_capex_BM).T
    df_capex_BM.columns = ['Stage', 'Description', 'Cost ($)']                                                     
    df_capex_BM = df_capex_BM.astype({'Stage':'string', 'Description': 'string', 'Cost ($)':'float64'})

    df_capex_BM.index.name= 'Bare-module cost' 

    # Electrolyzer 
    # H2A model - $342/kW for a 1.9V, 2000 mA/cm2 unit, 1.12x installation factor
    # electrolyzer_capex_USD_cm2 = electrolyzer_capex_USD_kW * 2000/1000 * 1.9 / 1000 * 1.12
    # df_capex_BM.loc['Electrolyzer', 'Cost ($)'] = electrolyzer_capex_USD_cm2 * area_m2 * 10**4 # Assume linear scaling
    
    # Badgett et al - $10,000/m2 to $5,000/m2 depending on manufacturing scale
    df_capex_BM.loc['Electrolyzer', 'Cost ($)'] = electrolyzer_capex_USD_m2 * area_m2 * 1.12 # Assume linear scaling

    # H2A - balance of plant is 34.5% of electrolyzer capex
    df_capex_BM.loc['Balance of plant', 'Cost ($)']  = df_capex_BM.loc['Electrolyzer', 'Cost ($)'] * 0.345

    ## TODO: Are these FOB costs  (without delivery)? Or including delivery, installation, material factors (including bare-module factor) to give bare-module costs?
    # PSA - reference cost with scaling factor
    df_capex_BM.loc['Cathode PSA - CO$_2$/products', 'Cost ($)']  = 1989043 * CEPCI_2024/CEPCI_2020 * (df_streams.loc['Cathode PSA1 inlet', 'Volumetric flow rates (m3/s)']*(60*60)/1000)**0.7 # relative to 1000 m3/hr 
    df_capex_BM.loc['Cathode PSA - Products/H$_2$', 'Cost ($)']  = 1989043 * CEPCI_2024/CEPCI_2020 * (df_streams.loc['Cathode PSA1 outlet', 'Volumetric flow rates (m3/s)']*(60*60)/1000)**0.7 # relative to 1000 m3/hr 
    if df_streams.loc['Anode PSA inlet', 'x_CO2'] != 0: # Check if there is any crossover, if none then unit cost is 0
        df_capex_BM.loc['Anode PSA - CO$_2$/O$_2$', 'Cost ($)']  = 1989043 * CEPCI_2024/CEPCI_2020 * (df_streams.loc['Anode PSA inlet', 'Volumetric flow rates (m3/s)']*(60*60)/1000)**0.7 # relative to 1000 m3/hr 
    else:
        df_capex_BM.loc['Anode PSA - CO$_2$/O$_2$', 'Cost ($)']  = 0
    if is_additional_capex:
        df_capex_BM.loc['Additional', 'Cost ($)'] = additional_capex_USD
        df_capex_BM.loc['Additional', 'Description'] = 'User input'
    
    # Create dataframe for capex summary
    df_capex_totals = pd.DataFrame(dict_capex_BM, columns = ['Capex summary', 'Stage', 
                                                         'Description', 'Cost ($)'])
    df_capex_totals.set_index('Capex summary', inplace = True)     

    # Cost of spares, storage and surge tanks, initial catalyst charges, controls and computers are all assumed to be 0

    ## Total bare-module investment is the total cost of installed equipment in the plant
    df_capex_totals.loc['Total bare-module investment', 'Cost ($)'] = df_capex_BM.loc[:,'Cost ($)'].sum(axis=0)
    df_capex_totals.loc['Total bare-module investment', ['Stage', 'Description']] = '', 'C_TBM'

    ## Total direct permanent investment includes the cost of all additional facilities constructed
    # Cost of site preparation, service facilities, utility plants and auxiliary facilities is ignored here
    ## Allocated cost: capital for offsite allocated costs, e.g. utility plants, steam, electricity generation, waste disposal - zero here since analysis focuses on CO2R
    C_alloc = 0 # C_TDC*0.8 
    df_capex_totals.loc['Allocated capital', 'Cost ($)'] = C_alloc
    df_capex_totals.loc['Allocated capital', ['Stage', 'Description']] = '', ' C_alloc'
    # Battery storage - $200/kWh from NREL 2021 report, mid costing scenario. 
    # Battery cost is NOT included in total bare-module investment, which is used for calculating parts of opex; we include it in df_capex_BM only for plotting
    if battery_capacity > 0:
        battery_cost = (electricity_kJ_per_kg * product_rate_kg_day) / kJ_per_kWh * battery_capex_USD_kWh * battery_capacity
        df_capex_BM.loc['Battery storage', ['Stage', 'Description', 'Cost ($)']] = '', '${}/kWh'.format(battery_capex_USD_kWh), battery_cost 
        # Assumes that avbl_renewables is a fraction per day, and the battery needs to store this amount daily, and that its lifetime = process lifetime
        df_capex_BM.loc['Battery storage', ['Stage', 'Description']] = 'Battery', 'NOT included in total bare-module investment - shown here for figure generation only'  
        df_capex_totals.loc['Allocated capital', 'Cost ($)'] += battery_cost
        
    df_capex_totals.loc['Total direct permanent investment' , 'Cost ($)'] = df_capex_totals.loc['Total bare-module investment', 'Cost ($)'] + df_capex_totals.loc['Allocated capital', 'Cost ($)']
    df_capex_totals.loc['Total direct permanent investment', ['Stage', 'Description']] = '', 'C_DPI'  
    
    ## Total depreciable capital includes all capital costs for the actual installed equipment
    df_capex_totals.loc['Contractor and contingencies', 'Cost ($)'] = 0.18 * df_capex_totals.loc['Total direct permanent investment' , 'Cost ($)'] 
    # 18% of C_DPI = 3% contractors, 15% contingencies. This is very low for contingency for a new technology, typically 35 - 100% is estimated
    df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] = df_capex_totals.loc['Total direct permanent investment' , 'Cost ($)'] + df_capex_totals.loc['Contractor and contingencies', 'Cost ($)']
    df_capex_totals.loc['Total depreciable capital', ['Stage', 'Description']] = '', ' C_TDC'    
    C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)']

    ## Total permanent investment includes the cost of land 
    # Land, patent royalties and plant startup costs are ignored. Land rent is included in opex calculations
    df_capex_totals.loc['Total permanent investment', 'Cost ($)'] = df_capex_totals.loc['Total depreciable capital', 'Cost ($)']
    df_capex_totals.loc['Total permanent investment', ['Stage', 'Description']] = '', ' C_TPI'    

    ## Total capital investment includes all capital investments including working capital 
    ## Working capital covers operating costs before the plant begins to sell product - inventory, accounts receivable, raw materials, other operational requirements
    C_WC = 0.15 * C_TDC # A coarse estimate for working capital is 15 - 25% of the total depreciable capital
    df_capex_totals.loc['Working capital', 'Cost ($)'] = C_WC 
    df_capex_totals.loc['Working capital', ['Stage', 'Description']] = '', ' C_WC'  
    df_capex_totals.loc['Total capital investment', 'Cost ($)'] = df_capex_totals.loc['Total permanent investment', 'Cost ($)'] + df_capex_totals.loc['Working capital', 'Cost ($)']
    df_capex_totals.loc['Total capital investment', ['Stage', 'Description']] = '', ' C_TCI'    
        
    return df_capex_BM, df_capex_totals, C_TDC, C_alloc

# %% [markdown]
# ## 2. Sales

# %%
@st.cache_data(ttl = "1h")
def sales(
    product_name,
    df_streams,
    product_cost_USD_kgprod ,
    H2_cost_USD_kgH2 ,
    product_rate_kg_day ,
    capacity_factor
):
    """
    Calculate sales from mass balance

    Arguments: product name of choice, dataframe of stream table, product selling price ($/kg product), H2 selling price ($/kg H2), 
    production basis (kg product/ day), capacity factor of plant operation () 

    Returns: dataframe of sales 
    """
    
    # Create dictionary
    dict_sales = {
        '{}'.format(product_name) : ['Separations', '{}/kg {}'.format(product_cost_USD_kgprod, product_name), np.NaN],
        'H2' : ['Separations', '{}/kg H2'.format(H2_cost_USD_kgH2), np.NaN] ,
    }

    # Create dataframe
    df_sales = pd.DataFrame(dict_sales).T
    df_sales.columns = ['Stage', 'Description', 'Earnings ($/yr)']
    df_sales = df_sales.astype({'Stage':'string', 'Description': 'string', 'Earnings ($/yr)':'float64'})

    df_sales.index.name= 'Sales'    

    # Fill in costs 
    df_sales.loc['{}'.format(product_name), 'Earnings ($/yr)'] = product_cost_USD_kgprod * df_streams.loc[ 'Product outlet', 'Mass flow rate (kg/day)'] * 365*capacity_factor
    df_sales.loc['H2','Earnings ($/yr)'] = H2_cost_USD_kgH2 * df_streams.loc[ 'H2 outlet', 'Mass flow rate (kg/day)'] * 365*capacity_factor

    # Calculate total
    df_sales.loc['Total', 'Earnings ($/yr)'] = df_sales['Earnings ($/yr)'].sum(axis=0)
    df_sales.loc['Total', ['Stage', 'Description']] = ''

    # Calculate per product
    df_sales['Earnings ($/kg {})'.format(product_name)] = df_sales['Earnings ($/yr)']/(product_rate_kg_day*365*capacity_factor)
    df_sales['Earnings ($/day)'] = df_sales['Earnings ($/yr)']/(365*capacity_factor)
    df_sales
    
    return df_sales

# %% [markdown]
# ## 3. Operating expenses

# %% [markdown]
# ### 3.1 Feedstocks

# %%
@st.cache_data(ttl = "1h")
def feedstocks(
    CO2_cost_USD_tCO2,
    water_cost_USD_kg,
    df_streams,
    capacity_factor
):
    """
    Calculate feedstock costs from mass balance

    Arguments: CO2 cost ($/ton), DI water cost ($/kg), dataframe of stream table, capacity factor of plant operation () 

    Returns: dataframe of feedstock costs
    """
    
    # Create dictionary
    dict_feedstocks = {
        'Captured CO2' : ['Carbon capture', '', np.NaN],
        'Deionized water' : ['Electrolyte makeup', '', np.NaN] ,
    }

    # Create dataframe
    df_feedstocks = pd.DataFrame(dict_feedstocks).T
    df_feedstocks.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_feedstocks = df_feedstocks.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_feedstocks.index.name= 'Feedstocks'

    # Fill in costs 
    df_feedstocks.loc['Captured CO2', 'Cost ($/yr)'] = CO2_cost_USD_tCO2/1000*df_streams.loc['Fresh CO2 feed', 'Mass flow rate (kg/day)']*365*capacity_factor
    df_feedstocks.loc['Deionized water','Cost ($/yr)'] = water_cost_USD_kg*df_streams.loc['Fresh water feed', 'Mass flow rate (kg/day)']*365*capacity_factor #df_streams[]

    df_feedstocks
    
    return df_feedstocks

# %% [markdown]
# ### 3.2 Utility costing and emissions

# %% [markdown]
# #### 3.2.1 Get utilities based on energy demands

# %%
@st.cache_data(ttl = "1h")
def utilities(df_energy,
              product_rate_kg_day,
              capacity_factor,
             product_name):
    """
    Calculate utility costs from energy balance

    Arguments: dataframe of energy table, production basis (kg product/ day), capacity factor of plant operation (), product name of choice

    Returns: dataframe of utility costs
    """
        
    # Create dataframe
    df_utilities = pd.DataFrame()
    
    # Calculate costs and emissions per utility
    df_utilities.loc['Electricity', ['Cost ($/kg {})'.format(product_name),
                                     'Energy (kJ/kg {})'.format(product_name)]] = abs(df_energy.loc[df_energy['Description'] == 'Electricity', ['Cost ($/kg {})'.format(product_name),
                                                                                                                                                'Energy (kJ/kg {})'.format(product_name)]]).sum(axis = 0) 
    # df_utilities.loc['Heat', 'Cost ($/kg {})'.format(product_name)] = abs(df_energy.loc[df_energy['Description'] == 'Heat', 'Cost ($/kg {})'.format(product_name)]).sum(axis = 0) 
    
    df_utilities.loc[:, 'Cost ($/yr)']  = df_utilities.loc[:,'Cost ($/kg {})'.format(product_name) ] * product_rate_kg_day * 365 * capacity_factor
    df_utilities.loc['Total', 'Energy (kJ/kg {})'.format(product_name)] = df_utilities.loc[:, 'Energy (kJ/kg {})'.format(product_name)].sum(axis=0)
   
      
    df_utilities.index.name= 'Utilities'

    return df_utilities

# %% [markdown]
# ### 3.3 Operations

# %%
@st.cache_data(ttl = "1h")
def operations(
    capacity_factor,
):
    """
    Calculate operations costs (labor) based on Seider, Lewin et. al.

    Arguments: capacity factor of plant operation () 

    Returns: dataframe of operations (labor) costs
    """
    
    ## SEIDER BOOK
    
    # Create dictionary
    dict_operations = {
        'Direct wages and benefits (DW&B)' : ['', '$40/operator/hr - 3 operators', np.NaN],
        'Direct salaries and benefits' : ['', '15% DW&B', np.NaN] ,
        'Operating supplies and services': ['', '6% DW&B', np.NaN],
        'Technical assistance to manufacturing' : ['', '$60000/(operator/shift)/yr - 1 operator/shift, 3 shifts/day', np.NaN ],
        'Control laboratory': ['', '$65000/(operator/shift)/yr - 1 operator/shift, 3 shifts/day', np.NaN]
    }

    # Create dataframe
    df_operations = pd.DataFrame(dict_operations).T
    df_operations.columns = ['Stage',  'Description', 'Cost ($/yr)'] 
    df_operations = df_operations.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_operations.index.name= 'Operations'

    # Fill in costs
    df_operations.loc['Direct wages and benefits (DW&B)', 'Cost ($/yr)'] = 40*365*capacity_factor*24*3
    df_operations.loc['Direct salaries and benefits','Cost ($/yr)'] = 0.15*df_operations.loc['Direct wages and benefits (DW&B)','Cost ($/yr)']
    df_operations.loc['Operating supplies and services', 'Cost ($/yr)'] = 0.06*df_operations.loc['Direct wages and benefits (DW&B)', 'Cost ($/yr)']
    df_operations.loc['Technical assistance to manufacturing', 'Cost ($/yr)'] =60000*1*3*capacity_factor
    df_operations.loc['Control laboratory', 'Cost ($/yr)'] =65000*1*3*capacity_factor

    return df_operations

# %% [markdown]
# ### 3.4 Maintenance

# %%
@st.cache_data(ttl = "1h")
def maintenance(
    C_TDC,
    C_electrolyzer,
):
    """
    Calculate maintenance costs based on Seider, Lewin et. al.

    Arguments: total depreciable capital ($), electrolyzer installed bare-module cost ($)

    Returns: dataframe of maintenance cost
    """
        
    # Create dictionary
    dict_maintenance = {
        'Maintenance wages and benefits (MW&B)' : ['', '3.5% of C_TCD - fluids handling process', np.NaN],
        'Maintenance salaries and benefits' : ['', '25% MW&B', np.NaN] ,
        'Materials and services': ['', '100% MW&B', np.NaN],
        'Maintenance overhead' : ['', '5% MW&B', np.NaN ],
    }

    # Create dataframe
    df_maintenance = pd.DataFrame(dict_maintenance).T
    df_maintenance.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_maintenance = df_maintenance.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_maintenance.index.name= 'Maintenance'  

    # Fill in costs
    # WARNING: hardcoded 18% factor on electrolyzer cost contributing to C_TDC
    df_maintenance.loc['Maintenance wages and benefits (MW&B)', 'Cost ($/yr)'] = 0.035* ( C_TDC - C_electrolyzer*1.18 )
    df_maintenance.loc['Maintenance salaries and benefits','Cost ($/yr)'] = 0.25*df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)']
    df_maintenance.loc['Materials and services', 'Cost ($/yr)'] = 1.00*df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)']
    df_maintenance.loc['Maintenance overhead', 'Cost ($/yr)'] = 0.05*df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)']

    return df_maintenance

# %%
@st.cache_data(ttl = "1h")
def stack_replacement(C_electrolyzer,
                    stack_lifetime_years,
                    lifetime_years):
    """
    Calculate stack replacement cost, annualized

    Arguments: electrolyzer installed bare-module cost ($), stack lifetime after which full replacement is necessary (years), 
    total plant lifetime (years)

    Returns: dataframe of stack replacement cost
    """
         
    
    # Create dictionary
    dict_stack_replacement = {
        'Stack replacement' : ['', 'Full electrolyzer cost after {} years'.format(stack_lifetime_years), np.NaN],
    }

    # Create dataframe
    df_stack_replacement = pd.DataFrame(dict_stack_replacement).T
    df_stack_replacement.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_stack_replacement = df_stack_replacement.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_stack_replacement.index.name= 'Stack replacement'  

    # Fill in costs
    no_of_replacements = max( 0, (lifetime_years // stack_lifetime_years - 1))
    df_stack_replacement.loc['Stack replacement', 'Cost ($/yr)'] = C_electrolyzer * no_of_replacements / lifetime_years # Total replacement cost / plant lifetime
  
    return df_stack_replacement

# %% [markdown]
# ### 3.5 Operating overhead

# %%
@st.cache_data(ttl = "1h")
def overhead(df_maintenance,
            df_operations):
    """
    Calculate operating overheads according to Seider, Lewin et. al.

    Arguments: dataframe of maintenance expenses, dataframe of operating expenses

    Returns: dataframe of overheads
    """
    
    # Create dictionary
    dict_overhead = {
        'General plant overhead' : ['', '7.1% of M&O-SW&B', np.NaN],
        'Mechanical department services' : ['', '2.4% of M&O-SW&B', np.NaN] ,
        'Employee relations department': ['', '5.9% of M&O-SW&B', np.NaN],
        'Business services' : ['', '7.4% of M&O-SW&B', np.NaN ],
    }

    # Create dataframe
    df_overhead = pd.DataFrame(dict_overhead).T
    df_overhead.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_overhead = df_overhead.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_overhead.index.name= 'Overheads'  

    # Fill in costs
    # WARNING: mo_swb EXCLUDES STACK REPLACEMENT
    mo_swb = df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)'] + df_operations.loc['Direct wages and benefits (DW&B)','Cost ($/yr)'] + df_maintenance.loc['Maintenance salaries and benefits','Cost ($/yr)'] + df_operations.loc['Direct salaries and benefits','Cost ($/yr)'] 
    df_overhead.loc['General plant overhead', 'Cost ($/yr)'] = 0.071*mo_swb
    df_overhead.loc['Mechanical department services','Cost ($/yr)'] = 0.024*mo_swb
    df_overhead.loc['Employee relations department', 'Cost ($/yr)'] =0.059*mo_swb
    df_overhead.loc['Business services', 'Cost ($/yr)'] = 0.074*mo_swb

    return df_overhead

# %% [markdown]
# ### 3.6 Property taxes and insurance

# %%
@st.cache_data(ttl = "1h")
def taxes(C_TDC):    
    """
    Calculate taxes according to Seider, Lewin et. al.

    Arguments: total depreciable cost or inside battery limits capital cost
    
    Returns: dataframe of tax expenses
    """
    
    # Create dataframe
    df_taxes = pd.DataFrame(columns = [ 'Stage', 'Description', 'Cost ($/yr)']).astype({'Stage':'string', 
                                                                                        'Description':'string', 
                                                                                        'Cost ($/yr)':'float64'})
    df_taxes.index.name= 'Taxes' 

    # Fill in costs
    df_taxes.loc['Property taxes and insurance', 'Description'] = '2% C_TDC per year'
    df_taxes.loc['Property taxes and insurance', 'Cost ($/yr)'] = 0.02*C_TDC

    return df_taxes

# %% [markdown]
# ### 3.7 Depreciation

# %%
@st.cache_data(ttl = "1h")
def depreciation(
    C_TDC,
    C_alloc
):
    """
    Calculate depreciation with a linear rate for roughly 12 year depreciation period

    Arguments: total depreciable cost (or inside battery limits capital cost), allocated cost (or outside battery limits capital cost)

    Returns: dataframe of depreciation expenses
    """
        
    # Create dictionary
    dict_depreciation = {
        'Direct plant' : ['', '8% of (C_TDC – 1.18 C_alloc)', np.NaN],
        'Allocated plant' : ['', '6% of 1.18 C_alloc', np.NaN] ,
    }

    # Create dataframe
    df_depreciation = pd.DataFrame(dict_depreciation).T
    df_depreciation.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_depreciation = df_depreciation.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_depreciation.index.name= 'Depreciation'    

    # Fill in costs
    df_depreciation.loc['Direct plant', 'Cost ($/yr)'] =  0.08*(C_TDC - 1.18*C_alloc)
    df_depreciation.loc['Allocated plant','Cost ($/yr)'] =  0.06*1.18*C_alloc

    return df_depreciation

# %% [markdown]
# ### 3.8 General expenses

# %%
@st.cache_data(ttl = "1h")
def general(
    sales_USD_year
):
    """
    Calculate general expenses according to Seider, Lewin et. al.

    Arguments: dataframe of sales

    Returns: dataframe of general expenses
    """
    
    # Create dictionary
    dict_general = {
        'Selling (or transfer) expense' : ['', '1% of sales', np.NaN],
        'Direct research' : ['', '4.8% of sales', np.NaN] ,
        'Allocated research' : ['', '0.5% of sales', np.NaN] ,
        'Administrative expense' : ['', '2% of sales', np.NaN] ,
        'Management incentive compensation' : ['', '1.25% of sales', np.NaN] ,
    }

    # Create dataframe
    df_general = pd.DataFrame(dict_general).T
    df_general.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_general = df_general.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_general.index.name= 'General costs'  

    # Fill in costs
    df_general.loc['Selling (or transfer) expense', 'Cost ($/yr)'] = 0.01*(sales_USD_year)
    df_general.loc['Direct research','Cost ($/yr)'] = 0.048*(sales_USD_year)
    df_general.loc['Allocated research','Cost ($/yr)'] = 0.005*(sales_USD_year)
    df_general.loc['Administrative expense','Cost ($/yr)'] = 0.02*(sales_USD_year)
    df_general.loc['Management incentive compensation','Cost ($/yr)'] = 0.0125*(sales_USD_year)

    return df_general

# %% [markdown]
# ### 3.9 Opex summary

# %% [markdown]
# #### 3.9.1 Calculate totals and other units

# %%
# For all subparts of opex, calculate totals 

def totals(df,
          product_name,
          product_rate_kg_day ,
          capacity_factor):
    """
    Adds a totals row to a given df and fills in columns for cost/kg product, cost/day and cost/year

    Arguments: dataframe, product name of choice, production basis (kg product/day), capacity factor of plant operation ()

    Returns: None
    """

    df.loc['Total', 'Cost ($/yr)'] = df.select_dtypes(include=['int64', 'float64']).loc[:, 'Cost ($/yr)'].sum(axis=0)

    try:
        df['Cost ($/kg {})'.format(product_name)] = df['Cost ($/yr)']/(product_rate_kg_day*365*capacity_factor) # for utilities, this is calculated directly
        df['Cost ($/day)'] = df['Cost ($/yr)']/(365*capacity_factor)                
    except KeyError: # for some, cost was calculated per kg, and we now want to calculate it per day
        df['Cost ($/day)'] = df['Cost ($/kg {})'.format(product_name)]*product_rate_kg_day                
        df['Cost ($/yr)'] = df['Cost ($/day)']*(365*capacity_factor)                
    
    return

# %% [markdown]
# #### 3.9.2 Generate summary tables

# %%
@st.cache_data(ttl = "1h")
def opex_seider(df_feedstocks,
        df_utilities,
        df_sales,
        df_operations,
        df_maintenance,
        df_stack_replacement,
        df_overhead,
        df_taxes,
        df_depreciation,
        df_general,
        df_capex_totals,
        lifetime_years,
        capacity_factor,
        product_name,
        product_rate_kg_day,
        is_additional_opex,
        additional_opex_USD_kg
        ):
    """
    Calculates operating costs, mostly according to Seider, Lewin et. al.

    Arguments: Dataframe of feedstock cost, dataframe of utility cost, dataframe of sales, dataframe of operations costs, dataframe of maintenance costs,
    dataframe of stack replacement costs, dataframe of tax costs, dataframe of operating overheads, dataframe of depreciation costs, dataframe of general costs, 
    dataframe of total capital costs, plant lifetime (year), capacity factor of plant operation (),
    product name of choice, production basis (kg product/ day)

    Returns: dataframe of operating cost breakdown, dataframe of summarized total operating and levelized costs
    """

    ## SEIDER TEXTBOOK
    
    df_opex = pd.DataFrame(columns = ['Opex', 'Cost ($/yr)', 'Description']).astype({'Cost ($/yr)':'float64'})
    df_opex.set_index('Opex', inplace = True)     

    df_opex.loc['Feedstocks', 'Cost ($/yr)'] = df_feedstocks.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Feedstocks', 'Description'] = 'See Feedstocks'
    
    df_opex.loc['Utilities', 'Cost ($/yr)'] =  df_utilities.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Utilities', 'Description'] =  'See Utilities - used Seider book prices'

    df_opex.loc['Operations'] =  df_operations.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Maintenance'] = df_maintenance.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Stack replacement'] = df_stack_replacement.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Operating overhead'] =  df_overhead.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Property taxes and insurance'] =  df_taxes.loc['Total', 'Cost ($/yr)']
    # df_opex.loc['Depreciation'] = df_depreciation.loc['Total', 'Cost ($/yr)']
    df_opex.loc['General expenses'] =  df_general.loc['Total', 'Cost ($/yr)']
    if is_additional_opex:
        df_opex.loc['Additional operating cost'] =  additional_opex_USD_kg/(product_rate_kg_day*365*capacity_factor)

    if np.isclose(df_utilities.loc['Total', 'Cost ($/kg {})'.format(product_name)], 0.00): # if process does not exist (NaNs in FE/SPC for instance)
        df_opex['Cost ($/yr)'] = np.NaN

    df_opex['Cost ($/kg {})'.format(product_name)] = df_opex['Cost ($/yr)']/(product_rate_kg_day*365*capacity_factor)
    df_opex['Cost ($/day)'] = df_opex['Cost ($/yr)']/(365*capacity_factor)

    df_opex_totals = pd.DataFrame(columns = ['Cost ($/yr)']).astype({'Cost ($/yr)':'float64'})
    df_opex_totals.index.name = 'Opex summary'

    df_opex_totals.loc['Cost of manufacture'] = df_opex['Cost ($/yr)'].sum(axis=0) -  df_opex.loc['General expenses', 'Cost ($/yr)']
    df_opex_totals.loc['Production cost'] = df_opex['Cost ($/yr)'].sum(axis=0)
    df_opex_totals.loc['Levelized cost'] = df_opex_totals.loc['Production cost'] + df_capex_totals.loc['Total permanent investment', 'Cost ($)']/lifetime_years
    df_opex_totals.loc['Profit'] = df_sales.loc['Total', 'Earnings ($/yr)'] - df_opex_totals.loc['Levelized cost']
                                                           
    df_opex_totals['Cost ($/kg {})'.format(product_name)] = df_opex_totals['Cost ($/yr)']/(product_rate_kg_day*365*capacity_factor)
    df_opex_totals['Cost ($/day)'] = df_opex_totals['Cost ($/yr)']/(365*capacity_factor)

    return df_opex, df_opex_totals   

# %%
@st.cache_data(ttl = "1h")
def opex_sinnott(C_ISBL, # currently C_TBM
                 df_feedstocks,
                 df_utilities,
                 df_sales,
                 df_stack_replacement,
                 df_depreciation,
                 df_general,
                 df_capex_BM,
                 df_capex_totals,
                 lifetime_years,
                 capacity_factor,
                 product_name,
                 product_rate_kg_day,
                 additional_opex_USD_kg,
                 is_additional_opex,
                 ):
    """
    Calculates operating costs, mostly according to Sinnott and Towler.

    Arguments: Inside-battery-limits capital cost ($), dataframe of feedstock cost, dataframe of utility cost, 
    dataframe of sales, dataframe of stack replacement costs, dataframe of depreciation costs, dataframe of general costs, 
    dataframe of bare-module capex, dataframe of total capital costs, plant lifetime (year), capacity factor of plant operation (),
    product name of choice, production basis (kg product/ day)

    Returns: dataframe of operating cost breakdown, dataframe of summarized total operating and levelized costs
    """

    ## SINNOTT TEXTBOOK

    # Many costs are estimated based on the inside battery limits cost (ISBL), which excludes offsite, engineering and construction, and contingency costs
    df_opex = pd.DataFrame(columns = ['Opex', 'Cost ($/yr)', 'Description']).astype({'Cost ($/yr)':'float64'})
    df_opex.set_index('Opex', inplace = True)     
    
    df_opex.loc['Feedstocks', 'Cost ($/yr)'] = df_feedstocks.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Feedstocks', 'Description'] = 'See Feedstocks'
    
    df_opex.loc['Utilities', 'Cost ($/yr)'] =  df_utilities.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Utilities', 'Description'] =  'See Utilities - used Seider book prices'
    
    df_opex.loc['Operating labor', 'Cost ($/yr)'] = 4 * 80000
    df_opex.loc['Operating labor', 'Description'] = '$80000 per shift per year - Fluids processing, 3 sections + 1 controls'
    
    df_opex.loc['Supervision', 'Cost ($/yr)'] =  0.25 * df_opex.loc['Operating labor', 'Cost ($/yr)']
    df_opex.loc['Supervision', 'Description'] = '25% of operating labor'
    
    df_opex.loc['Direct salary overhead', 'Cost ($/yr)'] =  0.5 * (df_opex.loc['Operating labor', 'Cost ($/yr)'] + df_opex.loc['Supervision', 'Cost ($/yr)'])
    df_opex.loc['Direct salary overhead', 'Description'] =  '50% of operating labor + supervision'
    
    df_opex.loc['Maintenance', 'Cost ($/yr)'] = 0.04 * (C_ISBL - df_capex_BM.loc['Electrolyzer', 'Cost ($)'])
    df_opex.loc['Maintenance', 'Description'] =  '4% of ISBL plant cost'
    
    df_opex.loc['Stack replacement', 'Cost ($/yr)'] = df_stack_replacement.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Stack replacement', 'Description'] = 'See Stack replacement'
    
    df_opex.loc['Operating overhead', 'Cost ($/yr)'] = 0.65 * (df_opex.loc['Operating labor', 'Cost ($/yr)'] + df_opex.loc['Supervision', 'Cost ($/yr)']+ df_opex.loc['Maintenance', 'Cost ($/yr)'] + df_opex.loc['Direct salary overhead', 'Cost ($/yr)'])
    df_opex.loc['Operating overhead', 'Description'] = '65% of operating labor + supervision + labor overhead + maintenance'
    
    df_opex.loc['Property taxes and insurance', 'Cost ($/yr)'] = 0.02 * C_ISBL
    df_opex.loc['Property taxes and insurance', 'Description'] = '2% of ISBL plant cost'
    
    df_opex.loc['Environmental charges', 'Cost ($/yr)'] = 0.01 * C_ISBL
    df_opex.loc['Environmental charges', 'Description'] = '1% of ISBL plant cost'
    
    df_opex.loc['Land rent', 'Cost ($/yr)'] = 0.02 * (1.4 * C_ISBL)
    df_opex.loc['Land rent', 'Description'] = '2% of ISBL+OSBL, here 1.4 * ISBL'
    
    df_opex.loc['General expenses', 'Cost ($/yr)'] = df_general.loc['Total', 'Cost ($/yr)']
    df_opex.loc['General expenses', 'Description'] = 'See General Expenses - used Seider book'
    
    # df_opex.loc['Depreciation', 'Cost ($/yr)'] = df_depreciation.loc['Total', 'Cost ($/yr)']
    # df_opex.loc['Depreciation', 'Description'] = 'See Depreciation - used Seider book'
    if is_additional_opex:
        df_opex.loc['Additional', 'Cost ($/yr)'] =  additional_opex_USD_kg * (product_rate_kg_day*365*capacity_factor)
        df_opex.loc['Additional', 'Description'] =  '${}/kg {}'.format(additional_opex_USD_kg, product_name)

    if np.isclose(C_ISBL, 0.00): # if process does not exist (NaNs in FE/SPC for instance)
        df_opex['Cost ($/yr)'] = np.NaN

    df_opex['Cost ($/kg {})'.format(product_name)] = df_opex['Cost ($/yr)']/(product_rate_kg_day*365*capacity_factor)
    df_opex['Cost ($/day)'] = df_opex['Cost ($/yr)']/(365*capacity_factor)

    df_opex_totals = pd.DataFrame(columns = ['Cost ($/yr)']).astype({'Cost ($/yr)':'float64'})
    df_opex_totals.index.name = 'Opex summary'

    df_opex_totals.loc['Cost of manufacture'] = df_opex['Cost ($/yr)'].sum(axis=0) -  df_opex.loc['General expenses', 'Cost ($/yr)']
    df_opex_totals.loc['Production cost'] =  df_opex['Cost ($/yr)'].sum(axis=0)
    df_opex_totals.loc['Levelized cost'] = df_opex_totals.loc['Production cost'] + df_capex_totals.loc['Total permanent investment', 'Cost ($)']/lifetime_years
    df_opex_totals.loc['Profit'] = df_sales.loc['Total', 'Earnings ($/yr)'] - df_opex_totals.loc['Levelized cost']

    df_opex_totals['Cost ($/kg {})'.format(product_name)] = df_opex_totals['Cost ($/yr)']/(product_rate_kg_day*365*capacity_factor)
    df_opex_totals['Cost ($/day)'] = df_opex_totals['Cost ($/yr)']/(365*capacity_factor)
    
    return df_opex, df_opex_totals

# %% [markdown]
# ## 4. Time value, NPV and break-even costs

# %% [markdown]
# ### 4.1 Generate cashflow table

# %%
@st.cache_data(ttl = "1h")
def cashflow_years(    
    plant_lifetime,
    depreciation_schedule, # 'MACRS' or 'linear'
    D, # optional, used for MACRS only - depreciation%
    depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
    salvage_value, # optional, used for linear only - fraction of original capital that is recovered
    interest, # interest %
    f, # inflation %
    sales, # = df_sales.loc['Total', 'Earnings ($/yr)'],
    production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
    C_WC, # = 0,
    t, # = 26/100, # tax in % per year,
):
    """
    Calculate cashflow table for annual balance sheet
    
    Arguments: Plant lifetime (years), depreciation schedule - only 'linear' is programmed for now, 
    depreciation rate for MACRS (), depreciation period (years), salvage value ($), interest rate on invested returns (),
    inflation rate (), annual sales ($/year), production cost or annual opex ($/yr), 
    total depreciable capital ($), working capital ($), annual tax rate () 

    Returns: dataframe of cashflows
    """
    
    df_cashflows = pd.DataFrame(columns = ['Year', 'Capital cost', 'Working capital',
                                          'Production cost', 'Sales', 'Depreciation',
                                          'Salvage', 'Cash flow', 'Discounted cash flow', 'Cumulative']).astype(float)
    cashflows = []
    
    discount_factor_depreciation = 1/(1+interest)
    discount_factor = (1+f)/(1+interest)

    if plant_lifetime < 2:
        NPV = np.NaN
        df_cashflows = pd.DataFrame()
        
    else:
        
        for i in range(1, plant_lifetime+1):  
            # First year - capital
            if i == 1:
                capital_i = C_TDC
            else:
                capital_i = 0
            
            # First and last year - working capital
            if i == 1:
                WC_i = C_WC
            elif i == plant_lifetime:
                WC_i = -C_WC
            else:
                WC_i = 0
    
            # Production cost 
            C_i = production_cost

            # Sales
            S_i = sales
            
            # Depreciation
            if depreciation_schedule == 'MACRS':
                depreciation_i = D*C_TDC
            else:
                depreciation_i = (1 - salvage_value)*C_TDC/depreciation_lifetime

            # Salvage value
            if i == plant_lifetime:
                salvage_i = salvage_value*C_TDC
            else:
                salvage_i = 0
            
            # Overall cash flow
            cashflow_i = (1-t) * (S_i-C_i)  + depreciation_i - capital_i - WC_i  + salvage_i
            
            # Format into dataframe
            df_cashflows = pd.concat([df_cashflows, pd.DataFrame([[ i , capital_i,  WC_i,  C_i,  S_i,  depreciation_i,  salvage_i,  cashflow_i , 
                                                                           (cashflow_i - depreciation_i)*(discount_factor**(i-1)) + depreciation_i*(discount_factor_depreciation**(i-1)),
                                                                           0]], columns=df_cashflows.columns) ]
                                        , ignore_index=True)
            df_cashflows.loc[i-1, 'Cumulative'] = df_cashflows['Discounted cash flow'].loc[:i-1].sum(axis = 0)
            cashflows.append(cashflow_i)
    
        NPV = df_cashflows.iloc[-1,-1]
    
    return df_cashflows, cashflows, NPV

# %% [markdown]
# ### 4.2 Equation to determine the IRR

# %%
@st.cache_data(ttl = "1h")
def eqn_IRR(    
    x, # interest %
    plant_lifetime,
    depreciation_schedule, # 'MACRS' or 'linear'
    D, # optional, used for MACRS only - depreciation%
    depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
    salvage_value, # optional, used for linear only - fraction of original capital that is recovered
    f, # inflation %
    sales, # = df_sales.loc['Total', 'Earnings ($/yr)'],
    production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
    C_WC, # = 0,
    t, # = 26/100, # tax in % per year,
    ):
    """
    Equation NPV = 0 for calculating the internal rate of return such that NPV = 0.

    Arguments: Interest rate on invested returns (), plant lifetime (years), depreciation schedule - only 'linear' is programmed for now, 
    depreciation rate for MACRS (), depreciation period (years), salvage value ($), inflation rate (), annual sales ($/yr), production cost or annual opex ($/yr),
    production basis (kg/day), capacity factor of plant operation (), total depreciable capital ($), working capital ($), annual tax rate () 

    Returns: LHS of equation NPV = 0 
    """

    __, __, NPV = cashflow_years(    
        plant_lifetime = plant_lifetime,
        depreciation_schedule = depreciation_schedule, # 'MACRS' or 'linear'
        D = D, # optional, used for MACRS only - depreciation%
        depreciation_lifetime = depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
        salvage_value = salvage_value, # optional, used for linear only - fraction of original capital that is recovered
        interest = x, # determine interest rate (IRR)
        f = f, # inflation %
        sales = sales, # = df_sales.loc['Total', 'Earnings ($/yr)'],
        production_cost = production_cost,  # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
        C_TDC = C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
        C_WC = C_WC, # = 0,
        t = t, # = 26/100, # tax in % per year,
        )
    
    LHS = NPV

    return LHS

# %%
@st.cache_data(ttl = "1h")
def calculate_IRR(   
    plant_lifetime,
    depreciation_schedule, # 'MACRS' or 'linear'
    D, # optional, used for MACRS only - depreciation%
    depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
    salvage_value, # optional, used for linear only - fraction of original capital that is recovered
    f, # inflation %
    sales, # = df_sales.loc['Total', 'Earnings ($/yr)'],
    production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
    C_WC, # = 0,
    t, # = 26/100, # tax in % per year,
    ):
    """
    Calculate the internal rate of return such that NPV = 0.

    Arguments: Plant lifetime (years), depreciation schedule - only 'linear' is programmed for now, depreciation rate for MACRS (),
    depreciation period (years), salvage value ($), interest rate on invested returns (), inflation rate (), total annual sales ($/yr),
    production cost or process opex ($/year), production basis (kg/day), capacity factor of plant operation (),
    total depreciable capital ($), working capital ($), annual tax rate () 

    Returns: internal rate of return, IRR ()
    """

    if plant_lifetime < 2:
        IRR = np.NaN
        
    else:
        IRR = optimize.root_scalar(f = eqn_IRR,  
                                x0 = 0, x1 = 1,
                               args = (plant_lifetime,
                                        depreciation_schedule, # 'MACRS' or 'linear'
                                        D, # optional, used for MACRS only - depreciation%
                                        depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
                                        salvage_value, # optional, used for linear only - fraction of original capital that is recovered
                                        f, # inflation %
                                        sales, # = df_sales.loc['Total', 'Earnings ($/yr)'],
                                        production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
                                        C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
                                        C_WC, # = 0,
                                        t, # = 26/100, # tax in % per year,
                                        ),
                                        xtol = 1e-200
                                        ).root
    return IRR

# %% [markdown]
# ### 4.3 Calculate breakeven product price

# %%
@st.cache_data(ttl = "1h")
def eqn_breakeven_price(  
    x, # product price per kg
    plant_lifetime,
    depreciation_schedule, # 'MACRS' or 'linear'
    D, # optional, used for MACRS only - depreciation%
    depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
    salvage_value, # optional, used for linear only - fraction of original capital that is recovered
    interest, # interest %
    f, # inflation %
    product_rate_kg_day, # production in kg/day
    H2_rate_kg_day, # df_streams.loc['', '']
    capacity_factor, # capacity factor as a fraction of days in a year
    production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    H2_price_USD_kgH2, 
    C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
    C_WC, # = 0,
    t, # = 26/100, # tax in % per year,
    ):
    """
    Equation NPV = 0 for calculating the breakeven selling price such that NPV = 0.
    
    Arguments: Product price ($/kg product), plant lifetime (years), depreciation schedule - only 'linear' is programmed for now, 
    depreciation rate for MACRS (), depreciation period (years), salvage value ($), interest rate on invested returns (), 
    inflation rate (), production basis (kg/day), hydrogen production rate (kg/day), capacity factor of plant operation (), 
    production cost or process opex ($/year), hydrogen sale price ($/kg H2), total depreciable capital ($), 
    working capital ($), annual tax rate () 

    Returns: LHS - RHS of the equation NPV = 0
    """

    sales = (product_rate_kg_day * capacity_factor * 365 * x) + (H2_rate_kg_day * capacity_factor * 365 * H2_price_USD_kgH2)

    __, __, NPV = cashflow_years(    
        plant_lifetime = plant_lifetime,
        depreciation_schedule = depreciation_schedule, # 'MACRS' or 'linear'
        D = D, # optional, used for MACRS only - depreciation%
        depreciation_lifetime = depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
        salvage_value = salvage_value, # optional, used for linear only - fraction of original capital that is recovered
        interest = interest, # interest %
        f = f, # inflation %
        sales = sales, # = df_sales.loc['Total', 'Earnings ($/yr)'],
        production_cost  =production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
        C_TDC = C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
        C_WC = C_WC, # = 0,
        t = t, # = 26/100, # tax in % per year,
        )
        
    LHS = NPV

    return LHS

# %%
@st.cache_data(ttl = "1h")
def calculate_breakeven_price(
    plant_lifetime,
    depreciation_schedule, # 'MACRS' or 'linear'
    D, # optional, used for MACRS only - depreciation%
    depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
    salvage_value, # optional, used for linear only - fraction of original capital that is recovered
    interest, # interest %
    f, # inflation %
    product_rate_kg_day, # production in kg/day
    H2_rate_kg_day, # df_streams.loc['', '']
    capacity_factor, # capacity factor as a fraction of days in a year
    production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    H2_price_USD_kgH2, 
    C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
    C_WC, # = 0,
    t, # = 26/100, # tax in % per year,
    ):
    
    """
    Calculate the breakeven selling price such that NPV = 0.

    Arguments: Plant lifetime (years), depreciation schedule - only 'linear' is programmed for now, depreciation rate for MACRS (), 
    depreciation period (years), salvage value ($), interest rate on invested returns (), inflation rate (), production basis (kg/day), 
    hydrogen production rate (kg/day), capacity factor of plant operation (), production cost or process opex ($/year), 
    hydrogen sale price ($/kg H2), total depreciable capital ($), working capital ($), annual tax rate () 
    
    Returns: breakeven selling price of product ($/kg product)
    """

    if plant_lifetime < 2:
        breakeven_price_USD_kgprod = np.NaN
        
    else:
        breakeven_price_USD_kgprod = optimize.root_scalar(f = eqn_breakeven_price,  
                              x0 = 1, x1 = 2,
                              args = (plant_lifetime,
                                      depreciation_schedule, # 'MACRS' or 'linear'
                                      D, # optional, used for MACRS only - depreciation%
                                      depreciation_lifetime, # optional, used for linear only - total time before salvage value is recovered
                                      salvage_value, # optional, used for linear only - fraction of original capital that is recovered
                                      interest, # interest %
                                      f, # inflation %
                                      product_rate_kg_day, # production in kg/day
                                      H2_rate_kg_day, # df_streams.loc['', '']
                                      capacity_factor, # capacity factor as a fraction of days in a year
                                      production_cost, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
                                      H2_price_USD_kgH2, 
                                      C_TDC, # = df_capex_totals.loc['Total plant', 'Cost ($)'],
                                      C_WC, # = 0,
                                      t, # = 26/100, # tax in % per year,
                                      ),
                                      xtol = 1e-200
                                      ).root
    return breakeven_price_USD_kgprod

# %%
# def approx_ROI(
#     S, # = df_sales.loc['Total', 'Earnings ($/yr)'],
#     C, #= df_opex`_totals.loc['Production cost', 'Cost ($/yr)'], 
#     C_TCI, # = df_capex_totals.loc['Total plant', 'Cost ($)']
#     t, # = 26/100, # tax in % per year
#     i # interest rate
#         ):
    
#     ROI = (1-t) * (S-C) / C_TCI # net earnings / total capital investment
#     return ROI

# %%
# def approx_PBP(
#         C_TDC,
#         S, # = df_sales.loc['Total', 'Earnings ($/yr)'],
#         C, # = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
#         t, # = 26/100, # tax in % per year
#         D, # = 8/100, # straight-line depreciation, % per year)
#         ):
    
#     PBP = C_TDC/ ((1-t)*(S-C) + D)
#     return PBP


