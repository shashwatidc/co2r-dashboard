# %% [markdown]
# # Economic model
# Contact: Shashwati da Cunha, [shashwati.dc@utexas.edu](mailto:shashwati.dc@utexas.edu)
# 
# ### Instructions
# 1. Not designed for standalone run - this is only a collection of functions. Other notebooks call it.
# 

# %% [markdown]
# ## 0. Imports and setup

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
def nonaq_costing_assumptions(
    product_name,
    solvent_name,
    supporting_electrolyte_name,
    solvent_cost_USD_kg,
    electrolyte_cost_USD_kg,
    product_cost_USD_kgprod,
    H2_cost_USD_kgH2 ,
    electricity_cost_USD_kWh ,
    water_cost_USD_kg,
    CO2_cost_USD_tCO2,
    lifetime_years ,
    stack_lifetime_years,
    electrolyzer_capex_USD_m2,
    PSA_capex_USD_1000m3_hr,
    LL_capex_USD_1000mol_hr,
    capacity_factor,
    overridden_vbl
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
        '{}'.format(solvent_name) : [solvent_cost_USD_kg, '$/kg'],
        '{}'.format(supporting_electrolyte_name) : [electrolyte_cost_USD_kg, '$/kg'],
        'Plant lifetime': [lifetime_years, 'years'],
        'Stack lifetime': [stack_lifetime_years, 'years'],
        'Electrolyzer capex': [electrolyzer_capex_USD_m2, '$/m^2'],
        'PSA capex': [PSA_capex_USD_1000m3_hr, '$/1000 m^3/hr'],
        "Capacity factor" : [capacity_factor, ''],
    },
    ).transpose()

    df_costing_assumptions.columns = ['Cost', 'Units']
    df_costing_assumptions = df_costing_assumptions.astype({'Cost':'float64', 'Units':'string'})
    df_costing_assumptions.index.name = 'Assumed variable - inputs and costing'

    if 'Liquid separation capital cost' in overridden_vbl:
        df_costing_assumptions.loc['Liquid separation capital cost', 'Cost'] = LL_capex_USD_1000mol_hr
        df_costing_assumptions.loc['Liquid separation capital cost', 'Units'] = '$/1000 mol liquid/hr'

    return df_costing_assumptions

# %% [markdown]
# ## 1. Capital costs

# %% [markdown]
# ### 1.1 Bare-module capex of units

# %%
@st.cache_data(ttl = "1h")
def nonaq_capex(
    area_m2 , 
    df_products,
    product_name,
    solvent_name,
    supporting_electrolyte_name,
    electricity_kJ_per_kg,
    df_streams,
    product_rate_kg_day,
    battery_capex_USD_kWh ,
    electrolyzer_capex_USD_m2, # electrolyzer_capex_USD_kW
    PSA_capex_USD_1000m3_hr,
    LL_capex_USD_1000mol_hr,
    solvent_cost_USD_kg,
    electrolyte_cost_USD_kg,
    electrolyte_density_kg_m3,
    overridden_vbl,
    CO2_solubility_mol_mol,
    MW_supporting,
    MW_CO2,
    catholyte_conc_M,
    battery_capacity,
    kJ_per_kWh,
):
    """
    Calculate unit (bare-module) capital costs, and cumulative capital costs mostly according to the method of Seider, Lewin et. al.

    Arguments: electrolyzer area (m^2), electricity required per kg product, dataframe of stream table, production basis (kg/day),
    battery capex ($/kWh), electrolyzer capex ($/m^2), battery pseudo-"capacity factor" or fraction of time when battery powers plant (), 
    kJ/kWh conversion

    Returns: dataframe of bare-module costs per unit, and summary capital costs
    """
    
    ## Battery limits ("on-site") are the electrolysis and separations only. 
    # => There are no off-site units included, i.e. feedstocks (CO2, DI water) and utilities (electricity) come from external vendors 
    
    ## Bare module costs of onsite equipment (includes process equipment which is customized, like pressure vessels; and off-the-shelf process machinery, like pumps)
    dict_capex_BM = {
        'Electrolyzer' : ['Electrolysis', '${}/m2 - Linear scaling, Badgett Cortright J Cleaner Prod 2022'.format(electrolyzer_capex_USD_m2), np.NaN],
        'Solvent' : ['Electrolysis', '${}/kg {}'.format(solvent_cost_USD_kg, solvent_name), np.NaN],
        'Supporting electrolyte' : ['Electrolysis', '${}/kg {}'.format(electrolyte_cost_USD_kg, supporting_electrolyte_name), np.NaN],
        #         'Deionization' : ['', '', 0] , # TODO: add deionization capex and opex
        'Balance of plant' : ['Pressure changes etc', '34% of electrolyzer bare module - Linear scaling, H2A model', np.NaN] ,
        # 'Anode PSA - CO$_2$/O$_2$' : ['Separations', 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', np.NaN] ,
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
    
    # Cost of electrolyte
    df_capex_BM.loc['Solvent', 'Cost ($)'] = 2 * solvent_cost_USD_kg * electrolyte_density_kg_m3 * df_streams.loc[ 'Liquid drier inlet', 'Volumetric flow rates (m3/s)'] * (10*60) # Assume 2x drier volume is needed for looping
    df_capex_BM.loc['Supporting electrolyte', 'Cost ($)'] = 2 * electrolyte_cost_USD_kg * MW_supporting/1000 * (catholyte_conc_M*1000) * df_streams.loc[ 'Liquid drier inlet', 'Volumetric flow rates (m3/s)'] * (10*60) # Assume 2x drier volume is needed for looping

    # H2A - balance of plant is 34.5% of electrolyzer capex
    df_capex_BM.loc['Balance of plant', 'Cost ($)']  = df_capex_BM.loc['Electrolyzer', 'Cost ($)'] * 0.345

    # Catholyte drier - two parallel vessels. Medium cost is included in maintenance
    df_capex_BM.loc['Catholyte drier'] = {'Stage': 'Solvent recycle', 'Description' : 'Two stainless steel cone-roof tanks', 
                                                             'Cost ($)' : np.NaN} 
    df_capex_BM.loc['Catholyte drier', 'Cost ($)'] = 800/509.7 * 2 * 1.3 * (5000  + 1400 * (df_streams.loc[ 'Liquid drier inlet', 'Volumetric flow rates (m3/s)'] * (10*60))**0.7) # 1.3 factor for stainless steel, 10 minute residence time, costed as a cone roof tank from Sinnott et al
    df_capex_BM.loc['Catholyte drier', 'Cost ($)'] += 2 * 8 * 700 * (df_streams.loc[ 'Liquid drier inlet', 'Volumetric flow rates (m3/s)'] * (10*60)) # Add the additional charge of the drier dessicant MS4A - $8/kg, 700 kg/m3

    ## TODO: Are these FOB costs  (without delivery)? Or including delivery, installation, material factors (including bare-module factor) to give bare-module costs?
    # PSA - reference cost with scaling factor
    if df_products.loc[product_name, 'Phase'] == 'gas':
        df_capex_BM.loc['Cathode PSA - CO$_2$/products'] =  {'Stage': 'Separations', 'Description' : 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', 
                                                             'Cost ($)' : np.NaN} 
        df_capex_BM.loc['Cathode PSA - Products/H$_2$'] = {'Stage': 'Separations', 'Description' : 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', 
                                                             'Cost ($)' : np.NaN} 
        df_capex_BM.loc['Cathode PSA - CO$_2$/products', 'Cost ($)']  = PSA_capex_USD_1000m3_hr * (df_streams.loc['Cathode PSA1 inlet', 'Volumetric flow rates (m3/s)']*(60*60)/1000)**0.7 # relative to 1000 m3/hr 
        df_capex_BM.loc['Cathode PSA - Products/H$_2$', 'Cost ($)']  = PSA_capex_USD_1000m3_hr * (df_streams.loc['Cathode PSA1 outlet', 'Volumetric flow rates (m3/s)']*(60*60)/1000)**0.7 # relative to 1000 m3/hr 
    elif df_products.loc[product_name, 'Phase'] == 'liquid':
        df_capex_BM.loc['Cathode PSA - CO$_2$/H$_2$'] = {'Stage': 'Separations', 'Description' : 'Scaling factor 0.7 to Shin Jiao Nat Sust 2021', 
                                                             'Cost ($)' : np.NaN} 
        df_capex_BM.loc['Cathode PSA - CO$_2$/H$_2$', 'Cost ($)']  = PSA_capex_USD_1000m3_hr * (df_streams.loc['Cathode PSA1 inlet', 'Volumetric flow rates (m3/s)']*(60*60)/1000)**0.7 # relative to 1000 m3/hr 
        # Liquid-liquid separations - distillation only - formate (Shin et al)
        if product_name == 'Oxalic acid': # Distillation reference capex depends on the product necessary
            df_capex_BM.loc['GASP - (COOH)$_2$/Catholyte'] =  {'Stage': 'Separations', 'Description' : 'Scaling factor 0.7 from Boor 2023', 
                                                                'Cost ($)' : np.NaN} 
            if 'Liquid separation capital cost' in overridden_vbl:
                print('Costing L/L separation as a general separation')
                df_capex_BM.loc['GASP - (COOH)$_2$/Catholyte', 'Cost ($)'] = LL_capex_USD_1000mol_hr * (df_streams.loc['L/L separation inlet', 'Molar flow rate (mol/s)']*60*60/1000)**0.7
            else:
                CO2_GASP_mol_hr = (df_streams.loc['L/L separation inlet', 'Molar flow rate (mol/s)']*df_streams.loc['L/L separation inlet', 'x_{}'.format(solvent_name)]*60*60) * CO2_solubility_mol_mol
                CO2_GASP_kg_hr = CO2_GASP_mol_hr*(MW_CO2/1000)
                df_capex_BM.loc['GASP - (COOH)$_2$/Catholyte', 'Cost ($)'] = 800/444.2 * 1.3 * 1.24 * 950000 * (CO2_GASP_kg_hr/2160)**0.7 # relative to 1000 mol solvent/hr, 1.3 factor for stainless steel             
                
                print('Costing L/L separation as GASP, {} molCO2/hr'.format(CO2_GASP_mol_hr))

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

# %% [markdown]
# ## 3. Operating expenses

# %% [markdown]
# ### 3.1 Feedstocks

# %%
@st.cache_data(ttl = "1h")
def nonaq_feedstocks(
    CO2_cost_USD_tCO2,
    water_cost_USD_kg,
    solvent_name,
    solvent_cost_USD_kg,
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
    df_feedstocks.loc['{}'.format(solvent_name),'Cost ($/yr)'] = solvent_cost_USD_kg*df_streams.loc['Fresh solvent feed', 'Mass flow rate (kg/day)']*365*capacity_factor #df_streams[]

    df_feedstocks
    
    return df_feedstocks

# %% [markdown]
# ### 3.4 Maintenance and replacement

# %%
@st.cache_data(ttl = "1h")
def nonaq_maintenance(
    C_TDC,
    df_capex_BM,
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
    df_maintenance.loc['Maintenance wages and benefits (MW&B)', 'Cost ($/yr)'] = 0.035* ( C_TDC - (df_capex_BM.loc['Electrolyzer', 'Cost ($)'] + df_capex_BM.loc['Supporting electrolyte', 'Cost ($)'])*1.18)
    df_maintenance.loc['Maintenance salaries and benefits','Cost ($/yr)'] = 0.25*df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)']
    df_maintenance.loc['Materials and services', 'Cost ($/yr)'] = 1.00*df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)']
    df_maintenance.loc['Maintenance overhead', 'Cost ($/yr)'] = 0.05*df_maintenance.loc['Maintenance wages and benefits (MW&B)','Cost ($/yr)']

    return df_maintenance

# %%
@st.cache_data(ttl = "1h")
def nonaq_replacement(df_streams,
                      df_capex_BM,
                      stack_lifetime_years,
                      lifetime_years):
    """
    Calculate stack, dessicant, and supporting electrolyte replacement cost, annualized

    Arguments: electrolyzer installed bare-module cost ($), stack lifetime after which full replacement is necessary (years), 
    total plant lifetime (years)

    Returns: dataframe of stack replacement cost
    """
         
    
    # Create dictionary
    dict_replacement = {
        'Stack replacement' : ['', 'Full electrolyzer cost after {} years'.format(stack_lifetime_years), np.NaN],
        'Dessicant replacement' : ['', 'Replace after 2 years - http://www.jmcampbell.com/tip-of-the-month/2019/05/a-short-cut-method-for-evaluating-molecular-sieve-performance/', np.NaN],
        'Supporting electrolyte replacement' : ['', 'Assume 5-year lifetime = 0.2 replacements per year', np.NaN]
    }

    # Create dataframe
    df_replacement = pd.DataFrame(dict_replacement).T
    df_replacement.columns = ['Stage', 'Description', 'Cost ($/yr)']
    df_replacement = df_replacement.astype({'Stage':'string', 'Description': 'string', 'Cost ($/yr)':'float64'})
    df_replacement.index.name= 'Stack and material replacement'  

    # Fill in costs
    no_of_replacements = max( 0, (lifetime_years // stack_lifetime_years - 1))
    df_replacement.loc['Stack replacement', 'Cost ($/yr)'] = df_capex_BM.loc['Electrolyzer', 'Cost ($)'] * no_of_replacements / lifetime_years 
    # Total replacement cost / plant lifetime
    
    no_of_replacements = max( 0, (lifetime_years // 2 - 1))
    df_replacement.loc['Dessicant replacement', 'Cost ($/yr)'] = 8 * (df_streams.loc[ 'Liquid drier inlet', 'Volumetric flow rates (m3/s)'] * (10*60)) * 700 * no_of_replacements / lifetime_years
     # 1 replacement every 2 years, $6-8/kg for MS4A, 0.7 kg/L, good for drying nonpolar liquids e.g. https://www.impakcorporation.com/641A4MS55-13
    
    no_of_replacements = max( 0, (lifetime_years // 5 - 1))
    df_replacement.loc['Supporting electrolyte replacement', 'Cost ($/yr)'] = df_capex_BM.loc['Supporting electrolyte', 'Cost ($)'] * no_of_replacements / lifetime_years
    # 5 year replacement

    return df_replacement


# %% [markdown]
# #### 3.9.2 Generate summary tables

# %%
@st.cache_data(ttl = "1h")
def nonaq_opex_seider(df_feedstocks,
        df_utilities,
        df_sales,
        df_operations,
        df_maintenance,
        df_replacement,
        df_overhead,
        df_taxes,
        df_depreciation,
        df_general,
        df_capex_totals,
        lifetime_years,
        capacity_factor,
        product_name,
        product_rate_kg_day
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
    df_opex.loc['Stack and material replacement'] = df_replacement.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Operating overhead'] =  df_overhead.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Property taxes and insurance'] =  df_taxes.loc['Total', 'Cost ($/yr)']
    # df_opex.loc['Depreciation'] = df_depreciation.loc['Total', 'Cost ($/yr)']
    df_opex.loc['General expenses'] =  df_general.loc['Total', 'Cost ($/yr)']

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
def nonaq_opex_sinnott(C_ISBL, # currently C_TBM
                 df_feedstocks,
                 df_utilities,
                 df_sales,
                 df_replacement,
                 df_depreciation,
                 df_general,
                 df_capex_BM,
                 df_capex_totals,
                 lifetime_years,
                 capacity_factor,
                 product_name,
                 product_rate_kg_day
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
    df_opex.loc['Maintenance', 'Description'] =  '4% of ISBL plant cost minus electrolyzer'
    
    df_opex.loc['Stack and material replacement', 'Cost ($/yr)'] = df_replacement.loc['Total', 'Cost ($/yr)']
    df_opex.loc['Stack and material replacement', 'Description'] = 'See Stack and material replacement'
    
    df_opex.loc['Operating overhead', 'Cost ($/yr)'] = 0.65 * (df_opex.loc['Operating labor', 'Cost ($/yr)'] + df_opex.loc['Supervision', 'Cost ($/yr)']+ df_opex.loc['Direct salary overhead', 'Cost ($/yr)'] + df_opex.loc['Maintenance', 'Cost ($/yr)'])
    df_opex.loc['Operating overhead', 'Description'] = '65% of operating labor + supervision + labor overhead + maintenance'
    
    df_opex.loc['Property taxes and insurance', 'Cost ($/yr)'] = 0.02 * C_ISBL
    df_opex.loc['Property taxes and insurance', 'Description'] = '2% of ISBL plant cost'
    
    df_opex.loc['Environmental charges', 'Cost ($/yr)'] = 0.01 * C_ISBL
    df_opex.loc['Environmental charges', 'Description'] = '1% of ISBL plant cost'
    
    df_opex.loc['Land rent', 'Cost ($/yr)'] = 0.02 * C_ISBL
    df_opex.loc['Land rent', 'Description'] = '2% of ISBL+OSBL, here ISBL only'
    
    df_opex.loc['General expenses', 'Cost ($/yr)'] = df_general.loc['Total', 'Cost ($/yr)']
    df_opex.loc['General expenses', 'Description'] = 'See General Expenses - used Seider book'
    
    # df_opex.loc['Depreciation', 'Cost ($/yr)'] = df_depreciation.loc['Total', 'Cost ($/yr)']
    # df_opex.loc['Depreciation', 'Description'] = 'See Depreciation - used Seider book'

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
