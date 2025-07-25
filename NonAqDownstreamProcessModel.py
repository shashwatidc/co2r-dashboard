# %% [markdown]
# # Downstream model
# Contact: Shashwati da Cunha, [shashwati.dc@utexas.edu](mailto:shashwati.dc@utexas.edu)
# 
# ### Instructions
# 1. Not designed for standalone run - this is only a collection of functions. Other notebooks call it.
# 

# %% [markdown]
# ## 0. Imports and setup

# %% [markdown]
# ### 0.1 Imports and styling

# %%
# UNCOMMENT TO RUN STANDALONE

import streamlit as st
import pandas as pd
import numpy as np
from DownstreamProcessModel import *

# %% [markdown]
# ## 1. Stream table

# %% [markdown]
# ### 1.1 Create blank stream table

# %%
@st.cache_data(ttl = "1h")
def nonaq_blank_stream_table(df_products,
                       product_name):
    """
    Generates a skeleton dataframe for streams and hard codes stream numbers. These are based on the assumed PFD

    Arguments: product name of choice
    
    Returns: empty dataframe with rows = streams and columns = stream variables; a few variables may be added on later
    """
  
    df_streams = pd.DataFrame(columns = ['Stream number', 'Description', 'Temperature (∘C)', 'Phase', 'Pressure (Pa)',
                                         'Mass flow rate (kg/day)', 'Molar flow rate (mol/s)', 
                                         'x_{}'.format(product_name), 'x_H2', 'x_H2O', 'x_CO2', 'x_O2', 'x_MX']).astype({
        'Stream number':'int', 
        'Description':'string',
        'Temperature (∘C)':'float64', 
        'Phase':'string', 
        'Pressure (Pa)':'float64',
        'Mass flow rate (kg/day)':'float64', 
        'Molar flow rate (mol/s)':'float64', 
        'x_{}'.format(product_name):'float64', 
        'x_H2':'float64', 
        'x_H2O':'float64', 
        'x_CO2':'float64', 
        'x_O2':'float64',
        'x_MX': 'float64',
    })

    if df_products.loc[product_name, 'Phase'] == 'gas':
        df_streams.loc['Fresh CO2 feed', 'Stream number'] = 1
        df_streams.loc['Cathode gas inlet', 'Stream number'] = 2
        df_streams.loc['Cathode gas outlet', 'Stream number'] = 3
        df_streams.loc['Anode inlet', 'Stream number'] = 4
        df_streams.loc['Anode outlet', 'Stream number'] = 5
        df_streams.loc['Anolyte recycle', 'Stream number'] = 6
        # df_streams.loc['Anode PSA inlet', 'Stream number'] = 7
        df_streams.loc['O2 outlet', 'Stream number'] = 8
        # df_streams.loc['Anode CO2 recycle', 'Stream number'] = 9
        df_streams.loc['Fresh water feed', 'Stream number'] = 10
        df_streams.loc['Total DI feed', 'Stream number'] = 11
        df_streams.loc['Cathode CO2 recycle', 'Stream number'] = 12
        df_streams.loc['Cathode PSA1 inlet', 'Stream number'] = 13
        df_streams.loc['Cathode gas water recycle', 'Stream number'] = 14
        df_streams.loc['Cathode PSA1 outlet', 'Stream number'] = 15
        df_streams.loc['Product outlet', 'Stream number'] = 16
        df_streams.loc['H2 outlet', 'Stream number'] = 17

        df_streams.loc['Fresh solvent feed', 'Stream number'] = 18
        df_streams.loc['Cathode liquid inlet', 'Stream number'] = 19
        df_streams.loc['Cathode liquid outlet', 'Stream number'] = 20
        df_streams.loc['Solvent offgas', 'Stream number'] = 21

        df_streams.loc['Liquid drier inlet', 'Stream number'] = 24
        df_streams.loc['Catholyte recycle', 'Stream number'] = 25
        df_streams.loc['Catholyte water recycle', 'Stream number'] = 26
    
    if df_products.loc[product_name, 'Phase'] == 'liquid':
        df_streams.loc['Fresh CO2 feed', 'Stream number'] = 1
        df_streams.loc['Cathode gas inlet', 'Stream number'] = 2
        df_streams.loc['Cathode gas outlet', 'Stream number'] = 3
        df_streams.loc['Anode inlet', 'Stream number'] = 4
        df_streams.loc['Anode outlet', 'Stream number'] = 5
        df_streams.loc['Anolyte recycle', 'Stream number'] = 6
        # df_streams.loc['Anode PSA inlet', 'Stream number'] = 7
        df_streams.loc['O2 outlet', 'Stream number'] = 8
        # df_streams.loc['Anode CO2 recycle', 'Stream number'] = 9
        df_streams.loc['Fresh water feed', 'Stream number'] = 10
        df_streams.loc['Total DI feed', 'Stream number'] = 11
        df_streams.loc['Cathode CO2 recycle', 'Stream number'] = 12
        df_streams.loc['Cathode PSA1 inlet', 'Stream number'] = 13
        df_streams.loc['Cathode gas water recycle', 'Stream number'] = 14
        # df_streams.loc['Cathode PSA1 outlet', 'Stream number'] = 15
        # df_streams.loc['Product outlet', 'Stream number'] = 16

        df_streams.loc['H2 outlet', 'Stream number'] = 17
        df_streams.loc['Fresh solvent feed', 'Stream number'] = 18
        df_streams.loc['Cathode liquid inlet', 'Stream number'] = 19
        df_streams.loc['Cathode liquid outlet', 'Stream number'] = 20
        df_streams.loc['Solvent offgas', 'Stream number'] = 21
        df_streams.loc['L/L separation inlet', 'Stream number'] = 22
        df_streams.loc['Product outlet', 'Stream number'] = 23
        df_streams.loc['Liquid drier inlet', 'Stream number'] = 24
        df_streams.loc['Catholyte recycle', 'Stream number'] = 25
        df_streams.loc['Catholyte water recycle', 'Stream number'] = 26
    
    return df_streams

# %% [markdown]
# ### 1.2 Electrolyzer and separator units for mass balances

# %%
@st.cache_data(ttl = "1h")
def nonaq_update_stream_table(product_name, 
                        solvent_name,
                        supporting_electrolyte_name,
                        df_products, # product data - MW 
                        df_streams, 
                        df_electrolyzer_streams_mol_s,
                        catholyte_conc_M,
                        anolyte_conc_M,
                        solvent_loss_fraction,
                        water_density_kg_m3,
                        electrolyte_density_kg_m3,
                        cathode_outlet_humidity,
                        T_streams,
                        T_sep,
                        P,
                        MW_CO2,
                        MW_H2O,
                        MW_O2,
                        MW_MX,
                        MW_solvent,
                        MW_supporting,
                        R,
                        K_to_C):
    
    """
    Fills in stream table for given product, electrolyzer streams run
    
    Arguments: product name of choice, product dataframe with [product name, molecular weights (g/mol)],
    blank df_streams dataframe with stream names filled in and product mole flow rates and mole fraction columns, 
    (this will be filled with data), dataframe with mole flow rates for electrolyzer streams, electrolyte concentration (M),
    electrolyte density (kg/m3), humidity of the cathode outlet stream (), temperature of streams (K), temperature of separation inlets (K),
    pressure of all streams (Pa), molecular weight for CO2 (g/mol), molecular weight for H2O (g/mol), molecular weight for O2 (g/mol),
    molecular weight of electrolyte salt (g/mol), gas constant (J/mol.K), conversion from K to C (273.15)

    Returns a dataframe of flow rates and mole fractions for all streams
    """

    ### Extract data on specific product
    MW_product = df_products.loc[product_name, 'Molecular weight (g/mol)']
    MW_H2 = df_products.loc['H2', 'Molecular weight (g/mol)']    

    # Update fresh CO2 feed to cathode gas inlet
    df_streams.loc['Fresh CO2 feed', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['CO2 fresh']
    df_streams.loc['Fresh CO2 feed', ['x_{}'.format(product_name),
                                    'x_H2',
                                    'x_H2O',
                                    'x_CO2',
                                    'x_O2',
                                    'x_MX',
                                    'x_{}'.format(solvent_name),
                                    'x_{}'.format(supporting_electrolyte_name)
                                    ]] = [0, 0, 0, 1, 0, 0, 0, 0]

    # Update recycled CO2 to cathode gas inlet ## TODO: make wet
    df_streams.loc['Cathode gas inlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['CO2 inlet']
    df_streams.loc['Cathode gas inlet', ['x_{}'.format(product_name),
                                    'x_H2',
                                    'x_H2O',
                                    'x_CO2',
                                    'x_O2',
                                    'x_MX',
                                    'x_{}'.format(solvent_name),
                                    'x_{}'.format(supporting_electrolyte_name)
                                    ]] = [0, 0, 0, 1, 0, 0, 0, 0]

    # Update anode anolyte inlet
    water_molarity = (water_density_kg_m3/1000)/(MW_H2O/1000) # 1 kg/L / 0.018 kg/mol
    
    df_streams.loc[ 'Anode inlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['Water inlet']/(water_molarity/(anolyte_conc_M+water_molarity))
    df_streams.loc[ 'Anode inlet', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [0, 
                                                      0, 
                                                      water_molarity/(anolyte_conc_M+water_molarity), 
                                                      0, 
                                                      0, 
                                                      anolyte_conc_M/(anolyte_conc_M+water_molarity), 
                                                      0, 
                                                      0]

    # Update cathode outlet to drier inlet
    df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['Cathode gas outlet']
    df_streams.loc[ 'Cathode gas outlet', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [df_electrolyzer_streams_mol_s['Product gas outlet']/df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)'],
                                                            df_electrolyzer_streams_mol_s['H2 outlet']/df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)'],
                                                            cathode_outlet_humidity, 
                                                            df_electrolyzer_streams_mol_s['CO2 cathode outlet']/df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)'],
                                                            0,
                                                            0, 
                                                            0, 
                                                            0]
    
    # Update cathode drier outlet to DI column
    df_streams.loc[ 'Cathode gas water recycle', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode gas outlet', 'x_H2O']
    df_streams.loc[ 'Cathode gas water recycle', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            1, 
                                                            0,
                                                            0,
                                                            0, 
                                                            0, 
                                                            0]                                                       
    
    # Update drier outlet to cathode PSA1 inlet
    df_streams.loc[ 'Cathode PSA1 inlet', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)'] - df_streams.loc[ 'Cathode gas water recycle', 'Molar flow rate (mol/s)']  
    df_streams.loc[ 'Cathode PSA1 inlet', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [df_electrolyzer_streams_mol_s['Product gas outlet']/df_streams.loc[ 'Cathode PSA1 inlet', 'Molar flow rate (mol/s)'],
                                                            df_electrolyzer_streams_mol_s['H2 outlet']/df_streams.loc[ 'Cathode PSA1 inlet', 'Molar flow rate (mol/s)'],
                                                            0,
                                                            df_electrolyzer_streams_mol_s['CO2 cathode outlet']/df_streams.loc[ 'Cathode PSA1 inlet', 'Molar flow rate (mol/s)'],
                                                            0,
                                                            0,
                                                            0, 
                                                            0
                                                           ]

    # Update fresh water feed
    df_streams.loc[ 'Fresh water feed', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['Water makeup']
    df_streams.loc[ 'Fresh water feed', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            1, 
                                                            0,
                                                            0,
                                                            0, 
                                                            0, 
                                                            0]                                                       

    
    # Update cathode PSA1 CO2 outlet to cathode recycle
    df_streams.loc[ 'Cathode CO2 recycle', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['CO2 cathode outlet']
    df_streams.loc[ 'Cathode CO2 recycle', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            0,
                                                            1,
                                                            0,
                                                            0, 
                                                            0, 
                                                            0]   

    # Update cathode PSA1 outlet to PSA2 inlet
    if df_products.loc[product_name, 'Phase'] == 'gas':
      df_streams.loc[ 'Cathode PSA1 outlet', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode PSA1 inlet', 'Molar flow rate (mol/s)']-df_streams.loc[ 'Cathode CO2 recycle', 'Molar flow rate (mol/s)']
      df_streams.loc[ 'Cathode PSA1 outlet', ['x_{}'.format(product_name),
                                                            'x_H2',
                                                            'x_H2O',
                                                            'x_CO2',
                                                            'x_O2',
                                                            'x_MX',
                                                            'x_{}'.format(solvent_name),
                                                            'x_{}'.format(supporting_electrolyte_name)
                                                            ]] = [df_electrolyzer_streams_mol_s['Product gas outlet']/df_streams.loc[ 'Cathode PSA1 outlet', 'Molar flow rate (mol/s)'],
                                                                  df_electrolyzer_streams_mol_s['H2 outlet']/df_streams.loc[ 'Cathode PSA1 outlet', 'Molar flow rate (mol/s)'],
                                                                  0, 
                                                                  0,
                                                                  0,
                                                                  0, 
                                                                  0, 
                                                                  0]

    # Update product outlet
    if df_products.loc[product_name, 'Phase'] == 'gas':
      df_streams.loc[ 'Product outlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['Product gas outlet']
    elif df_products.loc[product_name, 'Phase'] == 'liquid':
      df_streams.loc[ 'Product outlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['Product liquid outlet']
    df_streams.loc[ 'Product outlet', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [1,
                                                            0,
                                                            0,
                                                            0,
                                                            0,
                                                            0, 
                                                            0, 
                                                            0]

    # Update cathode PSA2 H2 outlet
    df_streams.loc[ 'H2 outlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['H2 outlet']
    df_streams.loc[ 'H2 outlet', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            1,
                                                            0,
                                                            0,
                                                            0,
                                                            0, 
                                                            0, 
                                                            0]

    # Update O2 outlet
    df_streams.loc[ 'O2 outlet', 'Molar flow rate (mol/s)'] =df_electrolyzer_streams_mol_s['O2 outlet']
    df_streams.loc[ 'O2 outlet', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [0,
                                                      0,
                                                      0, 
                                                      0,
                                                      1,
                                                      0, 
                                                      0, 
                                                      0]

    # Update catholyte inlet 
    solvent_molarity = (electrolyte_density_kg_m3/1000)/(MW_solvent/1000) # e.g. 1 kg/L / 0.018 kg/mol
    
    df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['Solvent inlet']/(solvent_molarity/(catholyte_conc_M+solvent_molarity))
    df_streams.loc[ 'Cathode liquid inlet', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [0, 
                                                      0, 
                                                      0, 
                                                      0, 
                                                      0, 
                                                      0,
                                                      solvent_molarity/(catholyte_conc_M+solvent_molarity), 
                                                      catholyte_conc_M/(catholyte_conc_M+solvent_molarity), 
                                                      ]

    # Update catholyte outlet
    water_content_catholyte_outlet = 0.2 # assume 20% of the solvent mole fraction is water
    df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)'] + water_content_catholyte_outlet*df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid inlet', 'x_{}'.format(solvent_name)] + df_electrolyzer_streams_mol_s['Liquid product outlet']
    df_streams.loc[ 'Cathode liquid outlet', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX', 
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [df_electrolyzer_streams_mol_s['Liquid product outlet']/df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)'], 
                                                      0, 
                                                      water_content_catholyte_outlet*df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid inlet', 'x_{}'.format(solvent_name)]/df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)'],
                                                      0, 
                                                      0, 
                                                      0, 
                                                      df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid inlet', 'x_{}'.format(solvent_name)] /df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)'],
                                                      df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid inlet', 'x_{}'.format(supporting_electrolyte_name)] /df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)']
                                                      ]

    # Update solvent offgas
    df_streams.loc[ 'Solvent offgas', 'Molar flow rate (mol/s)'] = solvent_loss_fraction * df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid inlet', 'x_{}'.format(solvent_name)] # Assume % loss of solvent 
    df_streams.loc[ 'Solvent offgas', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            0, 
                                                            0,
                                                            0,
                                                            0, 
                                                            1, 
                                                            0]

    # Update fresh solvent feed
    df_streams.loc[ 'Fresh solvent feed', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Solvent offgas', 'Molar flow rate (mol/s)']
    df_streams.loc[ 'Fresh solvent feed', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            0, 
                                                            0,
                                                            0,
                                                            0, 
                                                            1, 
                                                            0]

    # Update catholyte recycle
    df_streams.loc['Catholyte recycle', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode liquid inlet', 'Molar flow rate (mol/s)'] - df_streams.loc['Fresh solvent feed', 'Molar flow rate (mol/s)'] 
    df_streams.loc['Catholyte recycle', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [0,
                                                      0, 
                                                      0,
                                                      0, 
                                                      0, 
                                                      0,
                                                      df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid outlet', 'x_{}'.format(supporting_electrolyte_name)]/df_streams.loc[ 'Catholyte recycle', 'Molar flow rate (mol/s)']
                                                      ]   
    df_streams.loc[ 'Catholyte recycle', 'x_{}'.format(solvent_name)] = 1.0 - df_streams.loc['Catholyte recycle', 'x_{}'.format(supporting_electrolyte_name)]

    # Update liquid drier inlet
    df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)'] - df_streams.loc[ 'Solvent offgas', 'Molar flow rate (mol/s)'] - df_electrolyzer_streams_mol_s['Liquid product outlet']
    df_streams.loc['Liquid drier inlet', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [0,
                                                      0, 
                                                      df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid outlet', 'x_H2O']/df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)'],
                                                      0, 
                                                      0, 
                                                      0,
                                                      ((df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid outlet', 'x_{}'.format(solvent_name)]) - df_streams.loc[ 'Solvent offgas', 'Molar flow rate (mol/s)'])/df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)'],
                                                      df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Cathode liquid outlet', 'x_{}'.format(supporting_electrolyte_name)]/df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)'],
                                                      ]   
    
    # Update Catholyte water recycle
    df_streams.loc[ 'Catholyte water recycle', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Liquid drier inlet', 'x_H2O']
    df_streams.loc['Catholyte water recycle', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [0,
                                                      0, 
                                                      1,
                                                      0, 
                                                      0, 
                                                      0,
                                                      0,
                                                      0,
                                                      ]   
   
    # Update total DI
    df_streams.loc[ 'Total DI feed', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Fresh water feed', 'Molar flow rate (mol/s)'] + df_streams.loc[ 'Cathode gas water recycle', 'Molar flow rate (mol/s)'] + df_streams.loc[ 'Catholyte water recycle', 'Molar flow rate (mol/s)']
    df_streams.loc[ 'Total DI feed', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_H2O',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            1, 
                                                            0,
                                                            0,
                                                            0, 
                                                            0, 
                                                            0] 
        
    if df_products.loc[product_name, 'Phase'] == 'liquid':
       # Update L/L separation inlet
       df_streams.loc[ 'L/L separation inlet', 'Molar flow rate (mol/s)'] = df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)'] - df_streams.loc[ 'Solvent offgas', 'Molar flow rate (mol/s)']
       df_streams.loc[ 'L/L separation inlet', ['x_{}'.format(product_name),
                                                'x_H2',
                                                'x_H2O',
                                                'x_CO2',
                                                'x_O2',
                                                'x_MX',
                                                'x_{}'.format(solvent_name),
                                                'x_{}'.format(supporting_electrolyte_name)
                                                ]] = [df_electrolyzer_streams_mol_s['Product liquid outlet']/df_streams.loc[ 'L/L separation inlet', 'Molar flow rate (mol/s)'], 
                                                      0, 
                                                      df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Liquid drier inlet', 'x_H2O']/df_streams.loc[ 'L/L separation inlet', 'Molar flow rate (mol/s)'], 
                                                      0, 
                                                      0, 
                                                      0,
                                                      df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Liquid drier inlet', 'x_{}'.format(solvent_name)]/df_streams.loc[ 'L/L separation inlet', 'Molar flow rate (mol/s)'], 
                                                      df_streams.loc[ 'Liquid drier inlet', 'Molar flow rate (mol/s)']*df_streams.loc[ 'Liquid drier inlet', 'x_{}'.format(supporting_electrolyte_name)]/df_streams.loc[ 'L/L separation inlet', 'Molar flow rate (mol/s)'], 
                                                      ]

    ### Anode liquid outlets  
    df_streams.loc[ 'Anode outlet', 'Molar flow rate (mol/s)'] = df_electrolyzer_streams_mol_s['O2 outlet'] + df_streams.loc[ 'Anode inlet', 'Molar flow rate (mol/s)'] - df_streams.loc[ 'Cathode gas outlet', 'Molar flow rate (mol/s)']*df_streams.loc['Cathode gas outlet', 'x_H2O'] - df_streams.loc[ 'Cathode liquid outlet', 'Molar flow rate (mol/s)']*df_streams.loc['Cathode liquid outlet', 'x_H2O'] - df_electrolyzer_streams_mol_s['H2 outlet']
    if df_products.loc[product_name, 'Phase'] == 'liquid':
       df_streams.loc[ 'Anode outlet', 'Molar flow rate (mol/s)'] -= df_electrolyzer_streams_mol_s['Product liquid outlet'] # According to H2 balance, 1 mol H2 is also in 1 mol (COOH)2 or HCOOH product

    df_streams.loc[ 'Anode outlet', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            df_electrolyzer_streams_mol_s['CO2 anode outlet']/df_streams.loc[ 'Anode outlet', 'Molar flow rate (mol/s)'],
                                                            df_electrolyzer_streams_mol_s['O2 outlet']/df_streams.loc[ 'Anode outlet', 'Molar flow rate (mol/s)'],
                                                            df_streams.loc[ 'Anode inlet', 'Molar flow rate (mol/s)']*df_streams.loc['Anode inlet', 'x_MX']/df_streams.loc[ 'Anode outlet', 'Molar flow rate (mol/s)'],   
                                                            0, 
                                                            0                             
                                                           ]
    df_streams.loc['Anode outlet', 'x_H2O'] = 1 - sum([df_streams.loc['Anode outlet', 'x_{}'.format(product_name)], 
                                                      df_streams.loc['Anode outlet', 'x_H2'],
                                                      df_streams.loc['Anode outlet', 'x_CO2'],
                                                      df_streams.loc['Anode outlet', 'x_O2'],
                                                      df_streams.loc['Anode outlet', 'x_MX'],
                                                      df_streams.loc['Anode outlet', 'x_{}'.format(solvent_name)],
                                                      df_streams.loc['Anode outlet', 'x_{}'.format(supporting_electrolyte_name)],
                                                      ])
                                                                 
    # Update anode water recyle 
    df_streams.loc[ 'Anolyte recycle', 'Molar flow rate (mol/s)'] = df_streams.loc['Anode outlet', 'Molar flow rate (mol/s)'] - df_electrolyzer_streams_mol_s['Anode gas outlet']
    df_streams.loc[ 'Anolyte recycle', ['x_{}'.format(product_name),
                                                      'x_H2',
                                                      'x_CO2',
                                                      'x_O2',
                                                      'x_MX',
                                                      'x_{}'.format(solvent_name),
                                                      'x_{}'.format(supporting_electrolyte_name)
                                                      ]] = [0,
                                                            0,
                                                            0,
                                                            0,
                                                            df_streams.loc[ 'Anode inlet', 'Molar flow rate (mol/s)']*df_streams.loc['Anode inlet', 'x_MX']/df_streams.loc[ 'Anolyte recycle', 'Molar flow rate (mol/s)'],                                
                                                            0, 
                                                            0]
    df_streams.loc['Anolyte recycle', 'x_H2O'] = 1 - df_streams.loc['Anolyte recycle', 'x_MX']

    # Calculate mass flow rates
    df_streams['Mass flow rate (kg/day)'] = df_streams['Molar flow rate (mol/s)'] * 24 * 60 * 60 / 1000 * (df_streams['x_{}'.format(product_name)]*MW_product + df_streams['x_H2']*MW_H2 + df_streams['x_H2O']*MW_H2O + df_streams['x_CO2']*MW_CO2 + df_streams['x_O2']*MW_O2 + df_streams['x_MX']*MW_MX  + df_streams['x_{}'.format(solvent_name)]*MW_solvent + df_streams['x_{}'.format(supporting_electrolyte_name)]*MW_supporting)

    # Update stream phases
    df_streams['Phase'] = 'Vapor'
    df_streams.loc[['Anode inlet', 'Anode outlet', 'Anolyte recycle', 'Fresh solvent feed',
                    'Fresh water feed', 'Total DI feed', 'Cathode gas water recycle',
                    'Cathode liquid inlet', 'Cathode liquid outlet', 'Liquid drier inlet', 'Catholyte water recycle',
                    'Catholyte recycle'], 'Phase'] = 'Liquid'

    # Update stream VFRs
    df_streams.loc[df_streams['Phase'] == 'Vapor', 'Pressure (Pa)'] = P
    df_streams.loc[df_streams['Phase'] == 'Vapor', 'Temperature (∘C)'] = T_streams - K_to_C
    df_streams.loc[df_streams['Phase'] == 'Liquid', 'Temperature (∘C)'] = T_streams - K_to_C
    df_streams.loc[['PSA' in i for i in df_streams.index], 'Temperature (∘C)'] = T_sep - K_to_C
    df_streams.loc[df_streams['Phase'] == 'Vapor', 'Volumetric flow rates (m3/s)'] = (df_streams.loc[df_streams['Phase'] == 'Vapor', 'Molar flow rate (mol/s)'] * R * (df_streams.loc[df_streams['Phase'] == 'Vapor', 'Temperature (∘C)'] + K_to_C)) / df_streams.loc[df_streams['Phase'] == 'Vapor', 'Pressure (Pa)']
    df_streams.loc[['Anode inlet', 'Anode outlet', 'Anolyte recycle', 'Catholyte water recycle',
                    'Fresh water feed', 'Total DI feed', 'Cathode gas water recycle',], 'Volumetric flow rates (m3/s)'] = df_streams['Mass flow rate (kg/day)'] / water_density_kg_m3 / (24 * 60 * 60) 
    df_streams.loc[['Cathode liquid inlet', 'Cathode liquid outlet', 'Fresh solvent feed', 'Liquid drier inlet', 
                    'Catholyte recycle'], 'Volumetric flow rates (m3/s)'] = df_streams['Mass flow rate (kg/day)'] / electrolyte_density_kg_m3 / (24 * 60 * 60)
    if df_products.loc[product_name, 'Phase'] == 'liquid':
          df_streams.loc[['L/L separation inlet', 'Product outlet'], 'Phase'] = 'Liquid'         
          df_streams.loc[['L/L separation inlet', 'Product outlet'], 'Volumetric flow rates (m3/s)'] = df_streams['Mass flow rate (kg/day)'] / electrolyte_density_kg_m3 / (24 * 60 * 60)

    return df_streams

# %% [markdown]
# ## 2. Energies

# %% [markdown]
# ### 2.1 Separation energies

# %% [markdown]
# ### 2.2 Energy table

# %%
@st.cache_data(ttl = "1h")
def nonaq_energy_table(product_name,
                 df_products,
                 df_potentials,
                 df_streams, 
                 PSA_second_law_efficiency,
                 LL_second_law_efficiency,
                 T_sep,
                 electricity_cost_USD_kWh,
                 heat_cost_USD_kWh,
                 electricity_emissions_kgCO2_kWh,
                 heat_emissions_kgCO2_kWh,
                 kJ_per_kWh,
                 R
                ):
    
    """
    Create energy table for given product and electrolyzer stream compositions

    Arguments: product name of choice, product data dataframe with [product names, molecular weights (g/mol), LHVs (kJ/kg prod),
    dataframe with energy of electrolyzer (kJ/kg product), dataframe of stream table, second-law efficiency (), 
    separations temperature (K), electricity cost ($/kWh), heat cost ($/kWh), electricity emissions or carbon intensity (kg CO2/ kWh), 
    heat emissions or carbon intensity (kg CO2/kWh), conversion of kJ/kWh, gas constant (J/mol.K)
    
    Returns: dataframe for energy table
    """
    
    ## Extract data on a specific product
    MW_product = df_products.loc[product_name, 'Molecular weight (g/mol)']
    prod_LHV_kJ_kg = df_products.loc[product_name, 'LHV (kJ/kg product)']
    # prod_E0_V = df_products.loc[product_name, 'Standard potential (V vs RHE)']

    ## Pump for compressed air in dryer
    drier_energy_kW = ( (0.2 * df_streams.loc['Liquid drier inlet', 'Volumetric flow rates (m3/s)']) * ((6.894e5 - 1.01E+05)) / 0.7 ) / 1000 # Pump is 70% efficient, purge flow 30% of nameplate flow at 100 psig (689,476 Pa)

    ## CO2-product or CO2-H2 separation
    CO2_prod_sep_kJ_molmix = work_of_sep_kJ(df_streams.loc['Cathode PSA1 inlet', 'x_CO2'], PSA_second_law_efficiency, T_sep, R ) 
    CO2_prod_sep_kJ_molprod = CO2_prod_sep_kJ_molmix*df_streams.loc['Cathode PSA1 inlet', 'Molar flow rate (mol/s)']/df_streams.loc['Product outlet', 'Molar flow rate (mol/s)']
    CO2_prod_sep_kJ_kgprod = CO2_prod_sep_kJ_molprod/(MW_product/1000)

    ## H2-product separation    
    if df_products.loc[product_name, 'Phase'] == 'gas':
        prod_H2_sep_kJ_molmix = work_of_sep_kJ(df_streams.loc['Cathode PSA1 outlet', 'x_H2'], PSA_second_law_efficiency, T_sep, R)
        prod_H2_sep_kJ_molprod = prod_H2_sep_kJ_molmix*df_streams.loc['Cathode PSA1 outlet', 'Molar flow rate (mol/s)']/df_streams.loc['Product outlet', 'Molar flow rate (mol/s)']
        prod_H2_sep_kJ_kgprod = prod_H2_sep_kJ_molprod/(MW_product/1000)
        
    ## Liquid product separation    
    if df_products.loc[product_name, 'Phase'] == 'liquid':
        prod_solvent_sep_kJ_molmix = work_of_sep_kJ(df_streams.loc['L/L separation inlet', 'x_{}'.format(product_name)], LL_second_law_efficiency, T_sep, R)
        prod_solvent_sep_kJ_molmix = prod_solvent_sep_kJ_molmix*df_streams.loc['L/L separation inlet', 'Molar flow rate (mol/s)']/df_streams.loc['Product outlet', 'Molar flow rate (mol/s)']
        prod_solvent_sep_kJ_kgprod = prod_solvent_sep_kJ_molmix/(MW_product/1000)
    
    # ## CO2-O2 separation
    # CO2_O2_sep_kJ_molmix = work_of_sep_kJ(df_streams.loc['Anode PSA inlet', 'x_O2'], PSA_second_law_efficiency, T_sep, R)
    # CO2_O2_sep_kJ_molprod = CO2_O2_sep_kJ_molmix*df_streams.loc['Anode PSA inlet', 'Molar flow rate (mol/s)']/df_streams.loc['Product outlet', 'Molar flow rate (mol/s)']
    # CO2_O2_sep_kJ_kgprod = CO2_O2_sep_kJ_molprod/(MW_product/1000)
    
    ## Create dictionary
    dict_energy = {
    'Stage' : [ 'Electrolyte makeup', 'Electrolyte makeup', 'CO2 electrolysis', # 'Compression', 'Carbon capture', 
              'CO2 electrolysis',  'CO2 electrolysis','CO2 electrolysis',  'CO2 electrolysis','CO2 electrolysis',
             # 'Gas separations', 'Gas separations', 'Gas separations',
               ], 
    'Unit': [  'Deionization', 'Cell heating', 'Dessicant regeneration', #'Other', 'Carbon capture and transportation',
             'Cathode equilibrium potential', 'Anode equilibrium potential', 'Cathodic overpotential' , 'Anodic overpotential', 'Ohmic loss',
            #'Cathode PSA - CO$_2$/products','Cathode PSA - {}/H$_2$'.format(product_name), 'Anode PSA - CO$_2$/O$_2$',
             ],
                 }
    
    df_energy = pd.DataFrame(dict_energy)
    
    ## Create blank columns
    df_energy.set_index('Unit', inplace = True)
    df_energy.index.name = 'Energy'
    df_energy['Description'] = ''
    df_energy['Energy (kJ/kg {})'.format(product_name)] = np.NaN
    df_energy['Cost ($/kg {})'.format(product_name)] = np.NaN
    df_energy['Emissions (kg CO2/kg {})'.format(product_name)] = np.NaN
    
    ## Assign utility types
    df_energy.loc[['Deionization', 'Dessicant regeneration', 'Cathode equilibrium potential', 'Anode equilibrium potential', 
                   'Cathodic overpotential' , 'Anodic overpotential', 'Ohmic loss',
                  # 'Cathode PSA - CO$_2$/products','Cathode PSA - {}/H$_2$'.format(product_name), 
                   ],
                  'Description'] = 'Electricity'
    # df_energy.loc[['Cell heating',
    #                  ], 'Description'] = 'Heat'
    ## Currently, we assume no excess heat is required
    
    ## Electrolyzer energy
    df_energy.loc['Cathode equilibrium potential', 'Energy (kJ/kg {})'.format(product_name)] = df_potentials.loc['Cathode equilibrium energy per kg', 'Value']
    df_energy.loc['Anode equilibrium potential', 'Energy (kJ/kg {})'.format(product_name)] = df_potentials.loc['Anode equilibrium energy per kg', 'Value']
    df_energy.loc['Cathodic overpotential', 'Energy (kJ/kg {})'.format(product_name)] = df_potentials.loc['Cathodic overpotential energy per kg', 'Value']
    df_energy.loc['Anodic overpotential', 'Energy (kJ/kg {})'.format(product_name)] = df_potentials.loc['Anodic overpotential energy per kg', 'Value']
    df_energy.loc['Ohmic loss', 'Energy (kJ/kg {})'.format(product_name)] = df_potentials.loc['Ohmic loss energy per kg', 'Value']
    
    # df_energy.loc['Cell heating', 'Energy (kJ/kg {})'.format(product_name)] = 0 # Assume heat integration for cell heating
    
    ## Separations energy
    if df_products.loc[product_name, 'Phase'] == 'gas':
        df_energy.loc['Cathode PSA - CO$_2$/products', 'Energy (kJ/kg {})'.format(product_name)] = CO2_prod_sep_kJ_kgprod
        df_energy.loc['Cathode PSA - {}/H$_2$'.format(product_name), 'Energy (kJ/kg {})'.format(product_name, product_name)] = prod_H2_sep_kJ_kgprod
        df_energy.loc[['Cathode PSA - CO$_2$/products','Cathode PSA - {}/H$_2$'.format(product_name), 
                   ],
                  'Description'] = 'Electricity'
    else:
        df_energy.loc['Cathode PSA - CO$_2$/H$_2$', 'Energy (kJ/kg {})'.format(product_name)] = CO2_prod_sep_kJ_kgprod
        df_energy.loc['Liquid/liquid separation - {}/Catholyte'.format(product_name), 'Energy (kJ/kg {})'.format(product_name)] = prod_solvent_sep_kJ_kgprod
        df_energy.loc[['Cathode PSA - CO$_2$/H$_2$', 'Liquid/liquid separation - {}/Catholyte'.format(product_name)],
                  'Description'] = 'Electricity'
        
    ## Drier energy
    df_energy.loc['Dessicant regeneration', 'Energy (kJ/kg {})'.format(product_name)] = drier_energy_kW / df_streams.loc['Product outlet', 'Molar flow rate (mol/s)'] / (MW_product/1000)
        
    df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)] = abs(df_energy.loc[:, 'Energy (kJ/kg {})'.format(product_name)]).sum(axis=0)
    df_energy.loc['Total', ['Stage', 'Description']] = ''

    # Store overall cell potential after taking totals, so that it is not double counted
    df_energy.loc['Cell potential', 'Energy (kJ/kg {})'.format(product_name)] = df_potentials.loc['Electrolyzer energy per kg', 'Value']
    df_energy.loc['Cell potential', ['Stage', 'Description']] = ''

    # Account for cases where the cell potential is "overwritten", i.e. a cell potential is specified but no equilibrium potentials/ ohmic resistances/ etc
    if np.isnan(df_energy.loc['Cathode equilibrium potential', 'Energy (kJ/kg {})'.format(product_name)]) and ~np.isnan(df_energy.loc['Cell potential', 'Energy (kJ/kg {})'.format(product_name)]): # if cell voltage is overridden directly
        # Recalculate the total, this time including the cell potential in the total
        df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)] = abs(df_energy.loc[:, 'Energy (kJ/kg {})'.format(product_name)]).sum(axis=0)
        # Actually consider the cell potential as an electric outlay
        df_energy.loc['Cell potential', 'Description'] = 'Electricity'
    
    # Calculate emissions
    df_energy.loc[df_energy['Description'] == 'Electricity', 
                  'Emissions (kg CO2/kg {})'.format(product_name)] = np.abs(df_energy.loc[df_energy['Description'] == 'Electricity']['Energy (kJ/kg {})'.format(product_name)]) / kJ_per_kWh * electricity_emissions_kgCO2_kWh
    df_energy.loc[df_energy['Description'] == 'Heat', 'Emissions (kg CO2/kg {})'.format(product_name)] = np.abs(df_energy.loc[df_energy['Description'] == 'Heat']['Energy (kJ/kg {})'.format(product_name)]) / kJ_per_kWh * heat_emissions_kgCO2_kWh
        
    # Calculate costs
    df_energy.loc[df_energy['Description'] == 'Electricity', 'Cost ($/kg {})'.format(product_name)] = np.abs(df_energy.loc[df_energy['Description'] == 'Electricity','Energy (kJ/kg {})'.format(product_name)]) / kJ_per_kWh * electricity_cost_USD_kWh
    df_energy.loc[df_energy['Description'] == 'Heat', 'Cost ($/kg {})'.format(product_name)] = np.abs(df_energy.loc[df_energy['Description'] == 'Heat','Energy (kJ/kg {})'.format(product_name)]) / kJ_per_kWh * heat_cost_USD_kWh
    
    # Calculate efficiency
    if not np.isnan(df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)]):
        df_energy.loc['Efficiency vs LHV', 'Energy (kJ/kg {})'.format(product_name)] = prod_LHV_kJ_kg  / abs(df_energy.loc['Total', 'Energy (kJ/kg {})'.format(product_name)]) # LHV / process energy
    else:
        df_energy.loc['Efficiency vs LHV', 'Energy (kJ/kg {})'.format(product_name)] = np.NaN
        
    df_energy.loc['Efficiency vs LHV', ['Stage', 'Description']] = ''

    df_energy.loc['Total', 'Cost ($/kg {})'.format(product_name)] = abs(df_energy.loc[:, 'Cost ($/kg {})'.format(product_name)]).sum(axis=0)
    df_energy.loc['Total', 'Emissions (kg CO2/kg {})'.format(product_name)] = abs(df_energy.loc[:, 'Emissions (kg CO2/kg {})'.format(product_name)]).sum(axis=0)
   
    return df_energy


