# %% [markdown]
# # Electrolyzer model
# Contact: Shashwati da Cunha, [shashwati.dc@utexas.edu](mailto:shashwati.dc@utexas.edu)
# 
# ### Instructions
# 1. Not designed for standalone run - this is only a collection of functions. Other notebooks call it.
# 
# ### Notes:
# This is broken into functions to allow modular running - can run different sections by themselves depending on what data is available. As a result of this structure, there is some redundancy where the same function will be run more than once by an "overall" call
# 

# %% [markdown]
# ## 0. Imports and setup

# %% [markdown]
# ### 0.1. Imports and styling

# %%
import streamlit as st
import pandas as pd
import numpy as np
from scipy import optimize
from ElectrolyzerModel import *

# %% [markdown]
# ## 1. Polarization curve model

# %% [markdown]
# ### 1.1 Conversions for flow rates and basis

# %%
### Other units

# ### 1.4 Voltage

# %%
@st.cache_data(ttl = "1h")
def nonaq_cell_voltage(
    product_name,    
    product_rate_kg_day,
    FE_product,
    FE_product_specified,
    FE_CO2R_0,
    model_FE,
    j_total_mA_cm2,
    R_membrane_ohmcm2,
    kappa_electrolyte_S_cm,
    electrolyte_thickness_cm,
    df_products, # product data - MW, n, B-V parameters
    an_E_eqm,
    an_eta_ref,
    an_Tafel_slope,
    an_j_ref,
    overridden_vbl,
    overridden_value,
    overridden_unit,
    override_optimization,
    F
):
    
    """
    Electrolyzer voltage model, using product of choice and production rate as a basis.

    Arguments: product name of choice, production basis (kg product/day), 
    Faradaic efficiency (selectivity)  to product, specified selectivity (rarely used), 
    which model was used to compute selectivity from SPC,
    total current density (mA/cm2), specific cell resistance (ohm.cm2), 
    product data as a dataframe containing [product name, molecular weight (g/mol), 
    electron transfers for product (mol e-/ mol product), stoichiometry (mol CO2/ mol product), equilibrium potential (V), 
    reference overpotential (V), Tafel slope (mV/dec), reference current density (mA/cm2) ],
    anode equilibrium potential (V), anode reference overpotential (V),
    anode Tafel slope (mV/dec), anode reference current density (mA/cm2),
    any inputs that were overwritten, the value of the overwritten input, 
    the units of the overwritten input, whether an optimization was used (bool), Faraday's constant (C/mol e-)

    Returns: dataframe of assumed inputs, and dataframe of generated potentials, currents and power 
    """
    
    ### Extract data on specific product
    MW_product = df_products.loc[product_name, 'Molecular weight (g/mol)']
    n_product = df_products.loc[product_name, 'n (mol e-/ mol product)']
    cat_eta_ref = df_products.loc[product_name, 'Reference overpotential (V)']
    if 'Cathodic Tafel slope' not in overridden_vbl:
        cat_Tafel_slope = df_products.loc[product_name, 'Tafel slope (mV/dec)']
    else:
        cat_Tafel_slope = overridden_value
    cat_j_ref = df_products.loc[product_name, 'Reference current density (mA/cm2)'] 
    
    ### Convert rates into currents
    product_rate_kg_s = kg_day_to_kg_s(product_rate_kg_day)
    product_rate_mol_s = kg_s_to_mol_s(product_rate_kg_s, MW = MW_product)
    # product_rate_sccm = mol_s_to_sccm(rate_mol_s = product_rate_mol_s, R = R)

    ### Calculate all currents
    i_product_mA, i_total_mA, i_H2_mA, i_O2_mA = currents(product_rate_mol_s, FE_product, n_product, F)
    
    area_m2 = electrolyzer_area(i_total_mA, j_total_mA_cm2)
    area_cm2 = area_m2 * 1e4
    
    j_product_mA_cm2 = i_product_mA/area_cm2 # mA/cm2
    j_O2_mA_cm2 = i_O2_mA/area_cm2 # mA/cm2
    j_H2_mA_cm2 = i_H2_mA/area_cm2 # mA/cm2
    
    # Import equilibrium potentials
    cat_E_eqm = df_products.loc[product_name, 'Standard potential, pH = 0 (V vs SHE)']    
    
    ### Calculate cell potential
    
    if 'Cell voltage' in overridden_vbl:
        # Directly assign cell voltage if overridden
        cell_E_V = overridden_value
        cat_E_eqm = np.NaN
        cat_E_V = np.NaN
        # cat_power_kW = np.NaN
        BV_eta_cat_V = np.NaN
        BV_eta_an_V = np.NaN
        an_E_eqm = np.NaN
        an_E_V = np.NaN
        # an_power_kW = np.NaN
        ohmic_E_V = np.NaN
        # ohmic_power_kW = np.NaN
        R_ohm_electrolyte = np.NaN
        R_ohm_membrane = np.NaN

    else:
        
        ### Handle invalid cases
        if np.isnan(FE_product):
            cell_E_V = np.NaN
            i_total_mA = np.NaN
            j_total_mA_cm2 = np.NaN
            BV_eta_an_V = np.NaN
            BV_eta_cat_V = np.NaN
            an_E_eqm = np.NaN
            cat_E_eqm = np.NaN
            cat_E_V = np.NaN
            
        else:
            # Butler-Volmer overpotentials
            if 'Cathodic overpotential' not in overridden_vbl:
                BV_eta_cat_V = cat_eta_ref + cat_Tafel_slope/1000*np.log10(abs(j_product_mA_cm2/cat_j_ref)) 
                # V. B-V overpotential = Ref overpotential + Tafel slope*log_10(product current/ reference current); Tafel slope in mV/dec 
            else:
                BV_eta_cat_V = overridden_value

            if 'Anodic overpotential' not in overridden_vbl:
                BV_eta_an_V = an_eta_ref + an_Tafel_slope/1000*np.log10(abs(j_O2_mA_cm2/an_j_ref)) 
                # V. B-V overpotential = Ref overpotential + Tafel slope*log_10(product current/ reference current); Tafel slope in mV/dec
            else:  
                BV_eta_an_V = overridden_value
        
        # Equilibrium cell voltage
        # cell_E_eqm_V = cat_E_eqm - an_E_eqm # full standard cell voltage

        # Ohmic loss
        R_ohm_membrane = R_membrane_ohmcm2/area_cm2
        R_ohm_electrolyte = (1/kappa_electrolyte_S_cm)*electrolyte_thickness_cm/area_cm2
        ohmic_E_V = -i_total_mA/1000 * (R_ohm_membrane + R_ohm_electrolyte) # V = IR. Assumes flow cell => membrane resistance only if flow field thickness is 0, otherwise add resistance of electrolyte
        # ohmic_power_kW = (ohmic_E_V * i_total_mA/1000)/1000 # P = IV
        
        # Cathode power
        cat_E_V = cat_E_eqm + BV_eta_cat_V
        # cat_power_kW = (cat_E_V * i_total_mA/1000)/1000 # P = IV
        # cat_energy_kJ_kgproduct = cat_power_kW/product_rate_kg_s 
        # cat_energy_kJ_molproduct = cat_power_kW/product_rate_mol_s 
        
        # Anode power
        an_E_V = an_E_eqm + BV_eta_an_V
        # an_power_kW = (an_E_V * i_total_mA/1000)/1000 # P = IV
        # an_energy_kJ_kgproduct = an_power_kW/product_rate_kg_s 
        # an_energy_kJ_molproduct = an_power_kW/product_rate_mol_s 

        # Cell voltage
        cell_E_V = cat_E_V - an_E_V + ohmic_E_V 
    
    ### Calculate cell energy
    cell_power_kW, cell_energy_kJ_kgproduct, cell_energy_kJ_molproduct = voltage_to_energy(cell_E_V, i_total_mA, 
                                                                                           product_rate_kg_s, product_rate_mol_s)
    __, cathode_eqm_energy_kJ_kgproduct, __ = voltage_to_energy(cat_E_eqm, i_total_mA, product_rate_kg_s, product_rate_mol_s)
    __, anode_eqm_energy_kJ_kgproduct, __ = voltage_to_energy(an_E_eqm, i_total_mA, product_rate_kg_s, product_rate_mol_s)
    __, cathode_eta_energy_kJ_kgproduct, __ = voltage_to_energy(BV_eta_cat_V, i_total_mA, product_rate_kg_s, product_rate_mol_s)
    __, anode_eta_energy_kJ_kgproduct, __ = voltage_to_energy(BV_eta_an_V, i_total_mA, product_rate_kg_s, product_rate_mol_s)
    __, ohmic_energy_kJ_kgproduct, __ = voltage_to_energy(ohmic_E_V, i_total_mA, product_rate_kg_s, product_rate_mol_s)
        
    ### Write results to dataframe
    dict_potentials = {
        'Cell potential': [cell_E_V, 'V'],
        'Area': [area_m2, 'm2'],
        'Cathode equilibrium potential' : [cat_E_eqm, 'V'],
        'Anode equilibrium potential' : [an_E_eqm, 'V'],
        'Cathodic overpotential': [BV_eta_cat_V , 'V'],
        'Anodic overpotential': [BV_eta_an_V, 'V'],
        'Ohmic loss': [ohmic_E_V, 'V'],
        "Membrane resistance": [R_ohm_membrane, 'ohm'],
        "Electrolyte resistance": [R_ohm_electrolyte, 'ohm'],
        'Cathode total potential': [cat_E_V, 'V'],
        'Anode total potential': [an_E_V, 'V'],
        
        'Current': [i_total_mA, 'mA'],
        'Current density (assumed)': [j_total_mA_cm2, 'mA/cm2'],
        '{} current'.format(product_name): [i_product_mA, 'mA'],
        '{} current density'.format(product_name): [j_product_mA_cm2, 'mA/cm2'],
        'O2 current': [i_O2_mA, 'mA'],
        'O2 current density': [j_O2_mA_cm2, 'mA/cm2'],
        'H2 current': [i_H2_mA, 'mA'],
        'H2 current density': [j_H2_mA_cm2, 'mA/cm2'],
        
        'Cell power': [cell_power_kW, 'kW'],
        'Cathode equilibrium energy per kg': [cathode_eqm_energy_kJ_kgproduct, 'kJ/kg {}'.format(product_name)],
        'Anode equilibrium energy per kg': [anode_eqm_energy_kJ_kgproduct, 'kJ/kg {}'.format(product_name)],
        'Cathodic overpotential energy per kg': [cathode_eta_energy_kJ_kgproduct, 'kJ/kg {}'.format(product_name)],
        'Anodic overpotential energy per kg': [anode_eta_energy_kJ_kgproduct, 'kJ/kg {}'.format(product_name)],
        'Ohmic loss energy per kg': [ohmic_energy_kJ_kgproduct, 'kJ/kg {}'.format(product_name)],
        'Electrolyzer energy per kg': [cell_energy_kJ_kgproduct, 'kJ/kg {}'.format(product_name)],
        'Electrolyzer energy per mol': [cell_energy_kJ_molproduct, 'kJ/mol {}'.format(product_name)],
        
        }    

    df_potentials = pd.DataFrame(dict_potentials).transpose() # convert to dataframe
#     df_potentials.reset_index(drop = False, inplace = True) # reset index to numbering
    df_potentials.columns = ['Value', 'Units'] # Extract product details from product dictionary
    df_potentials = df_potentials.astype({'Value':'float64', 'Units':'string'}) # Extract product details from product dictionary
    df_potentials.index.name = 'Potential variable'

    ### Write assumptions into dictionary and dataframe
    dict_electrolyzer_assumptions = {
        "Production rate" : [product_rate_kg_day, 'kg/day'],
        "Specified FE {}".format(product_name) : [FE_product_specified, ''],
        "FE {} at 0% SPC".format(product_name) : [FE_CO2R_0, ''],
        "Current density": [j_total_mA_cm2, 'mA/cm2'],
        "Membrane specific ohmic resistance": [R_membrane_ohmcm2, 'ohm.cm2'],
        "Conductivity of electrolyte": [kappa_electrolyte_S_cm, 'S/cm'],
        "Electrolyte thickness": [electrolyte_thickness_cm, 'cm'],
        'Modeled FE?': [{None: 0, 
                         'Hawks': 1, 
                         'Kas': 2}[model_FE], ''],
        'FE {}'.format(product_name): [FE_product, ''],
        '{} (overridden)'.format(overridden_vbl): [overridden_value, overridden_unit],
        'Run optimization?': [override_optimization, ''],
        }
        
    df_electrolyzer_assumptions = pd.DataFrame(dict_electrolyzer_assumptions).transpose() # convert to dataframe
    df_electrolyzer_assumptions.columns = ['Value', 'Units'] # Extract product details from product dictionary
    df_electrolyzer_assumptions = df_electrolyzer_assumptions.astype({'Value':'float64', 'Units':'string'}) # Extract product details from product dictionary
    df_electrolyzer_assumptions.index.name = 'Assumed variable - cell potential'        
    
    return df_electrolyzer_assumptions, df_potentials

# %% [markdown]
# ## 3. Mass balance around electrolyzer

# %%
@st.cache_data(ttl = "1h")
def nonaq_electrolyzer_SS_mass_balance(
    product_name,
    product_rate_kg_day,
    FE_product,
    SPC,
    df_potentials,
    crossover_ratio,
    excess_water_ratio,
    excess_solvent_ratio,
    cathode_outlet_humidity,
    j_total_mA_cm2 ,
    catholyte_conc_M, 
    anolyte_conc_M,
    df_products , # product data - MW, n, z
    carbon_capture_efficiency,
    MW_CO2, 
    F
):
    
    """
    Steady-state mass balance on electrolyzer streams and adjacent streams using product of choice and production rate as a basis.

    Arguments: product name of choice, production basis (kg product/day), Faradaic efficiency (selectivity)  to product (), 
    single-pass conversion (), dataframe from voltage model containing [], crossover ratio (mol CO2 crossed/ mol e-), 
    molar ratio of liquid water supplied to anode versus product flow rate (),
    total assumed current density (mA/cm2), product data as a dataframe containing [product name, molecular weight, 
    electron transfers for product (mol e-/ mol product), stoichiometry as mol CO2/ mol product],
    carbon capture efficiency (fraction of CO2 produced that gets captured), molar mass of CO2 (g/mol),
    Faraday's constant (C/mol e-)
    
    Returns: dataframe of assumptions used in stream calculations, dataframe of molar flow rates of inlet and outlet streams of electrolyzer
    """
    
    ### Extract data on specific product
    MW_product = df_products.loc[product_name, 'Molecular weight (g/mol)']
    z_product = df_products.loc[product_name, 'z (mol CO2/ mol product)']
    
    ### Convert rates
    product_rate_kg_s = kg_day_to_kg_s(product_rate_kg_day)
    product_rate_mol_s = kg_s_to_mol_s(product_rate_kg_s, MW = MW_product)
    # product_rate_sccm = mol_s_to_sccm(rate_mol_s = product_rate_mol_s, R = R)
        
    # Get current for crossover calculation
    i_H2_mA = df_potentials.loc['H2 current', 'Value']
    i_total_mA = df_potentials.loc['Current', 'Value']
     
    ### Cathode inlet
    CO2_fresh_mol_s = product_rate_mol_s * z_product # mol/s. Fresh feed - mass balance assumes all generated CO2 is outlet
    CO2_inlet_mol_s = product_rate_mol_s * z_product / SPC # mol/s. Inlet CO2 flow rate to electrolyzer 
  
    ## HER
    H2_outlet_mol_s = i_H2_mA / (2 * F * 1000) # mol/s
    
    ## Anode gas outlet
    O2_outlet_mol_s = i_total_mA / (4 * F * 1000) # OER compensates both HER and CO2R 

    ## Anode inlet
    water_an_inlet_mol_s = excess_water_ratio * O2_outlet_mol_s # assume 2500x the consumed water is fed - vast excess
    
    ## Solvent inlet
    solvent_inlet_mol_s = excess_solvent_ratio * product_rate_mol_s # assume 500x the fed CO2 is fed as solvent - this is not a reactant, do not need much of it

    # Crossover
    CO2_an_outlet_mol_s = crossover_ratio * (i_total_mA / (1000*F)) ## Assume that there is negligible CO2 transport through CEM - crossover_ratio is 0   
    an_gas_outlet_mol_s = O2_outlet_mol_s + CO2_an_outlet_mol_s
    
    ## Cathode gas and liquid outlets
    if df_products.loc[product_name, 'Phase'] == 'gas':
        product_gas_outlet_mol_s = product_rate_mol_s # mol/s
        product_liq_outlet_mol_s = 0
    elif df_products.loc[product_name, 'Phase'] == 'liquid':
        product_gas_outlet_mol_s = 0
        product_liq_outlet_mol_s = product_rate_mol_s # mol/s
    
    CO2_cat_outlet_mol_s = CO2_inlet_mol_s - product_rate_mol_s * z_product - CO2_an_outlet_mol_s # carbon mass balance    
    cat_gas_outlet_mol_s = (product_gas_outlet_mol_s + H2_outlet_mol_s + CO2_cat_outlet_mol_s)/(1-cathode_outlet_humidity) # 5% humid gas means total moles = (moles of everything else)/(1-mol fraction water)

    ## Anode inlet
    water_makeup_mol_s = H2_outlet_mol_s 
    if df_products.loc[product_name, 'Phase'] == 'liquid':
        water_makeup_mol_s += 0.5*CO2_fresh_mol_s   # 2 * (n_product/4) * product_rate_mol_s 
    # Assume that water is consumed by HER and CO2R. The reaction makes either oxalic acid, formic acid, or carbonic acid by proton migration through the membrane
    # However, carbonic acid locally regenerates CO2 and water in the presence of protons. Therefore, when CO is made, there is no net water consumption.
    # If formate and oxalate were made instead, they would also not consume water. There is a net consumption of water to make protons that are not reg
    # The only water deionized is the water condensed from the wet cathode outlet gas, defined by the cathode_outlet_humidity
    
    ### Calculate emissions due to inefficiency in carbon capture
    carbon_capture_loss_mol_s = (1-carbon_capture_efficiency) * (CO2_fresh_mol_s)/carbon_capture_efficiency 
    carbon_capture_loss_kgperkg = (carbon_capture_loss_mol_s *MW_CO2 )/ (product_rate_mol_s *MW_product)
    
    ## Store assumptions
    dict_outlet_assumptions = {
        "Product": [product_name, ''],
        "Production rate" : [product_rate_kg_day, 'kg/day'],
        "FE {}".format(product_name) : [FE_product, ''],
        "Single-pass conversion": [SPC, ''],
        "Current density": [j_total_mA_cm2, 'mA/cm2'],
        "Crossover ratio" : [crossover_ratio, 'mol CO2/ mol e-'],
        "Humidity of cathode gas outlet (molar)": [cathode_outlet_humidity, ''],
        "Excess ratio of water feed vs product rate (molar)": [excess_water_ratio, 'mol/s water/ mol/s {}'.format(product_name)],
        "Excess ratio of solvent feed vs CO2 feed rate (molar)": [excess_solvent_ratio, 'mol/s solvent/ mol/s CO2'],
        "Catholyte concentration" : [catholyte_conc_M, 'M'],
        "Anolyte concentration" : [anolyte_conc_M, 'M'],
        'Carbon capture efficiency': [carbon_capture_efficiency, ''],
        'Carbon capture loss': [carbon_capture_loss_kgperkg, 'kg CO2/ kg {}'.format(product_name)]
    }
         
    df_outlet_assumptions = pd.DataFrame(dict_outlet_assumptions).transpose() # convert to dataframe
    df_outlet_assumptions.columns = ['Value', 'Units'] # Extract product details from product dictionary
    df_outlet_assumptions.index.name = 'Assumed variable - mass balance'
        
    dict_electrolyzer_streams_mol_s = {
        'CO2 fresh': CO2_fresh_mol_s,
        'CO2 inlet': CO2_inlet_mol_s,
        'Water inlet': water_an_inlet_mol_s,
        'Water makeup': water_makeup_mol_s,
        'Solvent inlet': solvent_inlet_mol_s,
        'O2 outlet': O2_outlet_mol_s,
        'CO2 anode outlet': CO2_an_outlet_mol_s,
        'Anode gas outlet': an_gas_outlet_mol_s,
        'H2 outlet': H2_outlet_mol_s,
        'Product gas outlet': product_gas_outlet_mol_s,
        'Product liquid outlet': product_liq_outlet_mol_s ,
        'CO2 cathode outlet': CO2_cat_outlet_mol_s,
        'Cathode gas outlet': cat_gas_outlet_mol_s,
        'Liquid product outlet': product_liq_outlet_mol_s
    }
         
    df_electrolyzer_streams_mol_s = pd.Series(dict_electrolyzer_streams_mol_s) # convert to dataframe
    
    if np.isnan(SPC):
        df_electrolyzer_streams_mol_s.loc[:] = np.NaN

    return df_outlet_assumptions, df_electrolyzer_streams_mol_s


