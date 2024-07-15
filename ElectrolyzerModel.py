# %% [markdown]
# # Electrolyzer model: notes
# Date updated: 2024/07/01 \
# Update notes: Allow manual FE specification \
# Contact: Shashwati da Cunha, [shashwati.dacunha@austin.utexas.edu](mailto:shashwati.dacunha@austin.utexas.edu)
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

# %% [markdown]
# ## 1. Polarization curve model

# %% [markdown]
# ### 1.1 Conversions for flow rates and basis

# %%
### Other units
@st.cache_data(ttl = "1d")
def kg_day_to_kg_s(rate_kg_day):
    """
    Convert mass flow rate in kg/day to kg/s
    Arguments: mass flow rate (kg/day)
    Returns: mass flow rate (kg/s)
    """
    
    rate_kg_s = rate_kg_day/(24*60*60)
    
    return rate_kg_s

@st.cache_data(ttl = "1d")
def kg_s_to_mol_s(rate_kg_s, MW):
    """
    Convert mass flow rate in kg/s to mole flow rate in mol/s
    Arguments: mass flow rate (kg/s), molecular weight (g/mol)
    Returns: molar flow rate (mol/s)
    """
    
    rate_mol_s = rate_kg_s/(MW/1000)
    
    return rate_mol_s

@st.cache_data(ttl = "1d")
def mol_s_to_sccm(rate_mol_s, R, P = 101325, T = 298.15):
    """
    Convert molar flow rate in mol/s to volumetric flow rate in standard cubic cm per minute
    Arguments: molar flow rate (mol/s), gas constant , pressure for standard conditions, temperature for standard conditions. 
                Units should match (e.g. J/mol K, Pa, K)
    Returns: volumetric flow rate (standard cubic centimers per minute), where standard conditions are defined by P and T inputs 
            to this function
    """    
    
    rate_sccm = rate_mol_s/60 * (R * T / P) * 100**3
    
    return rate_sccm

@st.cache_data(ttl = "1d")
def mol_s_to_mA(rate_mol_s, n, F):
    """
    Convert product mole flow rate in mol/s to current in mA
    Arguments: molar flow rate of product (mol/s), number of transferred electrons (mol e-/ mol product), Faraday's constant (C/mol e-)
    Returns: current density (mA)
    """    
    
    i_mA = rate_mol_s * n * F *1000
    
    return i_mA

# %% [markdown]
# ### 1.2 Electrolyzer area

# %%
@st.cache_data(ttl = "1d")
def electrolyzer_area(i_total_mA, j_total_mA_cm2):
    """
    Use maximum allowed current density and desired current to compute total area
    Arguments: total current (mA), total current density (mA/cm2)
    Returns: active area (m2)
    """    
    
    area_cm2 = i_total_mA/j_total_mA_cm2 # cm2. Area = i/j
    area_m2 = area_cm2/1e4
    
    return area_m2

# %% [markdown]
# ### 1.3 Currents

# %%
@st.cache_data(ttl = "1d")
def currents(    
    product_rate_mol_s,
    FE_product,
    n_product, 
    F
):
    """
    Get all the currents based on product flow rate and selectivities
    Arguments: product molar flow rate (mol/s), FE to product (), n (mol e-/ mol product), Faraday's constant (C/mol e-)
    Outputs: product current (mA), total current (mA), HER current (mA), OER current (mA)
    """    
    
    i_product_mA = mol_s_to_mA(product_rate_mol_s, n_product, F)
    # TODO: make this into a df and assign a product-by-product current breakddown

    # From Moore Hahn Joule 2023 code:
    #     eta = Single_Pass_Conversion_CO2 * (1 + n_CO2R*Fraction_Charge_Carried_by_CO32/((1-FE_H2_0)*s_CO2R*Charge_Carbonate_Ion))    #Single Pass CO2 Consumption
    #     i_CO2R = -i_CO2R_0 * eta/np.log(1-eta)                  #CO2R Partial Current Density, A/m2

    i_total_mA = i_product_mA/FE_product # mA
    
    i_H2_mA = i_total_mA - i_product_mA # mA. Assumes FE goes to only H2 and product
    i_O2_mA = i_total_mA # mA. Assumes anodic reaction is only O2 evolution

    return i_product_mA, i_total_mA, i_H2_mA, i_O2_mA
    

# %%
@st.cache_data(ttl = "1d")
def voltage_to_energy(E_V, i_total_mA, product_rate_kg_s, product_rate_mol_s):
    """
    Compute power and energy from P = i.V
    Arguments: voltage (V), total current (mA), resulting product mass flow rate (kg/s), corresponding product mole flow rate (mol/s)
    Returns: power used to generate product (kW), energy required per unit product (kJ/kg product), 
    energy required per mole product (kJ/mol product)
    """
    power_kW = (E_V * i_total_mA/1000)/1000 # P = IV
    energy_kJ_kgproduct = power_kW/product_rate_kg_s 
    energy_kJ_molproduct = power_kW/product_rate_mol_s 
    
    return power_kW, energy_kJ_kgproduct, energy_kJ_molproduct

# %% [markdown]
# ### 1.4 Voltage

# %%
@st.cache_data(ttl = "1d")
def cell_voltage(
    product_name,    
    product_rate_kg_day,
    FE_product,
    FE_product_specified,
    FE_CO2R_0,
    model_FE,
    j_total_mA_cm2,
    R_ohmcm2,
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
    cat_Tafel_slope = df_products.loc[product_name, 'Tafel slope (mV/dec)']
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
    cat_E_eqm = df_products.loc[product_name, 'Standard potential (V vs RHE)']    
    
    ### Calculate cell potential
    
    if overridden_vbl == 'Cell voltage':
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
            if overridden_vbl != 'Cathodic overpotential':
                BV_eta_cat_V = cat_eta_ref + cat_Tafel_slope/1000*np.log10(abs(j_product_mA_cm2/cat_j_ref)) 
                # V. B-V overpotential = Ref overpotential + Tafel slope*log_10(product current/ reference current); Tafel slope in mV/dec 
            else:
                BV_eta_cat_V = overridden_value

            if overridden_vbl != 'Anodic overpotential':
                BV_eta_an_V = an_eta_ref + an_Tafel_slope/1000*np.log10(abs(j_O2_mA_cm2/an_j_ref)) 
                # V. B-V overpotential = Ref overpotential + Tafel slope*log_10(product current/ reference current); Tafel slope in mV/dec
            else:  
                BV_eta_an_V = overridden_value
        
        # Equilibrium cell voltage
        # cell_E_eqm_V = cat_E_eqm - an_E_eqm # full standard cell voltage

        # Ohmic loss
        ohmic_E_V = -i_total_mA/1000 * R_ohmcm2/area_cm2 # V = IR. Assumes MEA => membrane resistance only
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
        "Specific ohmic resistance": [R_ohmcm2, 'ohm.cm2'],
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
# ## 2. Selectivity model

# %% [markdown]
# ### 2.1 Equation for Hawks, Baker (ACS Energy Lett. 2021) model

# %%
@st.cache_data(ttl = "1d")
def eqn_known_SPC_jtotal(FE_product,
            j_total,
            FE_CO2R_0,
            product_name,
            SPC,
            df_products,
            crossover_ratio,
            ):

    """
    Equation = 0 to be solved to compute FE_CO2R as a function of single-pass conversion

    Arguments: FE_CO2R (), total current (constant over range of SPCs) (mA/cm2), 
    product name of choice, product data as a dataframe containing [product name, electron transfers for product (mol e-/ mol product), 
    stoichiometry as mol CO2/ mol product], crossover ratio (mol CO2 crossed/ mol e-).

    Returns: LHS - RHS = ratio (iCO2R/iCO2R0)+ xi/ln(1-xi), xi = fraction of CO2 consumed by all rxns, iCO2R0 is the CO2R 
    current at limiting case of SPC -> 0

    Assumes no product distribution beyond a single CO2R product and H2, constant electrolyzer area, all assumptions of Hawks et al, 
    and that FE_CO2R,0 is measured at the same total current density as the FE we are trying to calculate 

    """

    # Extract data
    n_product = df_products.loc[product_name, 'n (mol e-/ mol product)']
    z_product = df_products.loc[product_name, 'z (mol CO2/ mol product)']

    # LHS = 0 is the equation
    frac_CO2_consumed = SPC*(1 + (crossover_ratio*n_product/(z_product*FE_product)))

    LHS = (j_total*FE_product)/(j_total * FE_CO2R_0) + frac_CO2_consumed/np.log(1-frac_CO2_consumed)
    
    return LHS

# %% [markdown]
# ### 2.2 Mass balance check and all model execution

# %%
@st.cache_data(ttl = "1d")
def SPC_check(FE_product_specified,
              exponent, 
              scaling,
              SPC,
              j_total,
              FE_CO2R_0,
              product_name,
              model_FE,
              df_products,
              crossover_ratio
              ):

    """
    Check FE and SPC to meet carbon mass balance; model their tradeoff if desired

    Arguments: specified selectivity if not using any tradeoff model (not typical) (), 
    exponent () and scaling () if using curve fit to Kas, Smith (ACS Sust. Chem. Eng. 2021),
    single-pass conversion of CO2 to product (), total current density (mA/cm2),
    FE_CO2R in the limit of 0 single-pass conversion,
    product name of choice, choice of model for selectivity as a function of SPC,
    product data as a dataframe containing [product name, electron transfers for product (mol e-/ mol product), 
    stoichiometry as mol CO2/ mol product], 
    crossover ratio (mol CO2 crossed/ mol e-)

    Returns: FE_CO2R (modeled or assumed, depending on choice of model_FE), SPC

    This function is called directly from the integrated model to adjust inputs into other functions, not indirectly by the electrolyzer model
    """

    ### Extract data
    n_product = df_products.loc[product_name, 'n (mol e-/ mol product)']
    z_product = df_products.loc[product_name, 'z (mol CO2/ mol product)']
    
    ### Check validity of carbon mass balance with crossover
    # Regardless of whether the SPC-FE tradeoff is being modeled, mass balance must be observed
    
    # First check that the given SPC is valid
    max_SPC = z_product*FE_CO2R_0 / (z_product*FE_CO2R_0 + crossover_ratio*n_product)

    if SPC > max_SPC:
        print('Specified SPC {} is impossibly high given the crossover! Instead, using {}; max SPC is {}'.format(SPC, np.NaN, max_SPC) )
        SPC = np.NaN
        FE_product = np.NaN
        
    ### Run SPC-FE tradeoff model 
    # If at this stage, the SPC is a number, then proceed to model FE. Otherwise both SPC and FE were reset to NaN above
    if not np.isnan(SPC):         
        if model_FE == 'Hawks': # if we want to override the given FE
            root, infodict, flag_converged, message = optimize.fsolve(func = eqn_known_SPC_jtotal,  
                                        x0 = FE_CO2R_0, # x1 = min_FE + 1e-3,
                                              # bracket = [(min_FE + 1e-5), (FE_CO2R_0 - 1e-5)],
                                        args = (j_total,
                                                FE_CO2R_0,
                                                product_name,
                                                SPC,
                                                df_products,
                                                crossover_ratio,
                                                ),
                                        full_output = True,
                                        xtol = 1e-50
                                         )
            if flag_converged == 1:
                FE_product = root[0]
                # print(root, infodict, flag_converged, message)
            elif infodict['fvec'] <= 1e-5: # function evaluated at the output
                FE_product = root[0]
                # print(root, infodict, flag_converged, message)
            else:
                print(root, infodict, flag_converged, message)
                FE_product = np.NaN
                SPC = np.NaN
                print('Model failed')
            
            # except ValueError:
            #     result = optimize.root_scalar(f = eqn_known_SPC_jtotal,  
            #                                     x0 = min_FE + 1e-3, x1 = FE_CO2R_0,
            # #                                   bracket = [min_FE, FE_CO2R_0],
            #                                     args = (j_total,
            #                                             FE_CO2R_0,
            #                                             product_name,
            #                                             SPC,
            #                                             df_products,
            #                                             crossover_ratio,
            #                                             ),
            #                                     xtol = 1e-200
            #                                      )
            #     if result.converged:
            #         FE_product = result.root
            #     else:
            #         FE_product = np.NaN
            #         print('Model failed')
            
        elif model_FE == 'Kas':
            FE_product = FE_CO2R_0 - scaling*(SPC**exponent)         # FE_product = FE_CO2R_0 - 4.7306*(SPC**5.4936)
            print('Using Kas Smith 2021 tradeoff with exponent = {}, scaling = {}'.format(exponent, scaling))
        else:
            FE_product = FE_product_specified
            print('Using manually specified FE_product = {}'.format(FE_product))
                
        # Get minimum allowed FE
        # By mass balance, the minimum FE = Ṅ_CO2R/ (Ṅ_CO2R + Ṅ_carbonate) = (z*FE_CO2R*i/n_CO2R*F) / ((z*FE_CO2R*i/n_CO2R*F) + (c*i/F))
        min_FE = (n_product*crossover_ratio/z_product)*(SPC/(1-SPC)) 
        
        # Check that FE specified is high enough for mass balance
        if FE_product < min_FE:
            print('Resulting FE {} is impossibly low given the crossover (must be > {})! Instead, using {}'.format(FE_product, min_FE, np.NaN)) # FE_product_specified) )
            SPC = np.NaN
            FE_product = np.NaN # FE_product_specified
  
    print('SPC_check returned SPC = {}%; FE {}% \n'.format(SPC*100, FE_product*100))

    return FE_product, SPC

# %% [markdown]
# ## 3. Mass balance around electrolyzer

# %%
@st.cache_data(ttl = "1d")
def electrolyzer_SS_mass_balance(
    product_name,
    product_rate_kg_day,
    FE_product,
    SPC,
    df_potentials,
    crossover_ratio ,
    excess_water_ratio,
    cathode_outlet_humidity,
    j_total_mA_cm2 ,
    electrolyte_conc, 
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
    
    # Crossover
    CO2_an_outlet_mol_s = crossover_ratio * (i_total_mA / (1000*F))
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
    cat_liq_outlet_mol_s = product_liq_outlet_mol_s 
    # TODO: assign this a stream number and separations, also any liquid flow additional to product (water?) expected here

    ## Anode inlet
    water_makeup_mol_s = H2_outlet_mol_s   # 2 * (n_product/4) * product_rate_mol_s 
    # Assume that water is only consumed by HER
    # Water is consumed at the anode, but is equally regenerated by ion recombination;
    # More physically accurately, O2 is generated from hydroxide, not from water, resulting in a net zero water generation/consumption
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
         "Electrolyte concentration" : [electrolyte_conc, 'M'],
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
        'O2 outlet': O2_outlet_mol_s,
        'CO2 anode outlet': CO2_an_outlet_mol_s,
        'Anode gas outlet': an_gas_outlet_mol_s,
        'H2 outlet': H2_outlet_mol_s,
        'Product gas outlet': product_gas_outlet_mol_s,
        'Product liquid outlet': product_liq_outlet_mol_s ,
        'CO2 cathode outlet': CO2_cat_outlet_mol_s,
        'Cathode gas outlet': cat_gas_outlet_mol_s,
        'Cathode liquid outlet': cat_liq_outlet_mol_s
    }
         
    df_electrolyzer_streams_mol_s = pd.Series(dict_electrolyzer_streams_mol_s) # convert to dataframe
    
    if np.isnan(SPC):
        df_electrolyzer_streams_mol_s.loc[:] = np.NaN

    return df_outlet_assumptions, df_electrolyzer_streams_mol_s


