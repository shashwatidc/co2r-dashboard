# %% [markdown]
# # Single model run
# Contact: Shashwati da Cunha, [shashwati.dc@utexas.edu](mailto:shashwati.dc@utexas.edu)
# 
# ### Instructions
# 1. Not designed for standalone run - this is only a collection of functions. Other notebooks call it.

import streamlit as st

# Duplicate functions wherever possible
from ElectrolyzerModel import *
from ProcessEconomics import *
from NonAqElectrolyzerModel import *
from NonAqDownstreamProcessModel import *
from NonAqProcessEconomics import *

# %%
## Check if SPC is reasonable given FE and crossover. Generate FE and SPC - if model_FE is true, FE will be modeled; either way, both will be checked for mass balance

@st.cache_data(ttl = "1h")
def single_run_nonaq(product_name,
                solvent_name,
                supporting_electrolyte_name,
                df_products,

                product_rate_kg_day,
                model_FE,
                FE_CO2R_0,
                FE_product_specified,
                j_total_mA_cm2,
                SPC,
                crossover_ratio,
                P,
                T_streams,

                R_membrane_ohmcm2,
                electrolyte_thickness_cm,
                
                an_E_eqm,
                an_eta_ref,
                an_Tafel_slope,
                an_j_ref,

                cathode_outlet_humidity,
                excess_water_ratio,   
                excess_solvent_ratio,
                catholyte_conc_M,  
                anolyte_conc_M,
                water_density_kg_m3,
                electrolyte_density_kg_m3,
                solvent_loss_fraction,

                LL_second_law_efficiency,
                PSA_second_law_efficiency,
                T_sep,         
                CO2_solubility_mol_mol,

                carbon_capture_efficiency,
                electricity_emissions_kgCO2_kWh,
                heat_emissions_kgCO2_kWh,

                electricity_cost_USD_kWh,
                heat_cost_USD_kWh,
                product_cost_USD_kgprod,
                H2_cost_USD_kgH2,
                water_cost_USD_kg,
                CO2_cost_USD_tCO2,
                electrolyzer_capex_USD_m2,  
                PSA_capex_USD_1000m3_hr,
                LL_capex_USD_1000mol_hr,     
                solvent_cost_USD_kg,
                electrolyte_cost_USD_kg,    

                lifetime_years,
                stack_lifetime_years,
                capacity_factor,

                battery_capex_USD_kWh,               
                battery_capacity,

                kappa_electrolyte_S_cm,
                viscosity_cP,  

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
                MW_solvent,
                MW_supporting,
                R, 
                F,
                
                K_to_C = 273.15,
                kJ_per_kWh = 3.60E+03,
    ):
    """
    Returns all outputs from a single run of the physics-informed technoeconomic model.
    """

    FE_product, SPC = SPC_check(FE_product_specified = FE_product_specified,
                                exponent = exponent,
                                scaling = scaling,
                                SPC = SPC,
                                j_total = j_total_mA_cm2,
                                FE_CO2R_0 = FE_CO2R_0,
                                product_name = product_name,
                                df_products = df_products,
                                crossover_ratio = crossover_ratio,
                                model_FE = model_FE,
                                )

    ## Generate electrolyzer model
    df_electrolyzer_assumptions, df_potentials = nonaq_cell_voltage(
        product_name = product_name,    
        product_rate_kg_day = product_rate_kg_day,
        FE_product = FE_product,
        FE_product_specified = FE_product_specified,
        FE_CO2R_0 = FE_CO2R_0,
        j_total_mA_cm2 = j_total_mA_cm2,
        R_membrane_ohmcm2 = R_membrane_ohmcm2,
        kappa_electrolyte_S_cm = kappa_electrolyte_S_cm,
        electrolyte_thickness_cm = electrolyte_thickness_cm,
        df_products = df_products, 
        an_E_eqm = an_E_eqm,  
        an_eta_ref = an_eta_ref,
        an_Tafel_slope = an_Tafel_slope,
        an_j_ref = an_j_ref,
        overridden_vbl = overridden_vbl,
        overridden_value = overridden_value,
        overridden_unit = overridden_unit,
        model_FE = model_FE,
        override_optimization =  override_optimization,
        F = F
        ) 

    ## Generate stream information for electrolyzer outlets
    df_outlet_assumptions, df_electrolyzer_streams_mol_s = nonaq_electrolyzer_SS_mass_balance(
        product_name = product_name,
        product_rate_kg_day = product_rate_kg_day,
        FE_product = FE_product,
        SPC = SPC,
        df_potentials = df_potentials, 
        crossover_ratio = crossover_ratio,
        excess_water_ratio = excess_water_ratio,
        excess_solvent_ratio = excess_solvent_ratio,
        catholyte_conc_M = catholyte_conc_M,
        anolyte_conc_M = anolyte_conc_M,
        cathode_outlet_humidity = cathode_outlet_humidity,
        j_total_mA_cm2 = j_total_mA_cm2,
        df_products = df_products,     
        carbon_capture_efficiency = carbon_capture_efficiency ,
        MW_CO2 = MW_CO2,
        F = F,
        ) 

    ## Generate stream table
    df_streams = nonaq_blank_stream_table(df_products = df_products,
                    product_name = product_name) # Generate blank stream table
    df_streams = nonaq_update_stream_table(product_name = product_name, 
                    solvent_name = solvent_name,
                    supporting_electrolyte_name = supporting_electrolyte_name,
                    df_products = df_products, 
                    df_streams = df_streams, 
                    df_electrolyzer_streams_mol_s = df_electrolyzer_streams_mol_s,
                    catholyte_conc_M = catholyte_conc_M,
                    anolyte_conc_M = anolyte_conc_M,
                    water_density_kg_m3 = water_density_kg_m3,
                    electrolyte_density_kg_m3 = electrolyte_density_kg_m3,
                    solvent_loss_fraction = solvent_loss_fraction,
                    cathode_outlet_humidity = cathode_outlet_humidity,
                    T_streams = T_streams,
                    T_sep = T_sep,
                    P = P,
                    MW_CO2 = MW_CO2,
                    MW_H2O = MW_H2O,
                    MW_O2 = MW_O2,
                    MW_MX = MW_MX,    
                    MW_solvent = MW_solvent, 
                    MW_supporting = MW_supporting,             
                    K_to_C = K_to_C,
                    R = R,
                    ) # Populate stream table

    # Format stream table
    df_streams_formatted = df_streams.copy().T
    df_streams_formatted.loc['Description'] = df_streams_formatted.columns
    df_streams_formatted.columns = df_streams_formatted.loc['Stream number']
    df_streams_formatted.drop(index = 'Stream number', inplace = True)
    df_streams_formatted.index.name = 'Parameter/Unit'

    ## Generate energy table
    df_energy = nonaq_energy_table(product_name = product_name,
            df_products = df_products,
            df_potentials = df_potentials,
            df_streams = df_streams, 
            PSA_second_law_efficiency = PSA_second_law_efficiency,
            LL_second_law_efficiency = LL_second_law_efficiency,
            T_sep = T_sep,         
            electricity_cost_USD_kWh = electricity_cost_USD_kWh,
            heat_cost_USD_kWh = heat_cost_USD_kWh,
            electricity_emissions_kgCO2_kWh = electricity_emissions_kgCO2_kWh,
            heat_emissions_kgCO2_kWh = heat_emissions_kgCO2_kWh,
            kJ_per_kWh = kJ_per_kWh,
            R = R,
            )

    df_utilities = utilities(df_energy = df_energy,
                              product_rate_kg_day = product_rate_kg_day,
                              capacity_factor = capacity_factor,
                              product_name = product_name)

    df_costing_assumptions = nonaq_costing_assumptions(product_name = product_name,
                                                solvent_name = solvent_name,
                                                supporting_electrolyte_name = supporting_electrolyte_name,
                                                product_cost_USD_kgprod = product_cost_USD_kgprod,
                                                H2_cost_USD_kgH2 = H2_cost_USD_kgH2,
                                                electricity_cost_USD_kWh = electricity_cost_USD_kWh,
                                                water_cost_USD_kg = water_cost_USD_kg,
                                                CO2_cost_USD_tCO2 = CO2_cost_USD_tCO2,    
                                                solvent_cost_USD_kg = solvent_cost_USD_kg,
                                                electrolyte_cost_USD_kg = electrolyte_cost_USD_kg,
                                                electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2,
                                                PSA_capex_USD_1000m3_hr = PSA_capex_USD_1000m3_hr,
                                                LL_capex_USD_1000mol_hr = LL_capex_USD_1000mol_hr,
                                                lifetime_years = lifetime_years,
                                                stack_lifetime_years = stack_lifetime_years,
                                                capacity_factor = capacity_factor,
                                                overridden_vbl = overridden_vbl)
    ## Generate capex
    df_capex_BM, df_capex_totals, C_TDC, C_alloc = nonaq_capex(area_m2 = df_potentials.loc['Area', 'Value'],
                                                    df_products = df_products,
                                                    product_name = product_name,
                                                    solvent_name = solvent_name,
                                                    supporting_electrolyte_name = supporting_electrolyte_name,
                                                    electrolyte_density_kg_m3 = electrolyte_density_kg_m3,
                                                    overridden_vbl = overridden_vbl,
                                                    CO2_solubility_mol_mol = CO2_solubility_mol_mol,
                                                    MW_CO2 = MW_CO2,
                                                    MW_supporting = MW_supporting,
                                                    catholyte_conc_M = catholyte_conc_M,
                                                    electricity_kJ_per_kg = df_utilities.loc['Electricity', 'Energy (kJ/kg {})'.format(product_name)],
                                                    df_streams = df_streams,
                                                    product_rate_kg_day = product_rate_kg_day,
                                                    battery_capex_USD_kWh = battery_capex_USD_kWh,        
                                                    electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2 ,
                                                    solvent_cost_USD_kg = solvent_cost_USD_kg,
                                                    PSA_capex_USD_1000m3_hr = PSA_capex_USD_1000m3_hr,
                                                    LL_capex_USD_1000mol_hr = LL_capex_USD_1000mol_hr,
                                                    electrolyte_cost_USD_kg = electrolyte_cost_USD_kg,
                                                    battery_capacity = battery_capacity,
                                                    kJ_per_kWh = kJ_per_kWh,
                                                    )
    ## Generate subparts of opex - SEIDER TEXTBOOK       
    df_sales = sales(product_name = product_name,
                    df_streams = df_streams,
                    product_cost_USD_kgprod = product_cost_USD_kgprod,
                    H2_cost_USD_kgH2 = H2_cost_USD_kgH2,
                    product_rate_kg_day = product_rate_kg_day,
                    capacity_factor = capacity_factor)
    df_feedstocks = nonaq_feedstocks(CO2_cost_USD_tCO2 = CO2_cost_USD_tCO2,
                    water_cost_USD_kg = water_cost_USD_kg,
                    solvent_name = solvent_name,
                    solvent_cost_USD_kg = solvent_cost_USD_kg,
                    df_streams = df_streams,
                    capacity_factor = capacity_factor)
    df_depreciation = depreciation(C_TDC = C_TDC, C_alloc = C_alloc) 
    df_general = general(sales_USD_year = df_sales.loc['Total', 'Earnings ($/yr)'])

    df_operations = operations(capacity_factor = capacity_factor) # Not used in Sinott
    df_maintenance = nonaq_maintenance(C_TDC = C_TDC,
                              df_capex_BM = df_capex_BM) # Not used in Sinott
    df_replacement = nonaq_replacement(stack_lifetime_years = stack_lifetime_years,
                                            df_capex_BM = df_capex_BM,
                                            df_streams = df_streams,
                                            lifetime_years = lifetime_years)      
    df_overhead = overhead(df_maintenance, df_operations)      # Not used in Sinott   
    df_taxes = taxes(C_TDC = C_TDC)     # Not used in Sinott

    # Add totals rows to subparts of opex
    for df in [df_feedstocks, df_operations, df_maintenance, df_overhead, df_utilities,
               df_taxes, df_depreciation, df_general, df_replacement]:
        totals(df,  product_name = product_name,
                product_rate_kg_day = product_rate_kg_day,
                capacity_factor = capacity_factor)

    ## Generate opex - SINNOTT TEXTBOOK
    df_opex, df_opex_totals = nonaq_opex_sinnott(C_ISBL = df_capex_totals.loc['Total bare-module investment', 'Cost ($)'], # currently C_TBM
            df_feedstocks = df_feedstocks,
            df_utilities = df_utilities,
            df_replacement = df_replacement,
            df_sales = df_sales,
            df_depreciation = df_depreciation,
            df_general = df_general,
            df_capex_BM = df_capex_BM,
            df_capex_totals = df_capex_totals,
            capacity_factor = capacity_factor,
            lifetime_years = lifetime_years,
            product_name = product_name,
            product_rate_kg_day = product_rate_kg_day
            )

    #     ## Generate opex - SEIDER TEXTBOOK
    # df_opex, df_opex_totals = nonaq_opex_seider(df_feedstocks = df_feedstocks,
    #         df_utilities = df_utilities,
    #         df_sales = df_sales,
    #         df_operations = df_operations,
    #         df_capex_totals = df_capex_totals,
    #         df_maintenance = df_maintenance,
    #         df_replacement = df_replacement,
    #         df_overhead = df_overhead,
    #         df_taxes = df_taxes,
    #         df_depreciation = df_depreciation,
    #         df_general = df_general,
    #         capacity_factor = capacity_factor,
    #         lifetime_years = lifetime_years,
    #         product_name = product_name,
    #         product_rate_kg_day = product_rate_kg_day
    #     )

    # Calculate NPV at 15% interest rate
    df_cashflows, cashflows, NPV = cashflow_years(    
        plant_lifetime = int(lifetime_years),
        depreciation_schedule = 'linear', # 'MACRS' or 'linear'
        D = 0, # optional, used for MACRS only - depreciation%
        depreciation_lifetime = 12, # at roughly 8% depreciation per year, used elsewhere. optional, used for linear only - total time before salvage value is recovered
        salvage_value = 0, # conservative assumption. optional, used for linear only - fraction of original capital that is recovered
        interest = 0.15, # typical assumption
        f = 0.03, # typical inflation %
        sales = df_sales.loc['Total', 'Earnings ($/yr)'],
        production_cost = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
        C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)'], # df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] 
        C_WC = df_capex_totals.loc['Working capital', 'Cost ($)'],
        t = 0.26, # tax in % per year,
        )

    ## Calculate IRR at 0 salvage value
    IRR = calculate_IRR(    
        plant_lifetime = int(lifetime_years),
        depreciation_schedule = 'linear', # 'MACRS' or 'linear'
        D = 0, # optional, used for MACRS only - depreciation%
        depreciation_lifetime = 12, # at roughly 8% depreciation per year, used elsewhere. optional, used for linear only - total time before salvage value is recovered
        salvage_value = 0, # conservative assumption. optional, used for linear only - fraction of original capital that is recovered
        f = 0.03, # typical inflation %
        sales = df_sales.loc['Total', 'Earnings ($/yr)'],
        production_cost = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
        C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)'], # df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] 
        C_WC = df_capex_totals.loc['Working capital', 'Cost ($)'],
        t = 0.26, # tax in % per year,
        )

    breakeven_price_USD_kgprod = calculate_breakeven_price(
        plant_lifetime = int(lifetime_years),
        depreciation_schedule = 'linear', # 'MACRS' or 'linear'
        D = 0, # optional, used for MACRS only - depreciation%
        depreciation_lifetime = 12, # at roughly 8% depreciation per year, used elsewhere. optional, used for linear only - total time before salvage value is recovered
        salvage_value = 0, # conservative assumption. optional, used for linear only - fraction of original capital that is recovered
        interest  = 0.15, # interest %
        f = 0.03, # typical inflation %
        product_rate_kg_day = product_rate_kg_day, # production in kg/day
        H2_rate_kg_day = df_streams.loc['H2 outlet', 'Mass flow rate (kg/day)'], # production in kg/day
        capacity_factor = capacity_factor, # capacity factor as a fraction of days in a year
        production_cost = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
        H2_price_USD_kgH2 = H2_cost_USD_kgH2,
        C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)'], # df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] 
        C_WC = df_capex_totals.loc['Working capital', 'Cost ($)'],
        t = 0.26, # tax in % per year,
        )
    
    return df_capex_BM, df_capex_totals, df_costing_assumptions, df_depreciation, df_electrolyzer_assumptions, df_electrolyzer_streams_mol_s,\
            df_energy, df_feedstocks, df_general, df_maintenance, df_replacement, df_operations, df_opex, df_opex_totals, df_outlet_assumptions,\
            df_overhead, df_potentials, df_sales, df_streams, df_streams_formatted, df_taxes, df_utilities, df_cashflows, \
            cashflows, NPV, IRR, breakeven_price_USD_kgprod



