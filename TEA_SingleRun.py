# %% [markdown]
# ## Running notes
# Date updated: 2024/06/11 \
# Update notes: Strict function inputs\
# Contact: Shashwati da Cunha
# 
# ## Instructions
# 1. This is only a collection of functions. Please call it externally.
# 
# #### Notes:
# 
# #### To do:
# 

# %%
# Uncommment to run notebook standalone
# %run "20240611_0_ElectrolyzerModel.ipynb"
# %run "20240611_1_DownstreamProcessModel.ipynb"
# %run "20240611_2_ProcessEconomics.ipynb"

import streamlit as st

from ElectrolyzerModel import *
from DownstreamProcessModel import *
from ProcessEconomics import *

# %%
## Check if SPC is reasonable given FE and crossover. Generate FE and SPC - if model_FE is true, FE will be modeled; either way, both will be checked for mass balance

@st.cache_data
def single_run(product_name,
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
        additional_opex_USD_kg,
        additional_capex_USD,
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
    df_electrolyzer_assumptions, df_potentials = cell_voltage(
        product_name = product_name,    
        product_rate_kg_day = product_rate_kg_day,
        FE_product = FE_product,
        FE_product_specified = FE_product_specified,
        FE_CO2R_0 = FE_CO2R_0,
        j_total_mA_cm2 = j_total_mA_cm2,
        R_ohmcm2 = R_ohmcm2,
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
        SPC = SPC,
        R = R,
        F = F
        ) 

    ## Generate stream information for electrolyzer outlets
    df_outlet_assumptions, df_electrolyzer_streams_mol_s = electrolyzer_SS_mass_balance(
        product_name = product_name,
        product_rate_kg_day = product_rate_kg_day,
        FE_product = FE_product,
        SPC = SPC,
        df_potentials = df_potentials, 
        crossover_ratio = crossover_ratio,
        excess_water_ratio = excess_water_ratio,
        electrolyte_conc = electrolyte_conc,
        cathode_outlet_humidity = cathode_outlet_humidity,
        j_total_mA_cm2 = j_total_mA_cm2,
        df_products = df_products,     
        carbon_capture_efficiency = carbon_capture_efficiency ,
        MW_CO2 = MW_CO2,
        R = R,
        F = F,
        ) 

    ## Generate stream table
    df_streams = blank_stream_table(product_name = product_name) # Generate blank stream table
    df_streams = update_stream_table(product_name = product_name, 
                    df_products = df_products, 
                    df_streams = df_streams, 
                    df_electrolyzer_streams_mol_s = df_electrolyzer_streams_mol_s,
                    electrolyte_conc = electrolyte_conc,
                    density_kgm3 = density_kgm3,
                    cathode_outlet_humidity = cathode_outlet_humidity,
                    T_streams = T_streams,
                    T_sep = T_sep,
                    P = P,
                    MW_CO2 = MW_CO2,
                    MW_H2O = MW_H2O,
                    MW_O2 = MW_O2,
                    MW_MX = MW_MX,                 
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
    df_energy = energy_table(product_name = product_name,
             df_products = df_products,
             df_potentials = df_potentials,
             df_streams = df_streams, 
             PSA_second_law_efficiency = PSA_second_law_efficiency,
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

    df_costing_assumptions = costing_assumptions(product_name = product_name,
                                                 product_cost_USD_kgprod = product_cost_USD_kgprod,
                                                H2_cost_USD_kgH2 = H2_cost_USD_kgH2,
                                                electricity_cost_USD_kWh = electricity_cost_USD_kWh,
                                                water_cost_USD_kg = water_cost_USD_kg,
                                                CO2_cost_USD_tCO2 = CO2_cost_USD_tCO2,
                                                electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2,
                                                lifetime_years = lifetime_years,
                                                stack_lifetime_years = stack_lifetime_years,
                                                capacity_factor = capacity_factor)
    ## Generate capex
    df_capex_BM, df_capex_totals, C_TDC, C_alloc = capex(product_name = product_name,
        area_m2 = df_potentials.loc['Area', 'Value'],
        df_utilities = df_utilities,
        df_streams = df_streams,
        product_rate_kg_day = product_rate_kg_day,
        battery_capex_USD_kWh = battery_capex_USD_kWh,        
        electrolyzer_capex_USD_m2 = electrolyzer_capex_USD_m2 ,
        battery_capacity = battery_capacity,
        kJ_per_kWh = kJ_per_kWh,
        is_additional_capex = is_additional_capex,
        additional_capex_USD = additional_capex_USD)

    ## Generate subparts of opex - SEIDER TEXTBOOK       
    df_sales = sales(product_name = product_name,
                    df_streams = df_streams,
                    product_cost_USD_kgprod = product_cost_USD_kgprod,
                    H2_cost_USD_kgH2 = H2_cost_USD_kgH2,
                    product_rate_kg_day = product_rate_kg_day,
                    capacity_factor = capacity_factor)
    df_feedstocks = feedstocks(df_costing_assumptions,
                              df_streams,
                          capacity_factor)
    df_depreciation = depreciation(C_TDC = C_TDC, C_alloc = C_alloc) 
    df_general = general(df_sales = df_sales)

    df_operations = operations(capacity_factor = capacity_factor) # Not used in Sinott
    df_maintenance = maintenance(C_TDC = C_TDC,
                                 df_capex_BM = df_capex_BM)
    df_stack_replacement = stack_replacement(df_capex_BM = df_capex_BM,
                                 stack_lifetime_years = stack_lifetime_years,
                                 lifetime_years = lifetime_years)      
    df_overhead = overhead(df_maintenance, df_operations)      # Not used in Sinott   
    df_taxes = taxes(C_TDC = C_TDC)     # Not used in Sinott

    # Add totals rows to subparts of opex
    for df in [df_feedstocks, df_utilities, df_operations, df_maintenance, df_overhead, 
               df_taxes, df_depreciation, df_general, df_stack_replacement]:
        totals(df,  product_name = product_name,
                    product_rate_kg_day = product_rate_kg_day,
                    capacity_factor = capacity_factor)
    
    ## Generate opex - SINNOTT TEXTBOOK
    df_opex, df_opex_totals = opex_sinnott(C_ISBL = df_capex_totals.loc['Total bare-module investment', 'Cost ($)'], # currently C_TDC
             df_feedstocks = df_feedstocks,
             df_utilities = df_utilities,
             df_stack_replacement = df_stack_replacement,
             df_sales = df_sales,
             df_depreciation = df_depreciation,
             df_general = df_general,
             df_capex_BM = df_capex_BM,
             df_capex_totals = df_capex_totals,
             capacity_factor = capacity_factor,
             lifetime_years = lifetime_years,
             product_name = product_name,
             product_rate_kg_day = product_rate_kg_day,
             is_additional_opex = is_additional_opex,
             additional_opex_USD_kg= additional_opex_USD_kg
             )

    #     ## Generate opex - SEIDER TEXTBOOK
    # df_opex, df_opex_totals = opex_seider(df_feedstocks = df_feedstocks,
    #         df_utilities = df_utilities,
    #         df_sales = df_sales,
    #         df_operations = df_operations,
    #         df_capex_totals = df_capex_totals,
    #         df_maintenance = df_maintenance,
    #         df_stack_replacement = df_stack_replacement,
    #         df_overhead = df_overhead,
    #         df_taxes = df_taxes,
    #         df_depreciation = df_depreciation,
    #         df_general = df_general,
    #         capacity_factor = capacity_factor,
    #         lifetime_years = lifetime_years,
    #         product_name = product_name,
    #         product_rate_kg_day = product_rate_kg_day,
    #         cell_E_V = cell_E_V,
    #         is_additional_opex = is_additional_opex,
    #         additional_opex_USD_kg = additional_opex_USD_kg
    #     )

    # print('Here')

    # For now, NPV, IRR and breakeven cost are not displayed. Save computation time by not calculating them

    # Calculate NPV at 15% interest rate
    # df_cashflows, cashflows, NPV = cashflow_years(    
    #     plant_lifetime = int(lifetime_years),
    #     depreciation_schedule = 'linear', # 'MACRS' or 'linear'
    #     D = 0, # optional, used for MACRS only - depreciation%
    #     depreciation_lifetime = 12, # at roughly 8% depreciation per year, used elsewhere. optional, used for linear only - total time before salvage value is recovered
    #     salvage_value = 0, # conservative assumption. optional, used for linear only - fraction of original capital that is recovered
    #     interest = 0.15, # typical assumption
    #     f = 0.03, # typical inflation %
    #     sales = df_sales.loc['Total', 'Earnings ($/yr)'],
    #     production_cost = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    #     C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)'], # df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] 
    #     C_WC = df_capex_totals.loc['Working capital', 'Cost ($)'],
    #     t = 0.2, # tax in % per year,
    #     )

    ## Calculate IRR at 0 salvage value
    # IRR = calculate_IRR(    
    #     plant_lifetime = int(lifetime_years),
    #     depreciation_schedule = 'linear', # 'MACRS' or 'linear'
    #     D = 0, # optional, used for MACRS only - depreciation%
    #     depreciation_lifetime = 12, # at roughly 8% depreciation per year, used elsewhere. optional, used for linear only - total time before salvage value is recovered
    #     salvage_value = 0, # conservative assumption. optional, used for linear only - fraction of original capital that is recovered
    #     f = 0.03, # typical inflation %
    #     sales = df_sales.loc['Total', 'Earnings ($/yr)'],
    #     production_cost = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    #     C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)'], # df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] 
    #     C_WC = df_capex_totals.loc['Working capital', 'Cost ($)'],
    #     t = 0.2, # tax in % per year,
    #     )

    # breakeven_price_USD_kgprod = calculate_breakeven_price(
    #     plant_lifetime = int(lifetime_years),
    #     depreciation_schedule = 'linear', # 'MACRS' or 'linear'
    #     D = 0, # optional, used for MACRS only - depreciation%
    #     depreciation_lifetime = 12, # at roughly 8% depreciation per year, used elsewhere. optional, used for linear only - total time before salvage value is recovered
    #     salvage_value = 0, # conservative assumption. optional, used for linear only - fraction of original capital that is recovered
    #     interest  = 0.15, # interest %
    #     f = 0.03, # typical inflation %
    #     product_rate_kg_day = product_rate_kg_day, # production in kg/day
    #     capacity_factor = capacity_factor, # capacity factor as a fraction of days in a year
    #     production_cost = df_opex_totals.loc['Production cost', 'Cost ($/yr)'], 
    #     C_TDC = df_capex_totals.loc['Total depreciable capital', 'Cost ($)'], # df_capex_totals.loc['Total depreciable capital', 'Cost ($)'] 
    #     C_WC = df_capex_totals.loc['Working capital', 'Cost ($)'],
    #     t = 0.2, # tax in % per year,
    #     )
    
    return df_capex_BM, df_capex_totals, df_costing_assumptions, df_depreciation, df_electrolyzer_assumptions, df_electrolyzer_streams_mol_s,\
            df_energy, df_feedstocks, df_general, df_maintenance, df_operations, df_opex, df_opex_totals, df_outlet_assumptions,\
            df_overhead, df_potentials, df_sales, df_streams, df_streams_formatted, df_taxes, df_utilities 
    #  df_cashflows, \
    #         cashflows, NPV, IRR, breakeven_price_USD_kgprod