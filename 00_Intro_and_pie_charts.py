
################################################################################
################################### IMPORTS ####################################
################################################################################

### Import packages

import streamlit as st
# from streamlit_super_slider import st_slider as superslider

### This page makes the page navigation outline
### IMPORTANT: this file CANNOT be renamed!

page_00 = st.Page(
    "pages/00_Base_case.py", 
    title = "Base case techno-economics", 
    icon = ":material/home:", 
    default = True,)
page_01 = st.Page(
    "pages/01_CO_MEA.py", 
    title = "Sensitivity - CO, zero-gap", 
    icon = None, # ":material/analytics:",
    default = False,
    url_path = 'MEA-CO')
page_02 = st.Page(
    "pages/02_Ethylene_MEA.py", 
    title = "Sensitivity - ethylene, zero-gap", 
    icon = None, # ":material/analytics:", 
    default = False,
    url_path = 'MEA-ethylene')
page_03 = st.Page(
    "pages/03_CO_Nonaq.py", 
    title = "Non-aqueous CO2R to CO", 
    icon = None, # ":material/calculate:",
    default = False,
    url_path = 'nonaqueous-CO')
page_04 = st.Page(
    "pages/04_OA_Nonaq.py", 
    title = "Non-aqueous CO2R to oxalic acid", 
    icon = None, # ":material/calculate:", 
    default=False,
    url_path = 'nonaqueous-OA')

pg = st.navigation(
        pages = {
            "Aqueous MEA": [page_00, page_01, page_02],
            "Non-aqueous": [page_03, page_04],
        },
        position = 'top', # 'sidebar'
        # expanded = True,
    )

pg.run()
