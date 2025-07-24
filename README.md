### CO<sub>2</sub>R Costing Dashboard

**Cite this work** : 
> Da Cunha, S.; Resasco, J. Insights from Techno-Economic Analysis Can Guide the Design of Low-Temperature CO₂ Electrolyzers towards Industrial Scaleup. ACS Energy Lett. 2024, 9, 11, 5550–5561. DOI: [10.1021/acsenergylett.4c02647](https://pubs.acs.org/doi/10.1021/acsenergylett.4c02647). 

> Da Cunha, S.; Resasco, J. Techno-Economic Assessment of Non-Aqueous CO2 Reduction. ChemRxiv, 2025. DOI:[10.26434/chemrxiv-2025-k071x](https://doi.org/10.26434/chemrxiv-2025-k071x).

This tool is a techno-economic assessment (TEA) to generate the capital and operating cost for a CO₂ reduction process converting captured CO₂ into either CO or ethylene. It is based on the model in our [2024 paper](https://pubs.acs.org/doi/10.1021/acsenergylett.4c02647). It can also be used to calculate costs for a non-aqueous CO₂R system, as in our [2025 paper](https://doi.org/10.26434/chemrxiv-2025-k071x).

This is the source code for the [CO<sub>2</sub>R Costing Dashboard web app](https://co2r-dashboard.streamlit.app/). To use the app to generate technoeconomics for a CO<sub>2</sub> reduction process, please go straight to the [website](https://co2r-dashboard.streamlit.app/). You only need this repository if you want to see or change the backend for the web app.

**Front end**

Page navigation:
`00_Intro_and_pie_charts.py`

Front page and pie charts for cost breakdown: 
`pages\00_Base_case.py`

Bar charts for CO<sub>2</sub>R to CO: 
`pages\01_CO_MEA.py`

Bar charts for CO<sub>2</sub>R to ethylene: `pages\02_Ethylene_MEA.py`

Techno-economics for non-aqueous CO<sub>2</sub>R to CO: 
`pages\03_CO_Nonaq.py` 

Techno-economics for non-aqueous CO<sub>2</sub>R to oxalic acid: 
`pages\04_OA_Nonaq.py`


**Aqueous MEA Models**

Electrolyzer: `ElectrolyzerModel.py`

Mass and energy balances: `DownstreamProcessModel.py`

Technoeconomics: `ProcessEconomics.py`

Single run execution: `TEA_SingleRun.py`


**Non-aqueous Flow Cell Models**

Electrolyzer: `NonAqElectrolyzerModel.py`

Mass and energy balances: `NonAqDownstreamProcessModel.py`

Technoeconomics: `NonAqProcessEconomics.py`

**Data**

Data for constants, utilities and products for aqueous CO<sub>2</sub>R: `Supplementary Workbook.xlsx`

Data for constants, utilities and products for non-aqueous CO<sub>2</sub>R: `Supplementary Workbook Non-Aq.xlsx`

**Other**

Webpage formatting: `\.streamlit\config.toml`

Package specifications: 
- `requirements.txt` (installed through pip)
- `packages.txt` (installed through apt-get in Streamlit's Debian container)

License: `LICENSE.md`

Readme: `README.md`
