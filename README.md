### CO<sub>2</sub>R Costing Dashboard

**Cite this work : [__]()**

This is the source code for the [CO<sub>2</sub>R Costing Dashboard web app](https://co2r-dashboard.streamlit.app/). To use the app to generate technoeconomics for a CO<sub>2</sub> reduction process, please go straight to the [website](https://co2r-dashboard.streamlit.app/). You only need this repository if you want to see or change the backend for the web app.

**Front end**

Front page and pie charts for cost breakdown: `00_Intro_and_pie_charts.py`

Bar charts for CO<sub>2</sub>R to CO: `\pages\01_Bar_charts_CO.py`

Bar charts for CO<sub>2</sub>R to ethylene: `\pages\02_Bar_charts_Ethylene.py`


**Models**

Electrolyzer: `ElectrolyzerModel.py`

Mass and energy balances: `DownstreamProcessModel.py`

Technoeconomics: `ProcessEconomics.py`

Single run execution: `TEA_SingleRun.py`

**Data**

Data for constants, utilities and products: `DataForTEA.xlsx`


**Other**

Webpage formatting: `\.streamlit\config.toml`

Package specifications: 
- `requirements.txt` (installed through pip)
- `packages.txt` (installed through apt-get in Streamlit's Debian container)

License: `LICENSE.md`

Readme: `README.md`