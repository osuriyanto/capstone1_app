# SolarROI (Western Australia Household)

## About this app

This is your helper to estimate the payback time and return on investment for your rooftop solar panels. 
The input datasets are based on Western Australia's latest published data by [AEMO (Australian Energy Market Operator)](https://www.aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem/data-wem/market-data-wa) and [CER (Clean Energy Regulator)](https://cer.gov.au/markets/reports-and-data/small-scale-installation-postcode-data). 

## Inputs

What you need to provide:
1. Your electricity usage summary (*before installing solar panels*), ideally from the recent full year.
   * This can be downloaded as CSV file from the online Synergy account: ![Synergy account summary image](Synergy_Account_Summary.png)
   * If this is not possible, enter the best estimate of your annual electricity usage in kWh.
3. Details of the solar panels to be installed:
   * PV size, kW.
   * Capex, $ (user input, or use the built-in estimate based on PV size).
4. Fraction of PV generation for self-consumption.
   * Tip: set self-consumption to reflect appliance timing and household profile. 

## Assumptions and Fixed Inputs

### Fixed Inputs
1. Estimated DPV (distributed photovoltaics) - the collective WA's generated solar power - is published in [AEMO's public website](https://www.aemo.com.au/energy-systems/electricity/wholesale-electricity-market-wem/data-wem/market-data-wa). In this app, the [2024](https://data.wa.aemo.com.au/datafiles/distributed-pv/distributed-pv-2024.csv) full-year data is used.
2. The app assumes [Synergy's A1 Home Plan](https://www.synergy.net.au/Your-home/Energy-plans/Home-Plan-A1), which is the majority for WA's households.
3. Peak and off-peak PV export rebate follows the 2025 rate of [DEBS (Distributed Energy Buyback Scheme)](https://www.synergy.net.au/Your-home/Help-and-advice/Solar-credits-and-upgrades/What-will-be-the-DEBS-Buyback-rate). 
4. Average exported PV generation during peak hour is 20% of total PV export.
5. Total installed solar panels in WA (last updated in September 2025) is 3 GW according to [CER](https://cer.gov.au/document/sres-postcode-data-capacity-2011-to-present-and-totals).

### Assumptions
1. Discount rate is 0.05 (can be adjusted by user between 0.04 and 0.1).
2. Solar panel performance ratio is 0.85 (can be adjusted by user between 0.7 and 0.9).
3. Same annual savings throughout the year.
4. PV annual degradation is not included.
5. Electricity tariff increase is not included.

## Outputs
1. :date: Estimated payback time, with a yearly details table available for download.
2. :date: First year's monthly details of PV generation, electricity bill with and without solar, and total savings, also available for download.
3. :bar_chart: Visualisation of monthly electricity usage, PV generation, PV self-consumption, and the comparison of electricity bills.
4. :bar_chart: Visualisation of savings from PV self-consumption, peak and off-peak export rebate. 
