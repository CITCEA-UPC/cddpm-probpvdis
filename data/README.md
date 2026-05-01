# Data

This folder contains preprocessed input data for the **Ausgrid** dataset used in the paper.

## Source

**Ausgrid Solar Home Electricity Dataset**
- 300 residential customers in New South Wales, Australia
- Half-hourly resolution (T = 48 time steps per day)
- Period: July 2012 – June 2013
- Publicly available at: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data

After cleaning, 43 customers with anomalous or incomplete data were excluded (IDs listed in `src/main.py` → `DROP_CUSTOMERS`). The remaining 257 customers are split into:
- **107 training customers** (with submetered PV, used to train the model)
- **150 test customers** (held out for evaluation)

GHI data was retrieved from the [Open-Meteo Weather API](https://open-meteo.com/) using postcode centroids.

## Files

| File | Description |
|---|---|
| `pv_norm_cap_v2.csv` | PV generation normalised by estimated system capacity |
| `ghi_norm_v2.csv` | Global Horizontal Irradiance (GHI), normalised |
| `nc_norm_maxexp_v2.csv` | Net consumption normalised by maximum export |
| `nc_lag1_maxexp_v2.csv` | Net consumption lagged by 1 day, normalised by maximum export |
| `capacity_estimation_GG_pred.csv` | PV capacity estimates (predicted and real) per customer |

## Normalisation

- **PV**: divided by predicted PV capacity (`cap_pred`) so values are in [0, 1]
- **GHI**: min-max normalised per location
- **Net consumption**: normalised by the maximum export value (`max_export`) per customer
- **NC lag-1**: same normalisation scheme as NC, using the previous day's readings

## Note on the Dutch dataset

A second dataset from the Netherlands (Utrecht province) was used in the paper to evaluate cross-dataset generalisation. This dataset **cannot be shared** due to privacy constraints and is therefore not included here.
