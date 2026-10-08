# Data

All four datasets are public. They are **not** redistributed in this repository; download them into
`data/raw/` (or any folder pointed to by the environment variable `VIGIL_DATA`) with the layout below.
Paths are configured in `configs/data.yaml`.

```
data/raw/
├── caf/
│   ├── CAF_Sensor_Dataset.zip
│   └── extracted/                 # unzip CAF_Sensor_Dataset.zip here
├── power/
│   └── POWER_hourly_2011-2016.csv
├── scan/
│   └── SCAN2198_hourly_2013-2016.csv
└── bitbrains/
    ├── fastStorage.zip
    └── extracted/2013-8/          # 1250 per-VM CSV files
```

## 1. Cook Agronomy Farm (CAF) soil sensor network — primary data

* **Content:** hourly volumetric water content (VWC) and soil temperature at 30, 60, 90, 120 and 150 cm
  from 42 logger locations (April 2007 – June 2016), quality-control flag files, logger coordinates,
  10-m DEM, apparent electrical conductivity, Bt-horizon maps, soil texture and bulk density, crop
  assignments of the field strips.
* **Source:** Gasch, C.K., Brown, D.J. (2017). *Data from: A field-scale sensor network data set for
  monitoring and modeling the spatial and temporal variation of soil moisture in a dryland agricultural
  field.* USDA National Agricultural Library, Ag Data Commons. https://doi.org/10.15482/USDA.ADC/1349683
* **Licence:** U.S. Public Domain / CC0 (Ag Data Commons).
* **Download:** the file `CAF_Sensor_Dataset.zip` from the DOI landing page; unzip to `data/raw/caf/extracted/`.
* **Data paper:** Gasch, C.K., Brown, D.J., Campbell, C.S., Cobos, D.R., Brooks, E.S., Chahal, M., Poggio, M. (2017),
  Water Resources Research 53, 10878–10887, https://doi.org/10.1002/2017WR021307

## 2. NASA POWER hourly weather — model covariates

* **Content:** hourly 2-m air temperature (`T2M`), relative humidity (`RH2M`), corrected precipitation
  (`PRECTOTCORR`), 2-m wind speed (`WS2M`) and all-sky shortwave irradiance (`ALLSKY_SFC_SW_DWN`), local
  solar time.
* **Location / period:** 46.7807 N, 117.0821 W (CAF centroid), 2011-01-01 – 2016-06-16 (local solar time).
  Irradiance is delivered in MJ h⁻¹ m⁻² and converted to W m⁻² by the loader.
* **Source:** NASA Langley Research Center POWER Project, https://power.larc.nasa.gov/ (Data Access
  Viewer, or the API, e.g. `https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=T2M,RH2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN&community=AG&longitude=-117.0821&latitude=46.7807&start=20110101&end=20160616&format=CSV&time-standard=LST`).
* **Licence:** public; acknowledgement requested ("These data were obtained from the NASA Langley Research
  Center (LaRC) POWER Project funded through the NASA Earth Science/Applied Science Program.").
* Save the CSV (with its header block) as `data/raw/power/POWER_hourly_2011-2016.csv`.

## 3. NRCS SCAN site 2198 "Cook Farm Field D" — on-farm station

* **Content:** hourly air temperature, relative humidity, precipitation, solar radiation, wind and shallow
  soil moisture/temperature.
* **Period used:** 2013-07-24 – 2016-06-30 (the station reports no hourly data from 2016-01-05 to July 2016).
* **Source:** USDA NRCS National Water and Climate Center, Report Generator,
  https://wcc.sc.egov.usda.gov/reportGenerator/ (station 2198:WA:SCAN, hourly, all elements, CSV).
* **Licence:** U.S. Government work, public domain.
* Save as `data/raw/scan/SCAN2198_hourly_2013-2016.csv` (comment lines starting with `#` are skipped).

## 4. GWA-T-12 Bitbrains (fastStorage) — co-tenant VM load

* **Content:** 5-min CPU, memory, disk and network metrics of 1,250 VMs of a managed-hosting provider
  (August–September 2013).
* **Source:** Grid Workloads Archive, TU Delft, http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains
  (also mirrored on Kaggle). Shen, S., van Beek, V., Iosup, A. (2015), CCGrid, https://doi.org/10.1109/CCGrid.2015.60
* **Licence:** free for research use with citation of the GWA and the paper above.
* Unzip `fastStorage.zip` to `data/raw/bitbrains/extracted/` (the files are in `2013-8/`).

## Processed data

`python -m vigil.data.build` writes `data/processed/caf_processed.npz` (QC-ed hourly tensors, events,
graph, splits), `build_report.json`, `static_features.csv` and `crop_assignment.csv`. These files are
derived data and can be regenerated at any time.
