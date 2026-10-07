# Data distribution and attribution

The MIT license covers the project code. It does **not** relicense third-party data. The `data-snapshots-v1` release distributes only the prepared historical climate inputs listed below. The archives preserve the original NetCDF bytes and their recorded hashes.

| Material | Distribution decision | Source and conditions |
|---|---|---|
| Prepared SEAS5 precipitation | Included in the data release | ECMWF / Copernicus Climate Change Service, [seasonal monthly single levels](https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels), DOI [10.24381/cds.68dd14c3](https://doi.org/10.24381/cds.68dd14c3). The [CDS license](https://cds.climate.copernicus.eu/licences/cc-by) is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the source and identify the modifications. |
| Prepared CFSv2 precipitation | Included in the data release | NOAA/NWS/NCEP CFSv2, acquired through the [IRI Data Library](https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.HINDCAST/.PENTAD_SAMPLES_FULL/.prec/). Underlying US federal weather data are public domain under the [NWS use policy](https://www.weather.gov/disclaimer). Retain source attribution; these modified fields are not an official NOAA forecast product. See also [NCEI CFS metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00877). |
| Prepared ERSSTv5 SST | Optional research-only release asset | NOAA/NCEI US federal data. Cite Huang et al. (2017), *NOAA Extended Reconstructed Sea Surface Temperature (ERSST), Version 5*, DOI [10.7289/V5T72FNM](https://doi.org/10.7289/V5T72FNM), and the methods paper [10.1175/JCLI-D-16-0836.1](https://doi.org/10.1175/JCLI-D-16-0836.1). See [dataset use constraints](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00927). |
| Official competition observations, atmospheric fields, test features and sample CSV | Not redistributed | Obtain from [Kaggle](https://www.kaggle.com/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul/data) under its access conditions. This project does not infer redistribution permission from participation in the competition. |
| Credentials, account files and local paths | Never included | Downloading these release assets needs no account or token. |

## Changes to the source data

**SEAS5:** ECMWF system 51, forecast month 2, regional extraction and ensemble averaging. Rates were converted from m/s to mm/day; small negative packing residuals were handled per member according to the recorded audit. Contains modified Copernicus Climate Change Service information (2026). Neither the European Commission nor ECMWF is responsible for uses of this information.

**CFSv2:** `PENTAD_SAMPLES_FULL`, IRI lead 1.5, regional extraction and audited ensemble averaging into mm/day. The source manifests record the August 2019 unavailable-member exception. That historical exception is not permission to accept incomplete future ensembles. NOAA/NCEP supplies the model data; IRI supplies the access and processing service. Project packaging and derived fields imply no endorsement by either institution.

**ERSSTv5:** the archived 4-degree SST grid used by the research experiment, January 1982–November 2024. The file contains SST fields, not fitted PCA components. PCA, climatologies and masks must still be fitted within each training block.

These are fixed historical snapshots acquired in September 2026, not an operational feed. Publication does not establish that revised historical observations were available in their present form at every past forecast date. See [temporal validity](TEMPORAL_VALIDITY.md). Preserve each archive's `NOTICE.txt` when sharing the prepared inputs.

Source conditions checked on 7 October 2026. [Data preparation](DATA.md) describes filenames, coverage and acquisition separately from these rights and attribution notes.
