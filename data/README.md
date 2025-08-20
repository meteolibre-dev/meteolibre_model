---
tags:
- climate
size_categories:
- 100B<n<1T
---


![image/png](https://cdn-uploads.huggingface.co/production/uploads/6229c4e279337db2b0bf0c2e/40YOSNBG__f7mbWn1hb3w.png)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6229c4e279337db2b0bf0c2e/dYgWd6vttXGnCatRSDe0q.png)

<img src="https://cdn-uploads.huggingface.co/production/uploads/6229c4e279337db2b0bf0c2e/O2JSZdbc_aEnHoLGtbcRg.png" alt="drawing" width="200"/>

This dataset is a ready-to-use fusion of multiple datasets for the France region, including:

* **Ground station data:** Sourced from [data.gouv.fr](https://www.data.gouv.fr/datasets/) with a 1-hour resolution. It includes 7 weather KPIs: `"RR1"`, `"FF"`, `"DD"`, `"T"`, `"U"`, `"PMER"`, and `"VV"`.

* **Radar imagery:** Radar rainfall accumulation images from the [Météo-France open data initiative](https://portail-api.meteofrance.fr/web/fr/).

* **Geospatial data:** Land cover and ground height information from [EarthEnv](https://www.earthenv.org/landcover) and [OpenTopography](https://portal.opentopography.org/).

* **Satellite imagery:** Sourced from the [EUMETSAT platform](https://www.eumetsat.int/), containing a subset of channels: `vis_04`, `vis_09`, `nir_13`, and `nir_16`.

---

## Time Period

The data covers the period from **January 2025 to July 2025**.

---

## Resolution and Projection

All data have a **1km spatial resolution** (using EPSG:32630) and a **30-minute temporal resolution**.

---

## Data Description

The final dataset is structured in `.zip` files as follows:

* `ground_height_image/`: Contains ground height information as NumPy `.npz` files (256x256 images).

* `groundstation/`: Contains ground station readings as arrays of shape `(9, 256, 256, 7)`, where 9 is the number of time steps and 7 is the number of KPIs.

* `index.json`: A JSON file that indexes the entire dataset.

* `landcover/`, `landcover_image/`: Contain land cover data (256x256 images).

* `radar/`: Contains radar rainfall accumulation imagery as arrays of shape `(9, 256, 256)`, where 9 is the number of time steps.
 
* `satellite/`: Contains satellite imagery as arrays of shape `(9, 256, 256, 4)`, where 9 is the number of time steps and 4 is the number of selected channels.

Each subdirectory contains `.npz` files for different timestamps and locations, all indexed by `index.json`.

---

## Data Visualization

Here's a quick visualization of the data:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/6229c4e279337db2b0bf0c2e/TrVFQDFRW2jXfKJ-6ve7d.png)

We extracted full images of France and segmented them into smaller patches for easier ingestion into a training pipeline.

## Quick start (remote)

```python
from datasets import load_dataset

# The command is exactly the same!
# The library handles the sharded files behind the scenes.
ds = load_dataset("meteolibre-dev/mtg_meteofrance_256", streaming=True)

# You can iterate over it as if it were a single file
for example in ds["train"]:

    # Get the byte string
    satellite_bytes = example['satellite']
    # Convert bytes back to a NumPy array
    # You need to specify the correct dtype and shape
    satellite_array = np.frombuffer(satellite_bytes, dtype=np.int16).reshape(9, 4, 256, 256)

    hour = example['hour']
    minute = example['minute']
    datetime = example['datetime']
    radar =  np.frombuffer(example['radar'], dtype=np.float32).reshape(9, 256, 256)
    groundstation =  np.frombuffer(example['groundstation'], dtype=np.float32).reshape(9, 256, 256, 7)
    ground_height =  np.frombuffer(example['ground_height'], dtype=np.float32).reshape(256, 256)
    landcover =  np.frombuffer(example['landcover'], dtype=np.float32).reshape(256, 256, 4)

```


## Quick start (local)

First the command : 

```bash
hf download meteolibre-dev/mtg_meteofrance_256 --repo-type dataset --local-dir data/
```