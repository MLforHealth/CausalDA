# Chest X-rays
## Downloading the Data
### MIMIC-CXR
1. [Obtain access](https://mimic-cxr.mit.edu/about/access/) to the MIMIC-CXR-JPG Database Database on PhysioNet and download the [dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We recommend downloading from the GCP bucket:

```
gcloud auth login
mkdir MIMIC-CXR-JPG
gsutil -m rsync -d -r gs://mimic-cxr-jpg-2.0.0.physionet.org MIMIC-CXR-JPG
```

### CheXpert
1. Sign up with your email address [here](https://stanfordmlgroup.github.io/competitions/chexpert/).

2. Download either the original or the downsampled dataset (we recommend the downsampled version - `CheXpert-v1.0-small.zip`) and extract it.


### ChestX-ray8

1. Download the `images` folder and `Data_Entry_2017_v2020.csv` from the [NIH website](https://nihcc.app.box.com/v/ChestXray-NIHCC).

2. Unzip all of the files in the `images` folder.

## Data Processing
1. In `Constants.py`, update `image_paths` to point to each of the three directories that you downloaded.

2. Run `python -m data.preprocess.preprocess_cxr`. 

3. (Optional) If you are training a lot of models, it _might_ be faster to first cache all images to binary 224x224 files on disk. In this case, you should update the `cache_dir` path in `Constants.py` and then run `python -m data.preprocess.cache_data`, optionally parallelizing over `--env_id {0, 1, 2}` for speed. To use the cached files, pass --cache_cxr to train.py.


# WILDS
1. Download the datasets for [Camelyon17-v1.0](https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/) and [PovertyMap-1.0](https://worksheets.codalab.org/rest/bundles/0x9a2add5219db4ebc89965d7f42719750/contents/blob/) and extract it.

2. Update `wilds_root_dir` in `Constants.py`.

3. Run `python -m data.preprocess.preprocess_wilds`.