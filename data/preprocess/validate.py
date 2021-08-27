from pathlib import Path
import Constants 

def validate_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    assert (img_dir/'mimic-cxr-2.0.0-metadata.csv.gz').is_file()
    assert (img_dir/'mimic-cxr-2.0.0-negbio.csv.gz').is_file()
    assert (img_dir/'mimic-cxr-2.0.0-negbio.csv.gz').is_file()
    assert (img_dir/'files/p19/p19316207/s55102753/31ec769b-463d6f30-a56a7e09-76716ec1-91ad34b6.jpg').is_file()

def validate_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    assert (img_dir/'map.csv').is_file()
    assert (img_dir/'CheXpert-v1.0/train.csv').is_file()
    assert (img_dir/'CheXpert-v1.0/train/patient48822/study1/view1_frontal.jpg').is_file()
    assert (img_dir/'CheXpert-v1.0/valid/patient64636/study1/view1_frontal.jpg').is_file()

def validate_nih():
    img_dir = Path(Constants.image_paths['NIH'])
    assert (img_dir/'Data_Entry_2017.csv').is_file()
    assert (img_dir/'images/00002072_003.png').is_file()

def validate_splits():
    for env in Constants.df_paths:
        for fold in Constants.df_paths[env]:
            assert Path(Constants.df_paths[env][fold]).is_file(), Constants.df_paths[env][fold]

def validate_camelyon():
    img_dir = Path(Constants.camelyon_path)
    assert (img_dir/'metadata.csv').is_file()
    assert (img_dir/'patches'/'patient_017_node_4'/'patch_patient_017_node_4_x_3040_y_23520.png').is_file()

def validate_poverty():
    img_dir = Path(Constants.poverty_path)
    assert (img_dir/'dhs_metadata.csv').is_file()
    assert (img_dir/'landsat_poverty_imgs.npy').is_file()

def validate_cxr():
    validate_mimic()
    validate_cxp()
    validate_nih()

def validate_wilds():
    validate_camelyon()
    validate_poverty()