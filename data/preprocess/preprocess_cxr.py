import sys
import os
import Constants
from pathlib import Path 
from data.preprocess.validate import validate_cxr
import pandas as pd

def preprocess_mimic():
    img_dir = Path(Constants.image_paths['MIMIC'])
    out_folder = img_dir/'causalda'
    out_folder.mkdir(parents = True, exist_ok = True)  

    patients = pd.read_csv(img_dir/'patients.csv.gz')
    labels = pd.read_csv(img_dir/'mimic-cxr-2.0.0-negbio.csv.gz')
    meta = pd.read_csv(img_dir/'mimic-cxr-2.0.0-metadata.csv.gz')

    df = meta.merge(patients, on = 'subject_id').merge(labels, on = ['subject_id', 'study_id'])
    df['age_decile'] = pd.cut(df['anchor_age'], bins = list(range(0, 100, 10))).apply(lambda x: f'{x.left}-{x.right}').astype(str)
    df['frontal'] = df.ViewPosition.isin(['AP', 'PA'])

    df['path'] = df.apply(lambda x: os.path.join('files', f'p{str(x["subject_id"])[:2]}', f'p{x["subject_id"]}', f's{x["study_id"]}', f'{x["dicom_id"]}.jpg'), axis = 1)
    reproduce_split(df.reset_index(drop = True), dataset = 'MIMIC', output_dir = out_folder)

def preprocess_cxp():
    img_dir = Path(Constants.image_paths['CXP'])
    out_folder = img_dir/'causalda'
    out_folder.mkdir(parents = True, exist_ok = True)  
    df = pd.read_csv(img_dir/"map.csv")

    df['subject_id'] = df['Path'].apply(lambda x: int(Path(x).parent.parent.name[7:]))
    reproduce_split(df.reset_index(drop = True), dataset = 'CXP', output_dir = out_folder)

def preprocess_nih():
    img_dir = Path(Constants.image_paths['NIH'])
    out_folder = img_dir/'causalda'
    out_folder.mkdir(parents = True, exist_ok = True)  
    df = pd.read_csv(img_dir/"Data_Entry_2017.csv")
    df['labels'] = df['Finding Labels'].apply(lambda x: x.split('|'))

    for label in Constants.take_labels:
        df[label] = df['labels'].apply(lambda x: label in x)
    reproduce_split(df.reset_index(drop = True), dataset = 'NIH', output_dir = out_folder)


def reproduce_split(preprocessed_df: pd.DataFrame, dataset: str, output_dir: str, 
        fold_mapping_path: str = os.path.join(os.path.dirname(os.path.realpath(__file__)),  'fold_mapping.csv')):
    fold_mapping = pd.read_csv(fold_mapping_path).query(f"dataset == '{dataset}'")
    subject_id_col = 'subject_id' if dataset in ['MIMIC', 'CXP'] else 'Patient ID'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True, parents = True)
    
    for fold in ['train', 'valid', 'test']:
        sub_df = preprocessed_df[preprocessed_df[subject_id_col].isin(fold_mapping.loc[fold_mapping['fold'] == fold, 'subject_id'])]
        print(dataset, fold, len(sub_df))
        sub_df.to_csv(output_dir/f"{'val' if fold=='valid' else fold}.csv", index=False)

if __name__ == '__main__':
    print("Validating paths...")
    validate_cxr()
    print("Preprocessing MIMIC-CXR...")
    preprocess_mimic()
    print("Preprocessing CheXpert...")
    preprocess_cxp()
    print("Preprocessing ChestX-ray8...")
    preprocess_nih()