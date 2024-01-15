'''
given a .smiles file it generates a no_dup_{ref}_valid_properties.csv
the .smiles must have already been preprocessed s.t. make it do not contain invalid or duplicate mols

creates a:
no_dup_{ref}_valid_properties.csv
such as:
Maybridge_HitDiscover
creating a file as:
no_dup_Maybridge_HitDiscover_valid_properties.csv
used to create the method_vs_ref.csv dataset

Example:
path to no_dup_..._valid.smiles file
cmd: python build_property_df.py /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid.smiles
'''

import argparse
from data_utils import get_filename, extract_smi_props_to_csv_for_large_files

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
args = parser.parse_args()
filepath = args.filepath

name = get_filename(filepath, ext=False)
extract_smi_props_to_csv_for_large_files(filepath, name)