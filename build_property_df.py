import argparse
from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("filepath")
# path to no_dup_..._valid.smiles file
# example: python build_property_df.py /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid.smiles
args = parser.parse_args()
filepath = args.filepath

name = get_filename(filepath, ext=False)
extract_smi_props_to_csv_for_large_files(filepath, name)