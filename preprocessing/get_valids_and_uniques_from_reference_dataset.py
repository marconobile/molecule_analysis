'''
takes any sdf file as input
perform preprocessing steps on file pointed at path, while logging the counts during the procedures
step1) keep_valid_mols
step2) remove duplicates
step3) save file with remained smiles
'''

import os


import sys
sys.path.append("..")

from utils.data_utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
path = args.path
# example path: "/home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/Maybridge_HitDiscover.sdf"

dir, filename = os.path.split(path)
name = filename.split(".")[0]

log = create_log(dir, name + "_log")

mols = mols_from_file(path)
append_line_to_log(log, f"At read of file: {len(mols)}")

valid_mols = keep_valid_mols(mols)
append_line_to_log(
    log, f"After dropping invalid mols: {len(valid_mols)} #invalids: {len(mols)-len(valid_mols)}")

# use openbabel to remove duplicates from the .smiles file
smiles = mols2smi(valid_mols)
smi_name = name+"_valid.smiles"
save_smiles(smiles, dir, smi_name)
out_pathfile = os.path.join(dir, 'no_dup_' + smi_name)
in_pathfile = os.path.join(dir, smi_name)
drop_duplicates_with_openbabel(
    in_pathfile, out_pathfile)  # in/out are .smiles files

try:
    no_dup_mols = mols_from_file(out_pathfile)
except:
    smi_no_dup_mols = read_smiles_from_file(out_pathfile)
    no_dup_mols = smi2mols(smi_no_dup_mols)

append_line_to_log(
    log, f"After dropping duplicates: {len(no_dup_mols)} #duplicates, {len(valid_mols)-len(no_dup_mols)}")

# remove spurious input.smiles
delete_file(name + "_valid.smiles")
