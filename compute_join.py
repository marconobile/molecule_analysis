'''
Takes 2 _properties.csv and joins using an inner join them matching over desired equal properties
example: python compute_join_td.py /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv

'''

from data_utils import *
import argparse
import numpy as np
from pathlib import Path
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import dask.dataframe as dd
RDLogger.DisableLog('rdApp.*')

# ----- Parser ----- #
parser = argparse.ArgumentParser()
parser.add_argument("reference")
# path to csv file containing the properties of reference db eg: /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv
parser.add_argument("generated")
# path to csv file containing the properties of generated db eg: /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv
args = parser.parse_args()


# ----- Processing paths ----- #
# reference
reference_path = args.reference
ref_df = pd.read_csv(reference_path)
ref_df = ref_df.drop(ref_df.columns[0], axis=1)
ref_dir, _ = os.path.split(Path(reference_path))
_, ref_name = os.path.split(Path(reference_path).parent)

# generated
generated_path = args.generated
gen_df = pd.read_csv(generated_path)
gen_df = gen_df.drop(gen_df.columns[0], axis=1)
_, gen_name = os.path.split(Path(generated_path).parent)

outpath = f"/storage_common/nobilm/data_comparisons/data/generated_smiles/moses/{gen_name}/{gen_name}_vs_{ref_name}.csv"

# ----- Compute inner join ----- #
out = compute_join(gen_df, ref_df, gen_name, ref_name)
out_df = compute_distance_over_join(out, gen_name, ref_name)
out_df.to_csv(outpath, index=False)
