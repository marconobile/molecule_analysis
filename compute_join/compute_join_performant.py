'''
!Attention this script repeatedly cleans /tmp dir to avoid no mem errors
Computes join between 2 _properties.csv in a performant way
'''
try:
    from utils.data_utils import compute_join, get_filename
except:
    import sys
    sys.path.append("..")
    from utils.data_utils import compute_join, get_filename

import numpy as np
from pathlib import Path
import argparse
import dask.dataframe as dd
from rdkit import Chem, DataStructs
import pandas as pd
import os


parser = argparse.ArgumentParser()
parser.add_argument("generated")
# path to csv file containing the properties of generated db eg: /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv
parser.add_argument("reference")
# path to csv file containing the properties of reference db eg: /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv
parser.add_argument("output_folder")
args = parser.parse_args()

gen_path, ref_path, out_folder = Path(args.generated), Path(
    args.reference), Path(args.output_folder)
gen_df = pd.read_csv(gen_path)
ref_df = pd.read_csv(ref_path)
gen_name = get_filename(gen_path.parent)
ref_name = get_filename(ref_path.parent)
nsplits = 50
ref_df_splits = np.array_split(ref_df, nsplits)

gen_smiles_colname = "SMILES_"+str(gen_name)
ref_smiles_colname = "SMILES_"+str(ref_name)
gen_weight_colname = "WEIGHT_"+str(gen_name)
ref_weight_colname = "WEIGHT_"+str(ref_name)


def custom_func(group):
    # https://stackoverflow.com/questions/60721290/how-to-apply-a-custom-function-to-groups-in-a-dask-dataframe-using-multiple-col
    smi_gen = group[gen_smiles_colname].iloc[[0]].values[0]
    mol_gen = Chem.MolFromSmiles(smi_gen)
    if not mol_gen:
        return

    pd.DataFrame({
        gen_smiles_colname: group[gen_smiles_colname].tolist(),
        'NUM_ATOMS': group[f"NUM_ATOMS"].tolist(),
        'NUM_BONDS': group[f"NUM_BONDS"].tolist(),
        gen_weight_colname: group[gen_weight_colname].tolist(),
        ref_smiles_colname: group[ref_smiles_colname].tolist(),
        ref_weight_colname: group[ref_weight_colname].tolist(),
        "tanimoto": DataStructs.BulkTanimotoSimilarity(
            Chem.RDKFingerprint(mol_gen),  [Chem.RDKFingerprint(Chem.MolFromSmiles(
                smi)) for smi in group[ref_smiles_colname]])
    }).to_csv(out_folder/(smi_gen+".csv"))


for split in ref_df_splits:
    df = compute_join(gen_df, split, gen_name, ref_name)
    df = dd.from_pandas(df, npartitions=1).repartition(
        partition_size="100MB")
    df = df.groupby(f"SMILES_{gen_name}")
    df.apply(custom_func).compute()
    # TODO insert check if tmp folder exists
    cmd = "cd; cd /tmp; find /tmp -user nobilm@usi.ch -exec rm -r {} +;"
    os.system(cmd)
