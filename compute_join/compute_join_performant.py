'''
!Attention this script repeatedly cleans /tmp dir to avoid no mem errors

Computes join between 2 _properties.csv in a performant way


python compute_join_performant.py /wd/data/generated_smiles/moses/aae/aae_properties.csv /wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv

'''
import sys
sys.path.append("..")
import os
import pandas as pd
from rdkit import Chem, DataStructs
import dask.dataframe as dd
from utils.data_utils import compute_join, get_filename
import argparse
from pathlib import Path
import numpy as np


def custom_func(group, gen_name, ref_name, output_folder):
    # https://stackoverflow.com/questions/60721290/how-to-apply-a-custom-function-to-groups-in-a-dask-dataframe-using-multiple-col

    smi_gen = str(group[f"SMILES_{gen_name}"].iloc[[0]].values[0])
    mol_gen = Chem.MolFromSmiles(smi_gen)
    if not mol_gen:
        return

    smi_ref_colname = f"SMILES_{ref_name}"
    ref_mol_matched_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(
        smi)) for smi in group[smi_ref_colname]]
    tanimotos = DataStructs.BulkTanimotoSimilarity(
        Chem.RDKFingerprint(mol_gen), ref_mol_matched_fps)

    gen_smiles_colname = "SMILES_"+str(gen_name)
    ref_smiles_colname = "SMILES_"+str(ref_name)
    gen_weight_colname = "WEIGHT_"+str(gen_name)
    ref_weight_colname = "WEIGHT_"+str(ref_name)

    df = pd.DataFrame({
        gen_smiles_colname:  group[gen_smiles_colname].tolist(),
        'NUM_ATOMS':  group[f"NUM_ATOMS"].tolist(),
        'NUM_BONDS':  group[f"NUM_BONDS"].tolist(),
        gen_weight_colname: group[gen_weight_colname].tolist(),
        ref_smiles_colname: group[ref_smiles_colname].tolist(),
        ref_weight_colname: group[ref_weight_colname].tolist(),
        "tanimoto": tanimotos
    })
    df.to_csv(output_folder/(smi_gen+".csv"))


def main(gen_path, ref_path, out_folder):

    gen_df = pd.read_csv(gen_path)
    ref_df = pd.read_csv(ref_path)

    gen_name = get_filename(gen_path.parent)
    ref_name = get_filename(ref_path.parent)

    nsplits = 50
    ref_df_splits = np.array_split(ref_df, nsplits)

    for split in ref_df_splits:
        df = compute_join(gen_df, split, gen_name, ref_name)
        df = dd.from_pandas(df, npartitions=1).repartition(
            partition_size="100MB")
        df = df.groupby(f"SMILES_{gen_name}")
        df.apply(custom_func, gen_name, ref_name, out_folder).compute()

        # TODO insert check if tmp folder exists
        cmd = "cd; cd /tmp; find /tmp -user nobilm@usi.ch -exec rm -r {} +;"
        os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated")
    # path to csv file containing the properties of generated db eg: /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv
    parser.add_argument("reference")
    # path to csv file containing the properties of reference db eg: /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv
    parser.add_argument("output_folder")
    args = parser.parse_args()
    main(Path(args.generated), Path(args.reference), Path(args.output_folder))
