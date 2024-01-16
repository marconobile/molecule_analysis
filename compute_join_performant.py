import os
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from threading import Lock
import dask.dataframe as dd
from data_utils import compute_join, get_filename
import argparse
from pathlib import Path

mutex = Lock()


class Storage:
    def __init__(self):
        self.group_id = 0
        self.data = {}

    def init_entry(self, group_id):
        self.data[group_id] = {
            "smiles_gen": [],
            "NUM_ATOMS": [],
            "NUM_BONDS": [],
            "WEIGHT_gen": [],
            "SMILES_ref": [],
            "WEIGHT_ref": [],
            "tanimoto": []
        }

    def to_csv(self, pathtofile):
        smiles_gen, NUM_ATOMS, NUM_BONDS, WEIGHT_gen, SMILES_ref, WEIGHT_ref, tanimoto = [
        ], [], [], [], [], [], []
        for dict_ in self.data.values():
            smiles_gen.extend(dict_["smiles_gen"])
            NUM_ATOMS.extend(dict_["NUM_ATOMS"])
            NUM_BONDS.extend(dict_["NUM_BONDS"])
            WEIGHT_gen.extend(dict_["WEIGHT_gen"])
            SMILES_ref.extend(dict_["SMILES_ref"])
            WEIGHT_ref.extend(dict_["WEIGHT_ref"])
            tanimoto.extend(dict_["tanimoto"])

        df_to_write = pd.DataFrame({"smiles_gen": smiles_gen, "NUM_ATOMS": NUM_ATOMS, "NUM_BONDS": NUM_BONDS, "WEIGHT_gen":
                                    WEIGHT_gen, "SMILES_ref": SMILES_ref, "WEIGHT_ref": WEIGHT_ref, "tanimoto": tanimoto})
        df_to_write.to_csv(pathtofile)


def custom_func(group, storage, gen_name, ref_name):
    # https://stackoverflow.com/questions/60721290/how-to-apply-a-custom-function-to-groups-in-a-dask-dataframe-using-multiple-col

    smi_gen = str(group[f"SMILES_{gen_name}"].iloc[[0]].values[0])
    if not Chem.MolFromSmiles(smi_gen):
        return

    smi_ref_colname = f"SMILES_{ref_name}"

    ref_mol_matched_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(
        smi)) for smi in group[smi_ref_colname]]
    tanimotos = DataStructs.BulkTanimotoSimilarity(
        Chem.RDKFingerprint(Chem.MolFromSmiles(smi_gen)), ref_mol_matched_fps)

    with mutex:
        storage.init_entry(storage.group_id)
        storage.data[storage.group_id]["smiles_gen"] = (
            group[f"SMILES_{gen_name}"]).tolist()
        storage.data[storage.group_id]["NUM_ATOMS"] = (
            group["NUM_ATOMS"]).tolist()
        storage.data[storage.group_id]["NUM_BONDS"] = (
            group["NUM_BONDS"]).tolist()
        storage.data[storage.group_id]["WEIGHT_gen"] = (
            group[f"WEIGHT_{gen_name}"]).tolist()
        storage.data[storage.group_id]["SMILES_ref"] = (
            group[smi_ref_colname]).tolist()
        storage.data[storage.group_id]["WEIGHT_ref"] = (
            group[f"WEIGHT_{ref_name}"]).tolist()
        storage.data[storage.group_id]["tanimoto"] = tanimotos
        storage.group_id += 1


def main(gen_path, ref_path):

    gen_df = pd.read_csv(gen_path)
    ref_df = pd.read_csv(ref_path)

    gen_name = get_filename(gen_path.parent)
    ref_name = get_filename(ref_path.parent)
    df = compute_join(gen_df, ref_df, gen_name, ref_name)

    # store_path = "/storage_common/nobilm/{gen_name}_vs_intersection_join.csv"
    # df = pd.read_csv(store_path)

    df = dd.from_pandas(df, npartitions=1).repartition(partition_size="100MB")
    gb = df.groupby(f"SMILES_{gen_name}")
    storage = Storage()
    gb.apply(custom_func, storage, gen_name, ref_name).compute()

    outfile = Path("/storage_common/nobilm/data_comparisons/data/generated_smiles/moses/") / \
        gen_name/f"{gen_name}_vs_{ref_name}.csv"
    storage.to_csv(outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated")
    # path to csv file containing the properties of generated db eg: /home/marconobile/Desktop/wd/data/generated_smiles/moses/aae/aae_properties.csv
    parser.add_argument("reference")
    # path to csv file containing the properties of reference db eg: /home/marconobile/Desktop/wd/data/synthesized_datasets/maybridge/no_dup_Maybridge_HitDiscover_valid_properties.csv
    args = parser.parse_args()
    main(Path(args.generated), Path(args.reference))
