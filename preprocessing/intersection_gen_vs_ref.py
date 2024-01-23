'''
python intersection_gen_vs_ref.py data/generated_smiles/moses/aae/no_dup_aae_all_valid.smiles data/synthesized_datasets/asinex/no_dup_asinex_03_Feb_2022_valid.smiles
'''


from rdkit import RDLogger
import warnings
from multiprocessing import Pool
# import multiprocessing
from rdkit import DataStructs
from rdkit import Chem
import pandas as pd
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

try:
    from utils.data_utils import create_log, append_line_to_log, batched, argparser, read_smiles_from_file, get_filename, TimeCode, apply_f_parallelized_batched
except:
    import sys
    sys.path.append("..")
    from utils.data_utils import create_log, append_line_to_log, batched, argparser, read_smiles_from_file, get_filename, TimeCode, apply_f_parallelized_batched


def flatten_list_of_lists(l): return [el for list_ in l for el in list_]


def to_pd(out, name):
    '''
    out: list of MolStructs
    name: str name of the technique/df 
    '''
    smiles = []
    n_atoms = []
    fps = []
    for el in out:
        smiles.append(el.smile)
        n_atoms.append(el.n_atoms)
        fps.append(el.fp)
    return pd.DataFrame({
        f"SMILES_{name}": smiles,
        f"NUM_ATOMS_{name}": n_atoms,
        "FPs": fps
    })


class MolStruct:
    def __init__(self, smile):
        self.smile = smile
        self.mol = Chem.MolFromSmiles(smile)
        self.fp = Chem.RDKFingerprint(self.mol)
        self.n_atoms = self.mol.GetNumAtoms()


def batch_ctors(l): return [MolStruct(smi) for smi in l]


def compute_exact_matches(molstruct, ref_df):

    filter_ = ref_df[f"NUM_ATOMS_{ref_name}"] == molstruct.n_atoms
    possible_matches = ref_df[filter_]
    possible_matches["tanimoto"] = DataStructs.BulkTanimotoSimilarity(
        molstruct.fp, possible_matches["FPs"].tolist())
    filter_ = possible_matches[f"tanimoto"] == 1
    exact_matches = possible_matches[filter_]
    return exact_matches


# =======================================================================

args = argparser("gen_smiles", "ref_smiles")

gen_smiles = args.gen_smiles
ref_smiles = args.ref_smiles
num_cores = 10  # multiprocessing.cpu_count()
nels_per_split = 5000

print("Start reading smiles from files...")
t = TimeCode()
gen_name = get_filename(gen_smiles, ext=False)
ref_name = get_filename(ref_smiles, ext=False)
log = create_log(name=f"{gen_name}_vs_{ref_name}.txt")

# get smiles from files
gen_smiles = read_smiles_from_file(gen_smiles)
ref_smiles = read_smiles_from_file(ref_smiles)
print(
    f"End reading smiles from files, duration: {t.compute_ellapsed()} seconds")

print("Start computing fingerprints...")
t = TimeCode()
# apply MolStruct ctors to each smiles, in batches for performance
gen_out = apply_f_parallelized_batched(
    batch_ctors, gen_smiles, nels_per_split, num_cores)

# do the same for ref
ref_out = apply_f_parallelized_batched(
    batch_ctors, ref_smiles, nels_per_split, num_cores)
print(f"End computing fingerprints, duration: {t.compute_ellapsed()} seconds")

print("Start generating df...")
t = TimeCode()
# put moses in a pandas df with 3 cols: SMILES_moses, NUM_ATOMS_moses, FPs
df = to_pd(ref_out, ref_name)
print(f"End generating df..., duration: {t.compute_ellapsed()} seconds")

print("Start writing....")
t = TimeCode()
for out in gen_out:
    out_df = compute_exact_matches(out, df)
    if out_df.empty:
        continue
    append_line_to_log(
        log, f"{gen_name} smi: {out.smile}, {ref_name} smi: {out_df[f'SMILES_{ref_name}'].tolist()[0]}, tanimoto: {out_df['tanimoto'].tolist()[0]}, natoms: {out_df[f'NUM_ATOMS_{ref_name}'].tolist()[0]}")
print(f"End writing, duration: {t.compute_ellapsed()} seconds")
