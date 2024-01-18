import dask.dataframe as dd
import os
import pandas as pd
from pathlib import Path
from rdkit.Chem import Descriptors

from rdkit import Chem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def num_mols_over_treshold(df, t=.9):
    filter_ = df["tanimoto"] >= t
    filtered = df[filter_]
    return t, len(filtered)


def compute_distance_over_join(joindf, gen_name, ref_name):
    # ----- Build output ----- #
    gen_smiles_colname = "SMILES_"+str(gen_name)
    ref_smiles_colname = "SMILES_"+str(ref_name)
    gen_weight_colname = "WEIGHT_"+str(gen_name)
    ref_weight_colname = "WEIGHT_"+str(ref_name)

    smiles_gen = []
    NUM_ATOMS = []
    NUM_BONDS = []
    WEIGHT_gen = []
    SMILES_ref = []
    WEIGHT_ref = []
    tanimoto = []

    for i, (smi, df) in enumerate(joindf.groupby(gen_smiles_colname)):

        smiles_gen.extend(df[gen_smiles_colname])
        NUM_ATOMS.extend(df["NUM_ATOMS"])
        NUM_BONDS.extend(df["NUM_BONDS"])
        WEIGHT_gen.extend(df[gen_weight_colname])
        SMILES_ref.extend(df[ref_smiles_colname])
        WEIGHT_ref.extend(df[ref_weight_colname])

        # compute dists
        ref_mol_matched_fps = [Chem.RDKFingerprint(
            Chem.MolFromSmiles(smi)) for smi in df[ref_smiles_colname]]
        tanimoto.extend(DataStructs.BulkTanimotoSimilarity(
            Chem.RDKFingerprint(Chem.MolFromSmiles(smi)), ref_mol_matched_fps))

    return pd.DataFrame(
        {
            gen_smiles_colname: smiles_gen,
            'NUM_ATOMS': NUM_ATOMS,
            'NUM_BONDS': NUM_BONDS,
            gen_weight_colname: WEIGHT_gen,
            ref_smiles_colname: SMILES_ref,
            ref_weight_colname: WEIGHT_ref,
            "tanimoto": tanimoto
        })


def compute_join(gen_df, ref_df, gen_name, ref_name, columns=["NUM_ATOMS", "NUM_BONDS"]):
    '''
    Same of: 
    inner join on atom AND bond exact match
    out = pd.merge(gen_df, ref_df, on=["NUM_ATOMS", "NUM_BONDS"], how='inner', suffixes=(
        "_"+str(gen_name), "_"+str(ref_name)), copy=False)
    '''
    gen_df = dd.from_pandas(gen_df, npartitions=1).repartition(
        partition_size="100MB")
    ref_df = dd.from_pandas(ref_df, npartitions=1).repartition(
        partition_size="100MB")
    return dd.merge(gen_df, ref_df, on=columns, how='inner', suffixes=(
        "_"+str(gen_name), "_"+str(ref_name))).compute()


def get_dir(path):
    dir, _ = os.path.split(path)
    return dir


def get_filename(path, ext=True):
    if not ext:
        return get_filename_without_ext(path)
    _, filename = os.path.split(path)
    return filename


def get_filename_without_ext(path):
    _, filename = os.path.split(path)
    return filename.split(".")[0]


def get_ext(path):
    return os.path.splitext(path)[-1].lower()


def extract_smi_props_to_csv_for_large_files(path_to_smiles, name):

    if not name.endswith("_properties"):
        name += "_properties"

    gen = (Chem.MolFromSmiles(smi)
           for smi in read_smiles_from_file(path_to_smiles))

    filename = str(name)+'.csv'
    path_to_file = os.path.join(str(Path(path_to_smiles).parent), filename)

    for i, m in enumerate(gen):
        if (m and validate_rdkit_mol(m)):
            data = {
                'SMILES': Chem.MolToSmiles(m),
                'NUM_ATOMS': m.GetNumAtoms(),
                'NUM_BONDS': m.GetNumBonds(),
                'WEIGHT': Chem.Descriptors.ExactMolWt(m)
            }

            # TODO replace write from 1 line at the time to write as a whole
            df = pd.DataFrame(data, index=[i])
            if i == 0:
                df.to_csv(path_to_file)
            else:
                df.to_csv(path_to_file, mode='a', header=False)


def extract_smi_props_to_csv(path_to_smiles, name):

    if not name.endswith("_properties"):
        name += "_properties"
    mols = mols_from_file(path_to_smiles)
    smiles, atom_nums, bond_num, m_weight = [], [], [], []

    for m in mols:
        smiles.append(Chem.MolToSmiles(m))
        atom_nums.append(m.GetNumAtoms())
        bond_num.append(m.GetNumBonds())
        m_weight.append(Chem.Descriptors.ExactMolWt(m))

    data = {
        'SMILES': smiles,
        'NUM_ATOMS': atom_nums,
        'NUM_BONDS': bond_num,
        'WEIGHT': m_weight,
    }

    df = pd.DataFrame(data)
    filename = str(name) + '_properties.csv'
    path_to_file = os.path.join(str(Path(path_to_smiles).parent), filename)
    df.to_csv(path_to_file)


def mols_from_file(pathfile, drop_none=False):
    filename_ext = os.path.splitext(pathfile)[-1].lower()
    if filename_ext in ['.sdf']:
        suppl = Chem.SDMolSupplier(pathfile)
    elif filename_ext in ['.csv', '.txt', '.smiles']:
        suppl = Chem.SmilesMolSupplier(pathfile)
    else:
        raise TypeError(f"{filename_ext} not supported")
    if drop_none:
        return [x for x in suppl if x is not None]
    return [x for x in suppl]


def read_smiles_from_file(pathfile):
    with open(pathfile) as file:
        return [line.rstrip() for line in file]


def smi2mols(smiles):
    return [Chem.MolFromSmiles(smi) for smi in smiles]


def mols2smi(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]


def keep_valid_mols(mols):
    return [m for m in mols if (m and validate_rdkit_mol(m))]


def validate_rdkit_mol(mol):
    """
    Sanitizes an RDKit molecules and returns True if the molecule is chemically
    valid.
    :param mol: an RDKit molecule 
    :return: True if the molecule is chemically valid, False otherwise
    """
    if Chem is None:
        raise ImportError('`validate_rdkit_mol` requires RDkit.')
    if len(Chem.GetMolFrags(mol)) > 1:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except ValueError:
        return False


def generate_file(path, filename):
    '''
    if path does not exist it is created 
    if filename must have extension, default: '.txt'
    if file already exists it is overwritten

    args:
        - path directory where to save smiles list as txt 
        - filename name of the file. By default it is created a txt file
    '''
    os.makedirs(path, exist_ok=True)
    path_to_file = os.path.join(path, filename)
    filename_ext = os.path.splitext(path_to_file)[-1].lower()
    if not filename_ext:
        path_to_file += '.txt'
    if os.path.isfile(path_to_file):
        try:
            os.remove(path_to_file)
        except OSError:
            raise f"{path_to_file} already existing and could not be removed"
    return path_to_file


def save_smiles(smiles, path, filename, ext='.txt'):
    '''
    saves smiles in a file at path
    extension can be provided in filename or as separate arg
    args:
        - smiles str iterable 
        - path directory where to save smiles 
        - filename name of the file, must not have extension
    '''
    path_to_file = os.path.join(path, filename)
    filename_ext = os.path.splitext(path_to_file)[-1].lower()
    if not filename_ext:
        if ext not in ['.txt', '.smiles']:
            raise f"extension {ext} not valid"
        path_to_file += ext

    path_to_file = generate_file(path, filename)
    with open(path_to_file, "w+") as f:
        f.writelines("%s\n" % smi for smi in smiles)


def drop_duplicates_with_openbabel(in_file, out_file):
    cmd = f"obabel -ismiles {in_file} -osmiles -O{out_file} --unique"
    os.system(cmd)  # synchronous call, the result is waited


def create_log(path, name="log.txt"):
    if not name.endswith(".txt"):
        name += ".txt"
    generate_file(path, name)
    return os.path.join(path, name)


def append_line_to_log(path_to_log, line):
    with open(path_to_log, "a") as log:
        log.write(line + "\n")


def delete_file(path_to_file):
    if os.path.isfile(path_to_file):
        try:
            os.remove(path_to_file)
        except OSError:
            raise f"{path_to_file} already existing and could not be removed"
