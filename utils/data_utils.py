import dask.dataframe as dd
import os
import pandas as pd
from pathlib import Path
from itertools import islice
from rdkit import Chem
import time
from rdkit import RDLogger
import argparse
from multiprocessing import Pool
RDLogger.DisableLog('rdApp.*')


def argparser(*args):
    ''' 
    *args: ordered list of strings taken from cmd line
    usage: call gettatr from the return of this func to get associated string
    '''
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument(arg)
    return parser.parse_args()


def batched(iterable, n):
    ''' Given an iterable creates a generator that return batches of n elements of iterable'''
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def separate_tuples(tuples_list):
    '''Given a list of bidimensional tuples returns 2 lists
    each list is the list of the first els of the input list 
    and the second list is the list of the second elements of the tuples
    '''
    first_elements = [item[0] for item in tuples_list]
    second_elements = [item[1] for item in tuples_list]
    return first_elements, second_elements


def num_mols_over_treshold(df, t=.9):
    ''' given a pandas df with a "tanimoto" column, select all mols that have a value of tanimoto over t'''
    filter_ = df["tanimoto"] >= t
    filtered = df[filter_]
    return t, len(filtered)


def flatten_list_of_lists(l):
    '''l is a list of lists that have to be flattened to get a list of els'''
    return [el for list_ in l for el in list_]


def apply_f_parallelized_batched(f, iterable_, nels_per_split, ncores):
    ''' 
    given an iterable_, applies f to a batch of iterable_ of size nels_per_split
    the batches are treated in parallel over ncores
    f must be appliable over a list of elements
    '''
    it = batched(iterable_, nels_per_split)
    with Pool(processes=ncores) as P:
        out = P.map(f, it)
    return flatten_list_of_lists(out)


def get_dir(path):
    '''
    given a path/to/file.ext
    returns path/to/
    '''
    dir, _ = os.path.split(path)
    return dir


def get_filename(path, ext=True):
    '''
    given a path/to/file.ext
    returns file.ext if ext, just file otherwise
    '''
    _, filename = os.path.split(path)
    if not ext:
        return filename.split(".")[0]
    return filename


def get_ext(path):
    '''given a path/to/file.ext
    returns extension'''
    return os.path.splitext(path)[-1].lower()


# def extract_smi_props_to_csv(path_to_smiles, name):

#     if not name.endswith("_properties"):
#         name += "_properties"
#     mols = mols_from_file(path_to_smiles)
#     smiles, atom_nums, bond_num, m_weight = [], [], [], []

#     for m in mols:
#         smiles.append(Chem.MolToSmiles(m))
#         atom_nums.append(m.GetNumAtoms())
#         bond_num.append(m.GetNumBonds())
#         m_weight.append(Chem.Descriptors.ExactMolWt(m))

#     data = {
#         'SMILES': smiles,
#         'NUM_ATOMS': atom_nums,
#         'NUM_BONDS': bond_num,
#         'WEIGHT': m_weight,
#     }

#     df = pd.DataFrame(data)
#     filename = str(name) + '_properties.csv'
#     path_to_file = os.path.join(str(Path(path_to_smiles).parent), filename)
#     df.to_csv(path_to_file)


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


def smi2mols(smiles): return [Chem.MolFromSmiles(smi) for smi in smiles]
def mols2smi(mols): return [Chem.MolToSmiles(mol) for mol in mols]


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


class TimeCode:
    '''
    Utility class that times code execution. 
    Usage: instanciate it when code starts 
    call compute compute_ellapsed when you want to evaluate passed time
    '''

    def __init__(self):
        self.start_timer()

    def start_timer(self):
        self.start = time.time()

    def compute_ellapsed(self):
        self.end = time.time()
        return self.end - self.start  # in seconds


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

    cmd = f"touch {path_to_file};"
    os.system(cmd)
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
    '''
    Given an input .smiles file, generate an out_file.smiles with no duplicates or invalid mols
    '''
    cmd = f"obabel -ismiles {in_file} -osmiles -O{out_file} --unique"
    os.system(cmd)  # synchronous call, the result is waited


def create_log(path=".", name="log.txt"):
    if not name.endswith(".txt"):
        name += ".txt"
    generate_file(path, name)
    return os.path.join(path, name)


def append_line_to_log(path_to_log, line):
    with open(path_to_log, "a") as log:    
        if isinstance(line, list):
            for w in line:
                log.write(str(w) + "\n")
        else:
            log.write(str(line) + "\n")


def delete_file(path_to_file):
    if os.path.isfile(path_to_file):
        try:
            os.remove(path_to_file)
        except OSError:
            raise f"{path_to_file} already existing and could not be removed"


def are_mols_equal(m1, m2):
    '''
    evaluate if 2 mols (or smiles) refer to the same underlying mol
    '''
    if isinstance(m1, str):
        m1 = Chem.MolFromSmiles(m1)
    if isinstance(m2, str):
        m2 = Chem.MolFromSmiles(m2)
    return m1.HasSubstructMatch(m2) and m2.HasSubstructMatch(m1)


def get_all_ext_files_in_path(path, ext):
    file_list = []    
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                file_list.append(os.path.join(root, file))
    return file_list