from rdkit import RDLogger
from rdkit.Chem import Descriptors  # required
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
import multiprocessing
from rdkit import DataStructs
from rdkit import Chem
import os
import pandas as pd
RDLogger.DisableLog('rdApp.*')

try:
    from utils.data_utils import read_smiles_from_file, argparser, get_dir, get_filename, apply_f_parallelized_batched, create_log, append_line_to_log
except:
    import sys
    sys.path.append("..")
    from utils.data_utils import read_smiles_from_file, argparser, get_dir, get_filename, apply_f_parallelized_batched, create_log, append_line_to_log


# open moses and do full analysis then repeat em for other dataset
# so get path
# and then sequence of funcs to analyze data
# analysis natom moses in general ie. the dist/hist of natoms/molweights of moses

# studio statistico degli exact matches wrt moses properties
# 	analisi delle frequenze dei caratteri degli smiles

# idea: studio statistico completo degli exact matches wrt moses properties x capire quando gli exact matches occur
# -> this to introduce the study of the clustering
# -> plot every hist pair one on top of the other -> find properties for which exact matches occurr often


class MolStruct:
    def __init__(self, smile):
        self.smile = smile
        self.mol = Chem.MolFromSmiles(smile)
        self.descriptors = Chem.Descriptors.CalcMolDescriptors(self.mol)
        self.descriptors["NAtoms"] = self.mol.GetNumAtoms()
        self.descriptors["NBonds"] = len(list(self.mol.GetBonds()))


def f_to_list(l, f=MolStruct): return [f(el) for el in l]


# num_cores = multiprocessing.cpu_count()
# best params
# num_cores = multiprocessing.cpu_count() // 10
# nels_per_split = 6000

# =========================================================

args = argparser("path")
path = args.path
dir = Path(get_dir(path))
name = get_filename(dir, False)
outfile = os.path.join(dir, name) + "_mol_descriptors.csv"

gen_smiles = read_smiles_from_file(path)
out = apply_f_parallelized_batched(f_to_list, gen_smiles, 5000, 2)
df = pd.DataFrame([el.descriptors for el in out])
df.to_csv(outfile, index=False)
# do plots for all cols that are not 0s

log = create_log(dir, f"{name}_char_occurrences.txt")
chars = sorted(list(set(''.join(gen_smiles))))  # dictionary
append_line_to_log(log, "Chars dict:\n")
append_line_to_log(log, chars)
append_line_to_log(log, "\n")

# count occurreces of chars in text
# List of strings

# Create an empty dictionary to store character counts
char_counts = {}

# Loop through each string in the list
for string in gen_smiles:
    # Loop through each character in the string
    for char in string:
        # Update the count in the dictionary
        char_counts[char] = char_counts.get(char, 0) + 1

# Print the character counts
append_line_to_log(log, "Counts\n:")
for char, count in char_counts.items():
    append_line_to_log(log, f"Character '{char}' occurs {count} times\n")
