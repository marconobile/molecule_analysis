import pandas as pd
import sys
sys.path.append("..")
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')
from rdkit import DataStructs
import multiprocessing
from pathlib import Path
from multiprocessing import Pool
from itertools import islice

from utils.data_utils import read_smiles_from_file, argparser, get_dir, get_filename, apply_f_parallelized_batched


# open moses and do full analysis then repeat em for other dataset
# so get path
# and then sequence of funcs to analyze data
# analysis natom moses in general ie. the dist/hist of natoms/molweights of moses

# studio statistico degli exact matches wrt moses properties
# 	analisi delle frequenze dei caratteri degli smiles

# idea: studio statistico completo degli exact matches wrt moses properties x capire quando gli exact matches occur 
# -> this to introduce the study of the clustering  
# -> plot every hist pair one on top of the other -> find properties for which exact matches occurr often


args = argparser("path")
path = args.path

dit = get_dir(path)
name = get_filename(path, False)


gen_smiles = read_smiles_from_file(path)

class MolStruct:
    def __init__(self, smile):
        self.smile = smile
        self.mol = Chem.MolFromSmiles(smile)        
        self.descriptors = Chem.Descriptors.CalcMolDescriptors(self.mol)   
        self.descriptors["NAtoms"] = self.mol.GetNumAtoms()
        self.descriptors["NBonds"] = len(list(self.mol.GetBonds()))


# num_cores = multiprocessing.cpu_count()  
        
# best params 
# num_cores = multiprocessing.cpu_count() // 10
# nels_per_split = 6000

def f_to_list(l, f=MolStruct): return [f(el) for el in l]
out = apply_f_parallelized_batched(f_to_list, gen_smiles, 5000, 2)

data = [el.descriptors for el in out]
df = pd.DataFrame(data)
# do plots for all cols that are not 0s
df.to_csv(os.path.join(dir, f"/{name}.csv"), index=False)

chars = sorted(list(set(''.join(gen_smiles)))) # dictionary
print("chars ", chars)

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
for char, count in char_counts.items():
    print(f"Character '{char}' occurs {count} times.")
