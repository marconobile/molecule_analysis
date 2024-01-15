'''
to be executed in moses venv
gets moses training data and writes all its smiles to txt file
'''

from data_utils import save_smiles
import os, moses

save_smiles([smi for smi in moses.get_dataset('train')], '.', "moses.txt")
