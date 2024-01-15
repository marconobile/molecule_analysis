from data_utils import *
import os

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_filepaths():
    '''
    returns a dictionary of the form: k:model_name, v:{
        "root": path/to/csv
        "filename": name.csv
        "filepath": path/to/name.csv
        }
    '''
    cwd = os.getcwd()
    ext = ".csv"
    models = {}
    for root, _, files in os.walk(cwd):
        for file in files:
            if file.endswith("_all"+ext):
                models[os.path.basename(os.path.normpath(root))] = {'root': root,
                                                                    'filename': file,
                                                                    'filepath': os.path.join(root, file)}
    return models


def main():
    models = get_filepaths()
    for v in models.values():
        # create log file for current job
        log = create_log(v['root'])

        # load mols from file, purge from invalid, write smiles to .smiles file for obabel
        mols = mols_from_file(v['filepath'])
        append_line_to_log(log, f"At read of file: {len(mols)}")
        valid_mols = keep_valid_mols(mols)
        append_line_to_log(
            log, f"After dropping invalid mols: {len(valid_mols)} #invalids: {len(mols)-len(valid_mols)}")
        smiles = mols2smi(valid_mols)
        name = v["filename"].split(".")[0] + "_valid.smiles"
        save_smiles(smiles, v['root'], name)

        # use openbabel to remove duplicates from the .smiles file
        out_pathfile = os.path.join(v['root'], 'no_dup_'+name)
        drop_duplicates_with_openbabel(os.path.join(
            v['root'], name), out_pathfile)  # in/out are .smiles files
        no_dup_mols = mols_from_file(out_pathfile)
        append_line_to_log(
            log, f"After dropping duplicates: {len(no_dup_mols)} #duplicates, {len(valid_mols)-len(no_dup_mols)}")

        # remove spurious input.smiles
        delete_file(name)


if __name__ == '__main__':
    main()
