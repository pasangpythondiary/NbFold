import h5py
import numpy as np
import argparse
from tqdm import tqdm
from os import listdir, remove
import os.path as path
from pathlib import Path
import sys
sys.path.append('/home/sherpa/NbFold')
import nbfold
from nbfold.preprocess import antibody_text_parser as ab_parser
from helper_function_h5 import *

def antibody_to_h5(pdb_dir, out_file_path, fasta_dir=None, overwrite=False, print_progress=False):
    pdb_files = [_ for _ in listdir(pdb_dir) if _.endswith('.pdb')]
    num_seqs = len(pdb_files)

    if fasta_dir is not None:
        seq_info = ab_parser.antibody_db_seq_info(fasta_dir)
        max_h_len = seq_info['max_heavy_seq_len']
    else:
        print('WARNING: No fasta directory given! Defaulting max sequence length for heavy chain to 300')
        max_h_len = 300

    if overwrite and path.isfile(out_file_path):
        remove(out_file_path)
    h5_out = h5py.File(out_file_path, 'w')

    # Create datasets for heavy chain processing
    id_set = h5_out.create_dataset('id', (num_seqs,), compression='lzf', dtype='S25', maxshape=(None,))
    h_len_set = h5_out.create_dataset('heavy_chain_seq_len', (num_seqs,), compression='lzf', dtype='uint16', maxshape=(None,), fillvalue=0)
    h_prim_set = h5_out.create_dataset('heavy_chain_primary', (num_seqs, max_h_len), compression='lzf', dtype='uint8', maxshape=(None, max_h_len), fillvalue=-1)
    h1_set = h5_out.create_dataset('h1_range', (num_seqs, 2), compression='lzf', dtype='uint16', fillvalue=-1)
    h2_set = h5_out.create_dataset('h2_range', (num_seqs, 2), compression='lzf', dtype='uint16', fillvalue=-1)
    h3_set = h5_out.create_dataset('h3_range', (num_seqs, 2), compression='lzf', dtype='uint16', fillvalue=-1)

    # Embeddings and pairwise geometry matrices
    embs = h5_out.create_dataset('embs', (num_seqs, 1024, max_h_len), dtype='float')
    pairwise_geometry_set = h5_out.create_dataset(
        'pairwise_geometry_mat', (num_seqs, 6, max_h_len, max_h_len),
        maxshape=(None, 6, max_h_len, max_h_len),
        compression='lzf', dtype='float', fillvalue=-1
    )

    # Load embeddings
    dataloader_emb = embedding()

    for index, (file, i_m) in tqdm(enumerate(zip(pdb_files, dataloader_emb)), disable=(not print_progress), total=len(pdb_files)):
        ids, matrix = i_m
        m = matrix.reshape(1024, -1)
        numpy_array = m.detach().numpy()
        embs[index] = numpy_array

        id_ = ab_parser.get_id(file)
        pdb_file = str(path.join(pdb_dir, id_ + '.pdb'))
        fasta_file = None if fasta_dir is None else str(path.join(fasta_dir, id_ + '.fasta'))
        info = ab_parser.get_info(pdb_file, fasta_file=fasta_file, verbose=False)

        heavy_prim = info['H']
        total_len = len(heavy_prim)

        id_set[index] = np.string_(id_)
        pairwise_geometry_set[index, :6, :total_len, :total_len] = np.array(info['pairwise_geometry_mat'])

        h_len_set[index] = len(heavy_prim)
        h_prim_set[index, :len(heavy_prim)] = np.array(heavy_prim)

        for h_set, name in [(h1_set, 'h1'), (h2_set, 'h2'), (h3_set, 'h3')]:
            if len(info[name]) == 1:
                info[name] = [info[name][0], info[name][0]]
            if len(info[name]) == 2:
                h_set[index] = np.array(info[name])
            else:
                print(info[name])
                print(f'WARNING: {file} does not have any coordinates for the {name} loop!')

def cli():
    project_path = os.path.abspath(os.path.join(nbfold.__file__, "../.."))
    data_path = os.path.join(project_path, "data/")
    desc = 'Creates h5 files from all the truncated antibody PDB files in a directory'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('pdb_dir', type=str, help='Directory containing PDB files named as ID.pdb')
    parser.add_argument('--out_file', type=str, default=data_path + 'final_antibody.h5', help='Output H5 file path')
    parser.add_argument('--fasta_dir', type=str, default=None, help='Directory containing FASTA files named as ID.fasta')
    parser.add_argument('--overwrite', action="store_true", help='Overwrite existing file if it exists', default=False)

    args = parser.parse_args()
    antibody_to_h5(args.pdb_dir, args.out_file, fasta_dir=args.fasta_dir, overwrite=args.overwrite, print_progress=True)

if __name__ == '__main__':
    cli()
