import numpy as np
import pandas as pd
from os import listdir
from os.path import join
from os.path import basename, splitext
from Bio import SeqIO
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser

from nbfold.util.pdb import protein_pairwise_geometry_matrix
from nbfold.util.util import _aa_dict, letter_to_num


def get_id(pdb_file_path):
    return splitext(basename(pdb_file_path))[0]


def get_pdb_atoms(pdb_file_path):
    """Returns a list of the atom coordinates, and their properties in a pdb file."""
    with open(pdb_file_path, 'r') as f:
        lines = [line for line in f.readlines() if 'ATOM' in line]
    column_names = [
        'atom_num', 'atom_name', 'alternate_location_indicator',
        'residue_name', 'chain_id', 'residue_num',
        'code_for_insertions_of_residues', 'x', 'y', 'z', 'occupancy',
        'temperature_factor', 'segment_identifier', 'element_symbol'
    ]
    column_ends = np.array(
        [3, 10, 15, 16, 19, 21, 25, 26, 37, 45, 53, 59, 65, 75, 77])
    column_starts = column_ends[:-1] + 1
    column_ends = column_ends[1:]  

    rows = [[
        l[start:end + 1].replace(' ', '')
        for start, end in zip(column_starts, column_ends)
    ] for l in lines]
    return pd.DataFrame(rows, columns=column_names)


def antibody_db_seq_info(fasta_dir):
    fasta_files = [
        join(fasta_dir, _) for _ in listdir(fasta_dir) if _[-5:] == 'fasta'
    ]

    num_seqs = len(fasta_files)
    min_heavy_seq_len = min_total_seq_len = float('inf')
    max_heavy_seq_len = max_total_seq_len = -float('inf')

    for fasta_file in fasta_files:
        chains = list(SeqIO.parse(fasta_file, 'fasta'))
        if len(chains) != 1:
            msg = 'Expected 1 chain (H) in {}, got {}'.format(
                fasta_file, len(chains))
            raise ValueError(msg)

        for chain in chains:
            if ':H' in chain.id or 'heavy' in chain.id:
                h_len = len(chain.seq)
                max_heavy_seq_len = max(h_len, max_heavy_seq_len)
                min_heavy_seq_len = min(h_len, min_heavy_seq_len)
            else:
                raise ValueError(
                    '{} does not have >name:chain (H) format'.format(
                        fasta_file))

    return dict(num_seqs=num_seqs,
                max_heavy_seq_len=max_heavy_seq_len,
                min_heavy_seq_len=min_heavy_seq_len)


def get_chain_seqs(fasta_file_path):
    """Gets the sequence of the heavy chain in a fasta file."""
    seqs = dict()
    for chain in SeqIO.parse(fasta_file_path, 'fasta'):
        if ':H' in chain.id or 'heavy' in chain.id:
            seqs.update({'H': letter_to_num(str(chain.seq), _aa_dict)})
        else:
            raise ValueError('Expected a heavy chain (H) in {}'.format(
                fasta_file_path))
    return seqs


def get_cdr_indices(pdb_file_path):
    """Gets the indices of the CDR loop residues in the PDB file (for heavy chain only)."""
    cdr_ranges = {
        'h1': [26, 35],
        'h2': [50, 65],
        'h3': [95, 102],
    }

    data = get_pdb_atoms(pdb_file_path)
    data = data.drop_duplicates(
        ['chain_id', 'residue_num',
         'code_for_insertions_of_residues']).reset_index()

    heavy_chain_residues = data[data.chain_id == 'H']
    heavy_residue_nums = heavy_chain_residues.residue_num.astype('int32')

    h1_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h1'])].index)
    h2_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h2'])].index)
    h3_idxs = list(heavy_chain_residues[heavy_residue_nums.isin(
        cdr_ranges['h3'])].index)

    return dict(h1=h1_idxs, h2=h2_idxs, h3=h3_idxs)


def get_info(pdb_file, fasta_file=None, verbose=True, convert_to_degree=True):
    if fasta_file is not None:
        chain_seqs = get_chain_seqs(fasta_file)
    else:
        if verbose:
            print(
                'WARNING: No fasta file given to get_info(), getting sequence '
                'from PDB file')
        chain_seqs = dict()
        parser = PDBParser()
        structure = parser.get_structure(get_id(pdb_file), pdb_file)
        for chain in structure.get_chains():
            id_ = chain.id
            seq = seq1(''.join([residue.resname for residue in chain]))
            if id_ != 'H':
                raise ValueError(
                    'Expected a heavy chain (H). Got chain id: {}'.format(id_))

            chain_seqs.update({'H': letter_to_num(seq, _aa_dict)})

    id_ = get_id(pdb_file)
    cdr_indices = get_cdr_indices(pdb_file)
    pairwise_geometry_mat = protein_pairwise_geometry_matrix(
        pdb_file, fasta_file=fasta_file, convert_to_degree=convert_to_degree)

    info = cdr_indices
    info.update(chain_seqs)
    info.update(dict(pairwise_geometry_mat=pairwise_geometry_mat, id=id_))
    return info
