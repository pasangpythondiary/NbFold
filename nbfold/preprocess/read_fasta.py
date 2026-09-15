import h5py
import re
import os

def read_fasta(fasta_path, split_char="!", id_field=0):
    '''
        Reads in fasta file containing multiple sequences.
        Split_char and id_field allow to control identifier extraction from header.
        E.g.: set split_char="|" and id_field=1 for SwissProt/UniProt Headers.
        Returns dictionary holding multiple sequences or only single 
        sequence, depending on input file.
    '''
    seqs = dict()
    with open(fasta_path, 'r') as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip().split(split_char)[id_field]
                # replace tokens that are misinterpreted when loading h5
                uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                seqs[uniprot_id] = ''
            else:
                # replace all whitespace chars, join sequences spanning multiple lines, drop gaps, and cast to uppercase
                seq = ''.join(line.split()).upper().replace("-", "")
                # replace all non-standard AAs and map them to unknown/X
                seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X')
                seqs[uniprot_id] += seq
    example_id = next(iter(seqs))
    print("Read {} sequences.".format(len(seqs)))
    print("Example:\n{}\n{}".format(example_id, seqs[example_id]))

    return seqs

def read_all_seq(path, filename, fasta_dir):
    print(path + filename)

    # Open the HDF5 file
    with h5py.File(path + filename, "r") as f:
        # Get all keys from the HDF5 file
        print("Keys: %s" % f.keys())
        # Get the key of interest (adjust index if needed)
        a_group_key = list(f.keys())[5]

        # Extract dataset values as a list of filenames
        ds_arr = f[a_group_key][()]
        # print(ds_arr)
        train_data = ds_arr.tolist()
        print(train_data)

    # Convert bytes to strings for file names
    t_data = [i.decode('utf-8') + ".fasta" for i in train_data]
    print(t_data)

    # Initialize a dictionary to hold all sequences
    all_seq = {}

    # List all files in the FASTA directory
    path = os.listdir(fasta_dir)

    # Loop through each file in `t_data` to extract heavy chain sequences
    for f in t_data:
        seqs = read_fasta(fasta_dir + f)
        print(seqs)
        print(f)
        print(seqs.keys())
        for k in seqs.keys():
            # Only keep sequences with IDs containing "H" (for heavy chain)
            if "H" in k:
                all_seq[k] = seqs[k]

    # Print the number of heavy chain sequences read
    print(f"Total heavy chain sequences: {len(all_seq)}")

    return all_seq
