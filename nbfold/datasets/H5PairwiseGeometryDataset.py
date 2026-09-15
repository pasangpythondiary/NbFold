import h5py
import torch
import torch.utils.data as data
import torch.nn.functional as F
from tqdm import tqdm

from nbfold.util.tensor import pad_data_to_same_shape
from nbfold.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins
from nbfold.util.preprocess import bin_value_matrix
from nbfold.util.masking import MASK_VALUE


class H5PairwiseGeometryDataset(data.Dataset):
    """
    Dataset containing sequence-structure pairs for training AbResNet.
    This version has been modified for nanobody (only heavy chain, no light chain).
    """

    def __init__(self,
                 filename,
                 num_bins=37,
                 max_seq_len=None,
                 mask_distant_orientations=True,
                 mask_fill_value=MASK_VALUE):
        super(H5PairwiseGeometryDataset, self).__init__()

        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape

        self.num_bins = num_bins
        masked_bin_num = 1 if mask_distant_orientations else 0
        self.bins = [
            get_dist_bins(num_bins),
            get_dist_bins(num_bins),
            get_dist_bins(num_bins),
            get_dihedral_bins(num_bins - masked_bin_num),
            get_dihedral_bins(num_bins - masked_bin_num),
            get_planar_bins(num_bins - masked_bin_num)
        ]

        self.max_seq_len = max_seq_len
        self.valid_indices = None
        if max_seq_len is not None:
            self.valid_indices = self.get_valid_indices()
            self.num_proteins = len(self.valid_indices)

        self.mask_distant_orientations = mask_distant_orientations
        self.mask_fill_value = mask_fill_value

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')

        if self.valid_indices is not None:
            index = self.valid_indices[index]

        id_ = self.h5file['id'][index]
        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        embeddings = self.h5file['embs'][index]

        # Only heavy chain (nanobody), no light chain at all
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]

        heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        embeddings = torch.Tensor(embeddings).type(dtype=torch.float)

        # Get CDR loops (if needed for nanobody, might be different for VHHs)
        h3 = self.h5file['h3_range'][index]

        heavy_prim = F.one_hot(heavy_prim.long())

        # Load pairwise geometry matrix (for the heavy chain only)
        try:
            pairwise_geometry_mat = self.h5file['pairwise_geometry_mat'][index][:6, :heavy_seq_len, :heavy_seq_len]
            pairwise_geometry_mat = torch.Tensor(pairwise_geometry_mat).type(dtype=torch.float)
        except Exception:
            raise ValueError('Output matrix not defined')

        # Bin the matrices into discrete bins for classification
        nan_mat = torch.isnan(pairwise_geometry_mat)
        pairwise_geometry_mat = torch.stack([
            bin_value_matrix(mat, b)
            for mat, b in zip(pairwise_geometry_mat, self.bins)
        ])

        if self.mask_distant_orientations:
            distant_pairs = pairwise_geometry_mat[0] == self.num_bins - 1
            pairwise_geometry_mat[3:][:, distant_pairs] = self.num_bins - 1

        pairwise_geometry_mat[nan_mat] = self.mask_fill_value

        # Return heavy chain only — no light chain
        return id_, heavy_prim, pairwise_geometry_mat, h3, embeddings, heavy_seq_len

    def get_valid_indices(self):
        """Only filter based on heavy chain length (no light chain in nanobody)."""
        valid_indices = []
        for i in range(self.h5file['heavy_chain_seq_len'].shape[0]):
            h_len = self.h5file['heavy_chain_seq_len'][i]
            if h_len < self.max_seq_len:
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        samples.sort(key=lambda x: x[-1], reverse=True)  # Sort by heavy chain length
        return H5AntibodyBatch(zip(*samples)).data()


class H5AntibodyBatch:
    def __init__(self, batch_data, mask_fill_value=MASK_VALUE):
        (self.id_, self.heavy_prim, self.pairwise_geometry_mat, self.h3, self.embeddings, self.heavy_seq_len) = batch_data
        self.mask_fill_value = mask_fill_value

    def data(self):
        return self.features(), self.embeddings, self.heavy_seq_len, self.labels()

    def features(self):
        """One-hot encoding for nanobody heavy chain (no light chain)."""
        X = pad_data_to_same_shape(self.heavy_prim, pad_value=0).float()
        X = F.pad(X, (0, 1, 0, 0, 0, 0))

        # No need for chain delimiter for nanobody
        return X.transpose(1, 2).contiguous()

    def labels(self):
        """Pairwise geometry matrix padded to max size."""
        label_mat = pad_data_to_same_shape(
            self.pairwise_geometry_mat,
            pad_value=self.mask_fill_value
        ).transpose(0, 1)

        return label_mat


def h5_antibody_dataloader(filename, batch_size=1, max_seq_len=None, num_bins=36, **kwargs):
    if 'collate_fn' in kwargs:
        raise ValueError('Cannot modify collate_fn')

    kwargs['collate_fn'] = H5PairwiseGeometryDataset.merge_samples_to_minibatch
    kwargs['batch_size'] = batch_size

    return data.DataLoader(
        H5PairwiseGeometryDataset(filename, num_bins=num_bins, max_seq_len=max_seq_len),
        **kwargs
    )
