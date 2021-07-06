import os
import h5py
import numpy as np
import glob
import torch



def load_data_h5(base_dir, partition):
    """
    Load h5 point dataset.

    :param base_dir: path to dataset directory
    :param partition: train, validate or test
    :return all_full, all_partial: ndarray of point sets [num_sets, 2048, 3]
    """

    all_full = []
    all_partial = []
    for h5_name in glob.glob(os.path.join(base_dir, 'surf_data_%s*.h5' % partition)):
        with h5py.File(h5_name) as f:
            full = f['full'][:].astype('float32')
            partial = f['partial'][:].astype('float32')
        all_full.append(full)
        all_partial.append(partial)
    all_full = np.concatenate(all_full, axis=0)
    all_partial = np.concatenate(all_partial, axis=0)
    return all_full, all_partial


def MakeBatchData( fulldatasets, batchsize):
    batchdatalist = []
    size = len(fulldatasets)
    l = 0
    r = batchsize
    while r <= size:
        batch = torch.tensor(fulldatasets[l:r])
        batchdatalist.append(batch)
        l = l + batchsize
        r = r + batchsize
    return batchdatalist


if __name__ == '__main__':
    path = "D:/Git/3d-AAE/data/ScaR_F2P_UNI_2048/"
    full, partial = load_data_h5(path, "train")
    batchdatalist = MakeBatchData(full,32)
    a=1

