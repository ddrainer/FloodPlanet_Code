from collections import defaultdict

import torch
import numpy as np

from st_water_seg.datasets.utils import get_dset_path
from st_water_seg.datasets.floodplanet import Floodplanet_Dataset


DATASETS = {
    'floodplanet': Floodplanet_Dataset,
}


def tensors_and_lists_collate_fn(data):
    list_key_names = ['metadata']

    out_data = defaultdict(list)
    for ex in data:
        for k, v in ex.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v)
            out_data[k].append(v)
        
    for k, v in out_data.items():
        if k in list_key_names:
            out_data[k] = v
        else:
            out_data[k] = torch.stack(v, dim=0)
    
    return out_data


def build_dataset(dset_name, split, slice_params, eval_region, sensor,
                  channels, **kwargs):
    dset_root_dir = get_dset_path(dset_name)

    try:
        # Only directly input required parameters.
        dataset = DATASETS[dset_name](dset_root_dir,
                                      split,
                                      slice_params,
                                      channels=channels,
                                      eval_region=eval_region,
                                      sensor=sensor,
                                      **kwargs)
    except KeyError:
        raise KeyError(
            f'DATASETS dictionary does not contain a dataset class for dataset name "{dset_name}"'
        )
    return dataset
