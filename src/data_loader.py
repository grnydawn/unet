# File: data_loader.py
import os
import numpy as np

def load_bias_correction_data(
    input_base_dir: str,
    target_base_dir: str,
    lead_time: int,
    input_vars: list[str],
    output_vars: list[str],
    test: bool = False
):
    """
    Load NPZ data and compute normalization stats.

    In train/val mode (test=False) returns:
        train_data: dict with keys input_data, target_data, input_vars, output_vars
        val_data:   dict with same keys, using the “Test” files
        input_stats:  {var: {'mean':…, 'std':…}}
        target_stats: {var: {'mean':…, 'std':…}}

    In test mode (test=True) returns:
        test_data: dict with keys input_data, target_data, input_vars, output_vars
    """
    # helper to load a .npz into a dict[var→array]
    def _load_npz(path, vars):
        data = {}
        with np.load(path) as npz:
            for v in vars:
                if v in npz:
                    data[v] = npz[v]
        return data

    # paths
    train_in  = os.path.join(input_base_dir,  f"Train/MPAS_T{lead_time}.npz")
    train_out = os.path.join(target_base_dir, f"Train/ERA5_T{lead_time}.npz")
    test_in   = os.path.join(input_base_dir,  f"Test/MPAS_T{lead_time}.npz")
    test_out  = os.path.join(target_base_dir, f"Test/ERA5_T{lead_time}.npz")

    # load everything
    train_input  = _load_npz(train_in,  input_vars)
    train_target = _load_npz(train_out, output_vars)
    test_input   = _load_npz(test_in,   input_vars)
    test_target  = _load_npz(test_out,  output_vars)

    if test:
        return {
            'input_data':  test_input,
            'target_data': test_target,
            'input_vars':  input_vars,
            'output_vars': output_vars
        }

    # compute stats on training data
    input_stats = {}
    for v, arr in train_input.items():
        mean = arr.mean()
        std  = arr.std()
        input_stats[v] = {'mean': mean, 'std': std}

    target_stats = {}
    for v, arr in train_target.items():
        mean = arr.mean()
        std  = arr.std()
        target_stats[v] = {'mean': mean, 'std': std}

    train_data = {
        'input_data':  train_input,
        'target_data': train_target,
        'input_vars':  input_vars,
        'output_vars': output_vars
    }
    val_data = {
        'input_data':  test_input,
        'target_data': test_target,
        'input_vars':  input_vars,
        'output_vars': output_vars
    }

    return train_data, val_data, input_stats, target_stats

