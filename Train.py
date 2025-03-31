import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_hub.Unet import UNet
from model_hub.Residual_Unet import ResidualUNet
from model_hub.Attention_Unet import AttentionUNet
from model_hub.UnetPlus import UNetPlusPlus
from model_hub.Residual_UnetPlus import ResidualUNetPlusPlus


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class BiasCorrectionDataset(Dataset):
    def __init__(self, lead_time, input_variables=None, output_variables=None,
                 transform=None, input_data=None, target_data=None,
                 input_stats=None, target_stats=None):
        self.lead_time = lead_time
        self.transform = transform

        # Default to all variables if none specified
        self.input_variables = input_variables or ['height_500', 'height_850', 'height_925',
                                                  'temperature_500', 'temperature_850', 'temperature_925', 
                                                  'uzonal_500', 'uzonal_850', 'uzonal_925',
                                                  'umeridional_500', 'umeridional_850', 'umeridional_925', 
                                                  'spechum_500', 'spechum_850', 'spechum_925',
                                                  'relhum_500', 'relhum_850', 'relhum_925',
                                                  'mslp', 'u10', 'v10', 't2m', 'd2m']

        self.output_variables = output_variables or ['t2m']  # Default to t2m if none specified

        # Use pre-loaded data if provided, otherwise load it
        self.input_data = input_data
        self.target_data = target_data

        # Get number of samples
        self.num_samples = self.input_data[self.input_variables[0]].shape[0] if hasattr(self.input_data[self.input_variables[0]], 'shape') else 1

        # Store normalization statistics
        self.input_stats = input_stats
        self.target_stats = target_stats

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Extract selected input variables and stack them
        input_arrays = []
        for var in self.input_variables:
            if idx < self.input_data[var].shape[0]:  # Check if idx is valid
                var_data = self.input_data[var][idx].copy()  # Create a copy to avoid modifying original data

                # Apply normalization if stats available
                if self.input_stats is not None and var in self.input_stats:
                    mean, std = self.input_stats[var]['mean'], self.input_stats[var]['std']

                    # # Handle different dimensions between data and statistics
                    # if mean.ndim != var_data.ndim:
                    #     # Reshape mean and std to match var_data dimensions for proper broadcasting
                    #     if var_data.ndim == 2 and mean.ndim == 1:
                    #         # For 2D data like (720, 1440) and 1D stats like (365,)
                    #         # Assume the stats are per-day, reshape to apply properly
                    #         mean = np.mean(mean)  # Use a single global mean
                    #         std = np.mean(std)    # Use a single global std

                    # Normalize with broadcasting
                    var_data = (var_data - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

                input_arrays.append(var_data)
            else:
                raise IndexError(f"Index {idx} out of bounds for variable {var}")

        # Stack input arrays along the channel dimension
        stacked_input = np.stack(input_arrays, axis=0)

        # Extract selected output variables and stack them
        output_arrays = []
        for var in self.output_variables:
            if idx < self.target_data[var].shape[0]:  # Check if idx is valid
                var_data = self.target_data[var][idx].copy()  # Create a copy to avoid modifying original data

                # Apply normalization if stats available
                if self.target_stats is not None and var in self.target_stats:
                    mean, std = self.target_stats[var]['mean'], self.target_stats[var]['std']

                    # # Handle different dimensions between data and statistics
                    # if mean.ndim != var_data.ndim:
                    #     # Reshape mean and std to match var_data dimensions for proper broadcasting
                    #     if var_data.ndim == 2 and mean.ndim == 1:
                    #         # For 2D data like (720, 1440) and 1D stats like (365,)
                    #         mean = np.mean(mean)  # Use a single global mean
                    #         std = np.mean(std)    # Use a single global std

                    # Normalize with broadcasting
                    var_data = (var_data - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

                output_arrays.append(var_data)
            else:
                raise IndexError(f"Index {idx} out of bounds for variable {var}")

        # Stack output arrays along the channel dimension
        stacked_output = np.stack(output_arrays, axis=0)

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(stacked_input, dtype=torch.float32)
        target_tensor = torch.tensor(stacked_output, dtype=torch.float32)
        # Apply optional transformations
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)

        return input_tensor, target_tensor


def train_unet_individual_lead_time(model, train_dataset, val_dataset, criterion, optimizer,
                                    lead_time, dataset_name='bias_correction', num_epochs=200,
                                    batch_size=32, patience=5, device=None):
            
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    print("We are using ", device)
            
    print(f"\n🔹 Training Model for Lead Time = {lead_time} hours")
            
    # Create data loaders for train and validation
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Early stopping tracking
    epochs_no_improve = 0
    best_val_loss = float('inf')

    # Track losses
    train_losses = []
    val_losses = []
    batch_losses = []  # Track individual batch losses

    # Train for specified number of epochs
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        epoch_batch_losses = []  # Track batch losses for current epoch

        for batch_idx, (batch_inputs, batch_targets) in enumerate(train_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            loss.backward()
            optimizer.step()


            # Record batch loss
            batch_loss = loss.item()
            epoch_batch_losses.append(batch_loss)

            # Log every 10 batches
            if (batch_idx + 1) % 1 == 0:
                print(f"  Lead Time {lead_time}, Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Batch Loss: {batch_loss:.4f}")

            train_epoch_loss += batch_loss

        # Store batch losses for this epoch
        batch_losses.append(epoch_batch_losses)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)

                val_epoch_loss += loss.item()

        # Average losses
        avg_train_loss = train_epoch_loss / len(train_loader)
        avg_val_loss = val_epoch_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)


        # Print progress
        print(f"Lead Time {lead_time}, Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save best model
            torch.save(model.state_dict(), f'checkpoint/{dataset_name}_lead_time_{lead_time}_best.pth')
        else:
            epochs_no_improve += 1


        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return {
        'lead_time': lead_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'batch_losses': batch_losses,  # Return all batch losses
        'best_val_loss': best_val_loss
    }


def evaluate_model(model, test_loader, device, target_stats=None):
    """
    Evaluate the model on test data and collect predictions
    
    Args:
        model (nn.Module): Trained UNet model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        target_stats (dict, optional): Statistics to denormalize predictions
    
    Returns:
        Tuple of (all_truths, all_predictions)
    """
    model.eval()
    all_truths = []
    all_predictions = []


    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.cpu().numpy().squeeze(axis=1)

            # Move model predictions to CPU for numpy conversion
            outputs = model(batch_inputs).cpu().numpy().squeeze(axis=1)

            all_truths.append(batch_targets)
            all_predictions.append(outputs)

    # Concatenate all batches
    all_truths = np.concatenate(all_truths, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    print ("Evaluation Shape:")
    print ("Truth: ", all_truths.shape)
    print ("Predictions: ", all_predictions.shape)


    # Denormalize predictions and truths if stats are provided
    # Denormalize predictions and truths if stats are provided
    if target_stats is not None:
        for i, var in enumerate(test_loader.dataset.output_variables):
            if var in target_stats:
                mean, std = target_stats[var]['mean'], target_stats[var]['std']

                print(f"Denormalizing {var} with scalar mean={mean}, std={std}")

                all_truths = all_truths * (std + 1e-8) + mean
                all_predictions = all_predictions * (std + 1e-8) + mean

    return all_truths, all_predictions


def normalize(float_array, vmax, vmin):
    """
    Normalize the array to range [0, 255] as uint8.
    
    Args:
        float_array: Input array
        vmax: Maximum value for normalization
        vmin: Minimum value for normalization
        
    Returns:
        Normalized array as uint8
    """
    # Normalize the array to range [0, 1]
    norm_array = (float_array - vmin) / (vmax - vmin)
    # Scale and convert to integers in range [0, 255]
    int_array = (norm_array * 255).astype(np.uint8)
    return int_array


def quantile_rmse(x, y, q):
    """
    Calculate RMSE for values above a certain quantile.
    
    Args:
        x: predictions
        y: truth 
        q: quantile threshold (0-1). 1,2,3 sigma = 0.6827, 0.9545, 0.9973  
    """
    index = np.where(y >= np.quantile(y, q))
    rmse = np.sqrt(np.mean(np.square(x[index] - y[index])))
    return rmse

def psd(y, bins=np.arange(0, 0.52, 0.02), cache=None):
    """
    Compute Power Spectral Density (PSD) of an image y.
    Args:
    - y: Input image with shape (time, lat, lon).
    - bins: Array of bin edges for binning the wavenumbers.
    - cache: Dictionary to cache frequency grid calculations
    Returns:
    - psd_array: Array of PSD values with shape (time, K)
    - bin_edges: Bin edges used for binning the wavenumbers.
    """
    # Reuse frequency grids if available
    h, w = y.shape[-2], y.shape[-1]

    if cache is None:
        cache = {}

    grid_key = (h, w)
    if grid_key in cache:
        kx, ky = cache[grid_key]
    else:
        freq = np.fft.fftshift(np.fft.fftfreq(h))
        freq2 = np.fft.fftshift(np.fft.fftfreq(w))
        kx, ky = np.meshgrid(freq, freq2)
        kx = kx.T
        ky = ky.T
        cache[grid_key] = (kx, ky)

    # Compute 2D FFT of the input image (all at once if possible)
    ffts = np.fft.fftshift(np.abs(np.fft.fft2(y)) ** 2, axes=(-2, -1))

    # Precompute the flattened k values
    k_values = np.sqrt(kx.ravel() ** 2 + ky.ravel() ** 2)

    # Process all time steps in one batch if possible
    n_times = ffts.shape[0]
    results = np.zeros((n_times, len(bins) - 1))

    # Process in batches to save memory if needed
    batch_size = min(n_times, 10)  # Adjust based on available memory
    for batch_start in range(0, n_times, batch_size):
        batch_end = min(batch_start + batch_size, n_times)
        batch_ffts = ffts[batch_start:batch_end]

        # Reshape to (batch_size, h*w) for efficient binning
        batch_ffts_flat = batch_ffts.reshape(batch_end - batch_start, -1)

        for i, t in enumerate(range(batch_start, batch_end)):
            results[t] = binned_statistic(
                k_values,
                values=batch_ffts_flat[i],
                statistic="mean",
                bins=bins
            ).statistic



    # Compute PSD for the last time step (for normalization)
    bin_info = binned_statistic(
        k_values,
        values=ffts[-1].ravel(),
        statistic="mean",
        bins=bins
    )

    bin_width = np.abs(bin_info.bin_edges[1] - bin_info.bin_edges[0])
    # Normalize and return
    return results / bin_width, bin_info.bin_edges
def calc_RALSD(truths, preds, nsamples=-1):
    """
    Calculate the RALSD (Relative Amplitude Log-Spectral Distance) metric.
    
    Args:
        truths: Ground truth values
        preds: Model predictions
        nsamples: Number of samples to use (-1 for all)
    """
    if nsamples > 0:
        truths = truths[:nsamples]
        preds = preds[:nsamples]

    # Create cache for frequency grid calculations
    cache = {}

    # Calculate both PSDs at once
    truth_psd, _ = psd(truths, cache=cache)
    pred_psd, _ = psd(preds, cache=cache)


    # Calculate mean PSD across frequency bins
    truth_mean = truth_psd.mean(axis=0)
    pred_mean = pred_psd.mean(axis=0)

    # Avoid division by zero
    epsilon = 1e-10
    ratio = truth_mean / (pred_mean + epsilon)

    # Calculate RALSD
    return np.sqrt(np.mean(np.square(10 * np.log10(ratio + epsilon))))


    # Calculate mean PSD across frequency bins
    truth_mean = truth_psd.mean(axis=0)
    pred_mean = pred_psd.mean(axis=0)

    # Avoid division by zero
    epsilon = 1e-10
    ratio = truth_mean / (pred_mean + epsilon)

    # Calculate RALSD
    return np.sqrt(np.mean(np.square(10 * np.log10(ratio + epsilon))))


def calculate_metrics(truths, predictions, return_p_value=False):
    """
    Calculate evaluation metrics
    
    Args:
        truths (numpy.ndarray): Ground truth values
        predictions (numpy.ndarray): Model predictions
        return_p_value (bool, optional): Whether to return p-value with correlation coefficient
    
    Returns:
        Dict of evaluation metrics
    """
    # Make a copy to avoid modifying the original arrays
    truths_proc = truths.copy()
    preds_proc = predictions.copy()

    # Mean Squared Error
    mse = np.mean((truths_proc - preds_proc) ** 2)

    # Mean Absolute Error
    mae = np.mean(np.abs(truths_proc - preds_proc))


    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Pearson correlation coefficient
    corr_coef, p_value = stats.pearsonr(truths_proc.ravel(), preds_proc.ravel())

    # Quantile RMSE calculations
    s1, s2, s3 = 0.6827, 0.9545, 0.9973
    s1rmse = quantile_rmse(preds_proc, truths_proc, s1)
    s2rmse = quantile_rmse(preds_proc, truths_proc, s2)
    s3rmse = quantile_rmse(preds_proc, truths_proc, s3)

    # RALSD calculation
    ralsd = calc_RALSD(truths_proc, preds_proc)

    # Calculate SSIM and PSNR using batched approach for memory efficiency
    num_sam = truths.shape[0]
    ssim_scores = np.zeros(num_sam)
    psnr_values = np.zeros(num_sam)


    # Process in smaller batches to manage memory
    batch_size = 64  # Adjust based on memory constraints
    for i in range(0, num_sam, batch_size):
        end_idx = min(i + batch_size, num_sam)

        for j in range(i, end_idx):

            pred = preds_proc[j]
            truth = truths_proc[j]

            vmin = min(np.nanmin(pred), np.nanmin(truth))
            vmax = max(np.nanmax(pred), np.nanmax(truth))

            norm_pred = normalize(pred, vmax=vmax, vmin=vmin)
            norm_truth = normalize(truth, vmax=vmax, vmin=vmin)

            ssim_scores[j] = ssim(norm_truth, norm_pred)
            psnr_values[j] = psnr(norm_truth, norm_pred)


    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'PSNR': psnr_value,
        'CorrR': corr_coef,
        'SSIM': ssim_score,
        'RMSE_q0.6827': s1rmse,
        'RMSE_q0.9545': s2rmse,
        'RMSE_q0.9973': s3rmse,
        'RALSD': ralsd
    }
        
    return metrics

def save_results(dataset_name, lead_time, test_truths, test_predictions, metrics,
                output_variables):
    """
    Save test results and metrics for a given dataset
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'bias_correction')
        lead_time (int): Lead time in hours
        test_truths (numpy.ndarray): Ground truth values
        test_predictions (numpy.ndarray): Model predictions
        metrics (dict): Calculated evaluation metrics
        output_variables (list): List of output variable names
        specific_indices (list, optional): Specific indices to extract
    """
    # Create results directory with dataset name and lead time
    results_dir = f'{dataset_name}_results/lead_time_{lead_time}'
    os.makedirs(results_dir, exist_ok=True)

    # Save numpy arrays
    np.save(f'{results_dir}/test_truths.npy', test_truths)
    np.save(f'{results_dir}/test_predictions.npy', test_predictions)

def save_results(dataset_name, lead_time, test_truths, test_predictions, metrics,
                output_variables):
    """
    Save test results and metrics for a given dataset
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'bias_correction')
        lead_time (int): Lead time in hours
        test_truths (numpy.ndarray): Ground truth values
        test_predictions (numpy.ndarray): Model predictions
        metrics (dict): Calculated evaluation metrics
        output_variables (list): List of output variable names
        specific_indices (list, optional): Specific indices to extract
    """
    # Create results directory with dataset name and lead time
    results_dir = f'{dataset_name}_results/lead_time_{lead_time}'
    os.makedirs(results_dir, exist_ok=True)

    # Save numpy arrays
    np.save(f'{results_dir}/test_truths.npy', test_truths)
    np.save(f'{results_dir}/test_predictions.npy', test_predictions)


def main():

    # Set device for this process
    if torch.backends.mps.is_available():  # For Apple Silicon
        device = torch.device("mps")
    elif torch.cuda.is_available():
        # Check if ROCm (AMD HIP) is being used
        if torch.version.hip is not None:
            print("Using AMD HIP (ROCm)")
            device = torch.device("cuda")  # Still 'cuda' even if it's HIP
        else:
            print("Using NVIDIA CUDA")
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")


    # Print GPU status
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"No GPU available, using CPU.")

    seed_everything(0)

    model_options = {
    "unet": UNet,
    "residual_unet": ResidualUNet,
    "attention_unet": AttentionUNet,
    "unet_plus_plus": UNetPlusPlus,
    "residual_unet_plus": ResidualUNetPlusPlus
        }

    # Parse arguments
    parser = argparse.ArgumentParser(description="Select a segmentation model.")
    parser.add_argument("--model", type=str, choices=model_options.keys(), default="unet", help="Choose a model")
    parser.add_argument("--dataset", type=str, default="ResidualUNet", help="Specify the dataset name")
    parser.add_argument("--base_channels", type=int, default=8, help="Number of base channels for the model")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for the model")
    args = parser.parse_args()


    # Set dataset name
    dataset_name = args.dataset

    # Define base directories
    input_base_dir = "/lustre/orion/atm112/proj-shared/patrickfan/MPAS_data/"
    target_base_dir = "/lustre/orion/atm112/proj-shared/patrickfan/ERA5_data/"

    # Define lead times
    lead_times = [6, 12, 18, 24, 30, 36, 42, 48]

    # Define input and output variables
    input_variables = ['mslp', 'u10', 'v10', 't2m', 'q2', 'd2m']
    output_variables = ['t2m']

    # Define hyperparameters
    num_epochs = 200
    batch_size = args.batch_size
    patience = 5
    base_channels = args.base_channels

    # Model parameters
    in_channels = len(input_variables)
    out_channels = len(output_variables)


    # Instantiate the selected model
    ModelClass = model_options[args.model]

    # Create checkpoint directory (only rank 0 needs to do this)
    os.makedirs('checkpoint', exist_ok=True)

    # Create checkpoint directory (only rank 0 needs to do this)
    os.makedirs('Results', exist_ok=True)

    # Create checkpoint directory (only rank 0 needs to do this)
    os.makedirs(f'checkpoint/{dataset_name}', exist_ok=True)

     # Create checkpoint directory (only rank 0 needs to do this)
    os.makedirs(f'Results/{dataset_name}_results', exist_ok=True)

    # Store training results for each lead time
    all_lead_time_results = {}

    # Dictionary to store normalization stats for each lead time
    lead_time_input_stats = {}
    lead_time_target_stats = {}

    for lead_time in lead_times:

        print(f"\n{'='*50}")
        print(f"\nProcessing data for lead time {lead_time}...")
        print(f"\n{'='*50}")

        # Load data for this lead time only
        train_input_file = f"{input_base_dir}/Train/MPAS_T{lead_time}.npz"
        train_target_file = f"{target_base_dir}/Train/ERA5_T{lead_time}.npz"
        test_input_file = f"{input_base_dir}/Test/MPAS_T{lead_time}.npz"
        test_target_file = f"{target_base_dir}/Test/ERA5_T{lead_time}.npz"

        # Load training data
        train_input_data = {}
        with np.load(train_input_file) as data:
            for var in input_variables:
                if var in data:
                    train_input_data[var] = data[var]

        train_target_data = {}
        with np.load(train_target_file) as data:
            for var in output_variables:
                if var in data:
                    train_target_data[var] = data[var]

        input_stats = {}
        for var in input_variables:
            if var in train_input_data:
                var_data = train_input_data[var]
                # Handle different dimensions
                if var_data.ndim > 2:
                    # Calculate stats across all dimensions except the first (samples)
                    axes = tuple(range(0, var_data.ndim))
                    mean = np.mean(var_data, axis=axes)
                    std = np.std(var_data, axis=axes)
                else:
                    mean = np.mean(var_data)
                    std = np.std(var_data)

                input_stats[var] = {'mean': mean, 'std': std}

        # Calculate target stats for this lead time
        target_stats = {}
        for var in output_variables:
            if var in train_target_data:
                var_data = train_target_data[var]
                # Handle different dimensions
                if var_data.ndim > 2:
                    # Calculate stats across all dimensions except the first (samples)
                    axes = tuple(range(0, var_data.ndim))
                    mean = np.mean(var_data, axis=axes)
                    std = np.std(var_data, axis=axes)
                else:
                    mean = np.mean(var_data)
                    std = np.std(var_data)

                target_stats[var] = {'mean': mean, 'std': std}

        # Load test data
        test_input_data = {}
        with np.load(test_input_file) as data:
            for var in input_variables:
                if var in data:
                    test_input_data[var] = data[var]

        test_target_data = {}
        with np.load(test_target_file) as data:
            for var in output_variables:
                if var in data:
                    test_target_data[var] = data[var]

        print(f"\n{'='*50}")
        print(f"Training for Lead Time: {lead_time} hours")
        print(f"{'='*50}")

        # Create training dataset for the current lead time using pre-loaded data
        train_dataset = BiasCorrectionDataset(
            lead_time=lead_time,
            input_variables=input_variables,
            output_variables=output_variables,
            input_data=train_input_data,
            target_data=train_target_data,
            input_stats=input_stats,
            target_stats=target_stats
        )

        # Create validation dataset using a portion of test data
        val_dataset = BiasCorrectionDataset(
            lead_time=lead_time,
            input_variables=input_variables,
            output_variables=output_variables,
            input_data=test_input_data,
            target_data=test_target_data,
            input_stats=input_stats,  # Use same normalization stats as training
            target_stats=target_stats
        )

        # Initialize a new model for each lead time
        model = ModelClass(in_channels=in_channels, out_channels=out_channels, num_blocks=5, base_channels=base_channels)
        # model.load_state_dict(torch.load(f'checkpoint/{dataset_name}_lead_time_{lead_time}_best.pth'))
        model = model.to(device)

        # Loss and Optimizer
        criterion = nn.MSELoss()  # Mean Squared Error Loss
        optimizer = optim.Adam(model.parameters(), lr=5e-4)

        # Train model for this lead time
        lead_time_result = train_unet_individual_lead_time(
            model,
            train_dataset,
            val_dataset,
            criterion,
            optimizer,
            lead_time=lead_time,
            dataset_name=dataset_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            patience=patience,
            device=device
        )

        # Store results
        all_lead_time_results[lead_time] = lead_time_result

        # Load the best model for this lead time
        best_model = ModelClass(in_channels=in_channels, out_channels=out_channels, num_blocks=5, base_channels=base_channels)
        best_model.load_state_dict(torch.load(f'checkpoint/{dataset_name}_lead_time_{lead_time}_best.pth'))
        best_model = best_model.to(device)
        # Create test dataset using the full test data
        test_dataset = BiasCorrectionDataset(
                lead_time=lead_time,
                input_variables=input_variables,
                output_variables=output_variables,
                input_data=test_input_data,
                target_data=test_target_data,
                input_stats=input_stats,
                target_stats=target_stats
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        # Evaluate model on test data (with denormalization)
        test_truths, test_predictions = evaluate_model(
            best_model,
            test_loader,
            device,
            target_stats=target_stats  # Pass stats for denormalization
        )

        # Calculate metrics
        metrics = calculate_metrics(test_truths, test_predictions)

        # Save results with output variable information
        save_results(dataset_name, lead_time, test_truths, test_predictions, metrics,
                    output_variables)

    print("\nIndividual Lead Time Training and Evaluation Complete!")
    print(f"Trained {len(lead_times)} separate models, one for each lead time.")

if __name__ == "__main__":
    main()
