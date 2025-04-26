import torch 
from tabulate import tabulate

def compute_snr(clean, denoised):
    noise = clean - denoised
    snr_value = 10 * torch.log10(torch.sum(clean ** 2) / torch.sum(noise ** 2))
    return snr_value.item()

def frequency_band_error_db(fft_output, fft_target, freqs, bands, eps=1e-12):
    errors_db = {}
    for band_name, (f_low, f_high) in bands.items():
        band_mask = (freqs >= f_low) & (freqs < f_high)
        power_output = 10 * torch.log10(torch.sum(torch.abs(fft_output[..., band_mask])**2, dim=-1) + eps)
        power_target = 10 * torch.log10(torch.sum(torch.abs(fft_target[..., band_mask])**2, dim=-1) + eps)
        error_db = power_output/power_target
        errors_db[band_name] = error_db.mean().item()
    return errors_db

def evaluate(targets, outputs, fs):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45)
    }
    fft_output = torch.fft.rfft(outputs)
    fft_target = torch.fft.rfft(targets)
    freqs = torch.fft.rfftfreq(outputs.shape[-1], d=1/fs)
    
    snr_value = compute_snr(targets, outputs)
    band_errors = frequency_band_error_db(fft_output, fft_target, freqs, bands)
    l1_time = torch.mean(torch.abs(targets - outputs)).item()
    l2_time = torch.mean((targets - outputs) ** 2).item()
    
    l1_freq = torch.mean(torch.abs(torch.abs(fft_target) - torch.abs(fft_output))).item()
    l2_freq = torch.mean((torch.abs(fft_target) - torch.abs(fft_output)) ** 2).item()
    return snr_value, band_errors, l1_time, l2_time, l1_freq, l2_freq

def print_metrics(epoch, mode, avg_loss, snr, band_errors, l1_time, l2_time, l1_freq, l2_freq):
    # Prepare table data
    headers = ["Metric", "Value"]
    table_data = [
        ["Loss", f"{avg_loss:.6f}"],
        ["SNR (dB)", f"{snr:.2f}"],
        ["L1 Time", f"{l1_time:.6f}"],
        ["L2 Time", f"{l2_time:.6f}"],
        ["L1 Freq", f"{l1_freq:.6f}"],
        ["L2 Freq", f"{l2_freq:.6f}"],
    ]
    # Add frequency band errors
    for band_name, error in band_errors.items():
        table_data.append([f"{band_name} Error (dB)", f"{error:.2f}"])
    
    print(f"\nEpoch {epoch} - {mode.capitalize()} Metrics:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
