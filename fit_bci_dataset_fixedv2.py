import mne
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch
import json
import os
from glob import glob
from tqdm import tqdm
import time

# Double-Gaussian PSD model (theta + alpha core + power-law background)
def psd_model(f, A_theta, fp_theta, s_theta, A_alpha, fp_alpha, s_alpha, alpha, offset):
    theta = A_theta * np.exp( - (f - fp_theta)**2 / (2 * s_theta**2) )
    alpha_core = A_alpha * np.exp( - (f - fp_alpha)**2 / (2 * s_alpha**2) )
    background = (f + 0.1)**alpha
    return theta + alpha_core + background + offset

def process_file(vhdr_path, channel='Cz', label='data'):
    start_time = time.time()
    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        fs = raw.info['sfreq']
        data = raw.get_data(picks=[channel])[0]
        # Scale to uV if in V
        if np.max(np.abs(data)) < 1e-3:
            data *= 1e6
        max_amp = np.max(np.abs(data))
        print(f"Loaded '{label}': fs={fs} Hz, max amp {max_amp:.2f} uV")
        
        if max_amp < 1.0:
            print("  Skipping — amplitude too low (likely artifact/flat)")
            return None
    except Exception as e:
        print(f"  Load failed: {e}")
        return None

    nperseg = min(1024, len(data) // 2)
    noverlap = nperseg // 2
    f, psd = welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    f = np.asarray(f)
    psd = np.asarray(psd)
    
    mask = (f >= 1) & (f <= 50)
    f_fit = f[mask]
    psd_fit = psd[mask]
    
    if len(f_fit) < 8:
        print(f"  Too few PSD bins ({len(f_fit)})")
        return None
    
    max_psd = np.max(psd_fit)
    min_psd = np.min(psd_fit) if np.min(psd_fit) > 0 else 1e-10
    
    # p0 biased toward theta + alpha
    p0 = [max_psd * 0.4, 4, 3, max_psd * 0.3, 10, 2, -1.5, min_psd * 0.3]
    
    # Safe bounds for EEG PSD (uV²/Hz scale)
    lower = [1e-6, 0.5, 0.5, 1e-6, 5, 0.5, -3.5, 1e-6]
    upper = [1e6, 12, 12, 1e6, 18, 12, -0.4, 1e4]
    bounds = (lower, upper)
    
    # Clip p0 to bounds
    p0 = np.clip(p0, lower, upper)
    
    try:
        popt, pcov = curve_fit(psd_model, f_fit, psd_fit, p0=p0, bounds=bounds, maxfev=15000)
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None
    
    f_dense = np.linspace(1, 50, 2000)
    psd_dense = psd_model(f_dense, *popt)
    total_power = np.trapezoid(psd_dense, f_dense)
    moment2 = np.trapezoid(psd_dense * f_dense**2, f_dense)
    rms_freq = np.sqrt(moment2 / total_power) if total_power > 0 else 0.0
    
    psd_pred = psd_model(f_fit, *popt)
    residuals = psd_fit - psd_pred
    chi2 = np.sum(residuals**2 / np.var(psd_fit)) if np.var(psd_fit) > 0 else 0
    dof = len(f_fit) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else 0.0
    
    results = {
        "label": label,
        "file": vhdr_path,
        "params": dict(zip(["A_theta", "fp_theta", "s_theta", "A_alpha", "fp_alpha", "s_alpha", "alpha", "offset"], popt.tolist())),
        "errors": perr.tolist(),
        "derived": {"rms_freq_Hz": float(rms_freq), "chi2_red": float(chi2_red), "total_power": float(total_power)}
    }
    
    elapsed = time.time() - start_time
    print(f"\n{label}")
    print("─" * 60)
    print(f"fp_theta = {popt[1]:.2f} Hz ± {perr[1]:.2f}")
    print(f"fp_alpha = {popt[4]:.2f} Hz ± {perr[4]:.2f}")
    print(f"s_theta  = {popt[2]:.2f} Hz ± {perr[2]:.2f}")
    print(f"s_alpha  = {popt[5]:.2f} Hz ± {perr[5]:.2f}")
    print(f"alpha    = {popt[6]:.3f} ± {perr[6]:.3f}")
    print(f"RMS      = {rms_freq:.2f} Hz")
    print(f"χ²_red   = {chi2_red:.3f}")
    print(f"Time     = {elapsed:.2f} s")
    print("─" * 60)
    
    return results

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────
print("Current dir:", os.getcwd())
vhdr_files = glob("sub-*/ses-*/eeg/*.vhdr")
print(f"Found {len(vhdr_files)} .vhdr files")
if vhdr_files:
    print("First 5:", vhdr_files[:5])
else:
    print("No files found — check directory and ls sub-*/ses-*/eeg/*.vhdr")

save_dir = "./bci_fits"
os.makedirs(save_dir, exist_ok=True)

metrics_list = []
failed = 0

for vhdr in tqdm(vhdr_files):
    label = os.path.basename(vhdr).replace('.vhdr', '')
    result = process_file(vhdr, channel='Cz', label=label)
    if result:
        metrics_list.append(result)
    else:
        failed += 1
    
    # Save individual JSON
    json_path = os.path.join(save_dir, f"{label}.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4) if result else json.dump({"error": "fit failed"}, f, indent=4)

# Summary table
print("\nSUMMARY ({} successful, {} failed)".format(len(metrics_list), failed))
print("─" * 100)
print(f"{'File':<60} {'fp_theta':<12} {'fp_alpha':<12} {'s_theta':<10} {'s_alpha':<10} {'alpha':<10} {'RMS':<12} {'χ²_red':<10}")
print("─" * 100)
for m in metrics_list:
    p = m['params']
    d = m['derived']
    lbl = m['label'][:60]
    print(f"{lbl:<60} {p['fp_theta']:<12.2f} {p['fp_alpha']:<12.2f} {p['s_theta']:<10.2f} {p['s_alpha']:<10.2f} {p['alpha']:<10.3f} {d['rms_freq_Hz']:<12.2f} {d['chi2_red']:<10.3f}")

with open(os.path.join(save_dir, "all_metrics.json"), "w") as f:
    json.dump(metrics_list, f, indent=4)

print(f"\nAll saved to {save_dir}")
print(f"Total time estimate for full 384 files: ~{len(vhdr_files) * 0.3 / 60:.1f} minutes on your machine")
