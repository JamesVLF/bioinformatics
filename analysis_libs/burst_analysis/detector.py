import numpy as np
from scipy import ndimage

def detect_population_bursts(spike_data, bin_size_ms=10, smooth_sigma=2,
                              threshold_std=2.0, min_duration_ms=30):
    """
    Compute population firing rate and detect high-activity bursts.

    Args:
        spike_data (SpikeData): Aligned spike trains.
        bin_size_ms (int): Histogram bin size in ms.
        smooth_sigma (float): Gaussian smoothing sigma.
        threshold_std (float): Threshold in SD above baseline.
        min_duration_ms (float): Minimum burst length.

    Returns:
        List[dict]: Each dict = {'start', 'end', 't_peak'} in ms
    """
    times, rates = spike_data.population_firing_rate(
        bin_size=bin_size_ms, w=1, average=True
    )
    smoothed = ndimage.gaussian_filter1d(rates, sigma=smooth_sigma)

    baseline = np.median(smoothed)
    std_dev = np.std(smoothed)
    threshold = baseline + threshold_std * std_dev

    above = smoothed > threshold
    labeled, n = ndimage.label(above)

    bursts = []
    for i in range(1, n + 1):
        mask = labeled == i
        if not np.any(mask):
            continue
        start_idx, end_idx = np.where(mask)[0][[0, -1]]
        start, end = times[start_idx], times[end_idx]
        if end - start < min_duration_ms:
            continue
        peak_idx = start_idx + np.argmax(smoothed[start_idx:end_idx + 1])
        bursts.append({
            "start": float(start),
            "end": float(end),
            "t_peak": float(times[peak_idx])
        })

    return bursts