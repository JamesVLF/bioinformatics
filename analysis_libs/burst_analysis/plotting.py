import matplotlib.pyplot as plt
import os 

# --------------------------------------------------------------------
# POPULATION-LEVEL BURSTING & GENERAL PLOTTING UTILITIES
# --------------------------------------------------------------------

def plot_neuron_counts(self):
    """
    Plots neuron count over time for each condition using loaded datasets.
    Groups datasets by condition (e.g., 'treated', 'control') and plots neuron count over days.
    """
    from collections import defaultdict

    condition_data = defaultdict(list)

    for key in self.active_datasets:
        sd = self.loader.spike_data.get(key)
        if sd is None or not sd.neuron_attributes:
            continue

        # parse key like "d0_treated" → day=0, condition="treated"
        try:
            parts = key.split("_")
            day = int(parts[0].replace("d", ""))
            condition = parts[1] if len(parts) > 1 else "unknown"
        except Exception:
            print(f"[WARNING] Could not parse key '{key}' into day/condition. Skipping.")
            continue

        neuron_count = len(sd.neuron_attributes)
        condition_data[condition].append((day, neuron_count))

    # Plot each condition
    plt.figure(figsize=(10, 6))
    for condition, entries in condition_data.items():
        entries = sorted(entries)  # sort by day
        days, counts = zip(*entries)
        plt.plot(days, counts, marker='o', linestyle='-', label=condition)

    plt.xlabel("Time (days)")
    plt.ylabel("Neuron Count")
    plt.title("Neuron Count Over Time by Condition")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_raster(spike_data, time_range=None, title="Raster Plot"):
    trains = spike_data.train
    duration = spike_data.length / 1000.0
    t_start, t_end = time_range or (0, duration)

    plt.figure(figsize=(14, 5))
    for y, spikes in enumerate(trains):
        spikes = [s for s in spikes if t_start <= s <= t_end]
        plt.scatter(spikes, [y] * len(spikes), marker="|", color="black", s=4)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")
    plt.xlim(t_start, t_end)
    plt.tight_layout()
    plt.show()

def plot_population_rate(times, rate, bursts, peak_times=None, title=""):
    """
    Plots population firing rate with optional burst spans and peak markers.

    Parameters:
        times (np.ndarray): Time axis (seconds).
        rate (np.ndarray): Smoothed population rate.
        bursts (List[Tuple[int, int]]): List of (start_idx, end_idx) index pairs.
        peak_times (List[float], optional): List of peak times (seconds).
        title (str): Plot title.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(times, rate, label="Population rate")

    # Shade bursts
    if bursts:
        for start_idx, end_idx in bursts:
            plt.axvspan(times[start_idx], times[end_idx], color="gray", alpha=0.3)

    # Draw vertical lines at burst peaks (if provided)
    if peak_times:
        for t_peak in peak_times:
            plt.axvline(t_peak, color="red", linestyle="--", lw=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Firing Rate")
    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_overlay_raster_population(
        trains,
        times,
        smoothed,
        bursts,
        dataset_label="dataset",
        time_range=(0, 180),
        save=False,
        output_dir=None
):
    """
    Plots raster and population FR overlay using precomputed burst results.

    Args:
        trains (list): Spike trains (list of arrays).
        times (np.ndarray): Time axis from burst computation.
        smoothed (np.ndarray): Population firing rate over time.
        bursts (list of tuples): Start/end indices for each burst.
        dataset_label (str): Title/label for plot.
        time_range (tuple): X-axis limits.
        save (bool): Whether to save.
        output_dir (str): If saving, where to write PNG.
    """
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    # Raster
    for y, spike_times in enumerate(trains):
        ax1.scatter(spike_times, [y] * len(spike_times), marker="|", color='black', s=4, alpha=0.6)

    # Population FR
    ax2.plot(times, smoothed, color='red', linewidth=2, alpha=0.7, label="Population FR")

    # Burst shading
    for start_idx, end_idx in bursts:
        ax1.axvspan(times[start_idx], times[end_idx], color='gray', alpha=0.3)

    ax1.set_title(f"{dataset_label} — Raster + Population FR + Bursts")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron Index")
    ax2.set_ylabel("Firing Rate (Hz)", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.set_xlim(time_range)
    fig.tight_layout()

    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{dataset_label}_raster_population_overlay.png"))

    plt.show()

def plot_overlay_raster_dim_bursts(
        trains,
        bursts,
        dataset_label="dataset",
        time_range=(0, 180),
        save=False,
        output_dir=None
):
    """
    Plot raster with overlays showing coordinated low-level activity bursts (dim method).
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Raster
    for y, spike_times in enumerate(trains):
        ax.scatter(spike_times, [y] * len(spike_times), marker="|", color='black', s=4, alpha=0.6)

    # Dim burst windows
    for start, end in bursts:
        if time_range[0] <= end and start <= time_range[1]:  # Only draw if overlaps with visible time
            ax.axvspan(start, end, color='blue', alpha=0.2)

    ax.set_title(f"{dataset_label} — Raster + Dim Activity Windows")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron Index")
    ax.set_xlim(time_range)
    fig.tight_layout()

    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{dataset_label}_raster_dim_overlay.png"))

    plt.show()

def plot_overlay_raster_isi_bursts(
        trains,
        bursts,
        dataset_label="dataset",
        time_range=(0, 180),
        save=False,
        output_dir=None
):
    """
    Plot raster with shaded windows showing high-frequency ISI bursts (aggregated).
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Raster
    for y, spike_times in enumerate(trains):
        ax.scatter(spike_times, [y] * len(spike_times), marker="|", color='black', s=4, alpha=0.6)

    # ISI bursts (usually narrower than dim bursts)
    for start, end in bursts:
        if time_range[0] <= end and start <= time_range[1]:
            ax.axvspan(start, end, color='orange', alpha=0.3)

    ax.set_title(f"{dataset_label} — Raster + ISI Bursts (Neuron-Level Aggregated)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron Index")
    ax.set_xlim(time_range)
    fig.tight_layout()

    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{dataset_label}_raster_isi_overlay.png"))

    plt.show()

def plot_combined_burst_overlay(
        trains,
        times,
        smoothed,
        rms_bursts=None,
        peaks=None,
        dim_bursts=None,
        isi_bursts=None,
        dataset_label="dataset",
        time_range=(0, 180),
        save=False,
        output_dir=None
):
    """
    Plots raster and overlays from all burst detection methods.
    - RMS = Gray shaded + red peak lines
    - Dim = Blue shaded + blue midpoint markers
    - ISI = Orange shaded + orange midpoint markers
    """

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()

    # Raster
    for y, spike_times in enumerate(trains):
        ax1.scatter(spike_times, [y] * len(spike_times), marker="|", color='black', s=4, alpha=0.6)

    # Population FR
    if smoothed is not None and times is not None:
        ax2.plot(times, smoothed, color='red', linewidth=2, alpha=0.7, label="Population FR")
        ax2.set_ylabel("Firing Rate (Hz)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    # RMS Bursts (gray shaded + red dashed peak lines)
    if rms_bursts:
        for start_idx, end_idx in rms_bursts:
            t0, t1 = times[start_idx], times[end_idx]
            if t1 >= time_range[0] and t0 <= time_range[1]:
                ax1.axvspan(t0, t1, color='gray', alpha=0.3)
    if peaks:
        for peak_idx in peaks:
            peak_time = times[peak_idx]
            if time_range[0] <= peak_time <= time_range[1]:
                ax1.axvline(x=peak_time, color='red', linestyle='--', alpha=0.6)

    # Dim Bursts (blue shaded + blue dotted midpoint lines)
    if dim_bursts:
        for start, end in dim_bursts:
            if end >= time_range[0] and start <= time_range[1]:
                ax1.axvspan(start, end, color='blue', alpha=0.2)
                mid = (start + end) / 2
                ax1.axvline(x=mid, color='blue', linestyle=':', alpha=0.6)

    # ISI Bursts (orange shaded + orange dotted midpoint lines)
    if isi_bursts:
        for start, end in isi_bursts:
            if end >= time_range[0] and start <= time_range[1]:
                ax1.axvspan(start, end, color='orange', alpha=0.25)
                mid = (start + end) / 2
                ax1.axvline(x=mid, color='orange', linestyle=':', alpha=0.6)

    # Titles, axes, limits
    ax1.set_title(f"{dataset_label} — Combined Burst Detection Overlay")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron Index")
    ax1.set_xlim(time_range)
    fig.tight_layout()

    # Legend
    custom_lines = [
        plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label='RMS Peak'),
        plt.Line2D([0], [0], color='blue', linestyle=':', lw=2, label='Dim Midpoint'),
        plt.Line2D([0], [0], color='orange', linestyle=':', lw=2, label='ISI Midpoint'),
        plt.Line2D([0], [0], color='gray', lw=8, alpha=0.3, label='RMS Burst'),
        plt.Line2D([0], [0], color='blue', lw=8, alpha=0.2, label='Dim Burst'),
        plt.Line2D([0], [0], color='orange', lw=8, alpha=0.25, label='ISI Burst'),
        plt.Line2D([0], [0], color='red', lw=2, label='Population FR')
    ]
    ax1.legend(handles=custom_lines, loc='upper right')

    # Save (optional)
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{dataset_label}_raster_combined_overlay.png"))

    plt.show()

def plot_burst_similarity_matrix(sim_matrix, title="Burst Similarity (Cosine)", save_path=None):
    """
    mask = np.tril_indices_from(sim_matrix, k=0)
    masked_sim = np.copy(sim_matrix)
    masked_sim[mask] = np.nan
    """
    plt.figure(figsize=(7, 6))
    cmap = plt.cm.hot
    im = plt.imshow(masked_sim, cmap=cmap, vmin=0, vmax=1)

    plt.colorbar(im, label="Cosine Similarity")
    plt.title(title)
    plt.xlabel("Burst Index")
    plt.ylabel("Burst Index")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_pairwise_correlation_matrix(corr_matrix, title="Pairwise IFR Correlation Matrix"):
    """
    Plots a heatmap of pairwise correlations between units.

    Args:
        corr_matrix (np.ndarray): Square matrix of unit-to-unit correlation values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap="hot", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="Pearson r")
    plt.title(title)
    plt.xlabel("Unit Index")
    plt.ylabel("Unit Index")
    plt.tight_layout()
    plt.show()