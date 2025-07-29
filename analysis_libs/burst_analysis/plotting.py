import matplotlib.pyplot as plt
import os 

# --------------------------------------------------------------------
#               BURST DETECTION PLOTS
# --------------------------------------------------------------------
class BurstDetectionPlots
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

        Args:
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

# ---------------------------------------------------------
#           BURST ANALYSIS MACRO PLOTS
# --------------------------------------------------------
class BAMacPlots
    def plot_unit_participation(grouped_participation, ordered_short_labels, control_subjects):
        """
        Plots violin plots for unit burst participation data.

        Args:
        - grouped_participation: dict {label: [participation fractions]}
        - ordered_short_labels: list of dataset labels in desired x-axis order
        - control_subjects: set of subject IDs considered control
        """

        plot_labels = []
        plot_data = []
        positions = []

        for idx, short_label in enumerate(ordered_short_labels):
            subject = short_label.split("s")[-1]
            subject_label = f"s{subject}"
            group = "CONTROL" if subject_label in control_subjects else "TREATED"
            label = f"{group}_{short_label}"
            values = grouped_participation.get(label)

            if values and len(values) > 0:
                plot_labels.append(label)
                plot_data.append(values)
                positions.append(idx)
            else:
                print(f"[NOTE] No data for {label}")

        if not plot_data:
            print("No valid data to plot.")
            return

        plt.figure(figsize=(16, 6))
        plt.violinplot(plot_data, positions=positions, showmedians=True, widths=0.7)
        plt.xticks(ticks=positions, labels=plot_labels, rotation=45)
        plt.ylabel("Burst Participation Rate (≥2 spikes per burst)")
        plt.title("Per-Unit Burst Participation Across Datasets")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_zscore_distributions(zscore_distributions, ordered_short_labels, control_subjects):
        """
        Plot violin distributions of Spearman z-scores grouped by condition and dataset.

        Args:
        zscore_distributions : dict
            Mapping of {label: list of z-scores} for each dataset-condition pair.
        ordered_short_labels : list of str
            Labels defining the x-axis order for datasets (e.g., "d-6s1", "d1s1", ...).
        control_subjects : set
            Set of subject IDs (e.g., {"s1", "s2"}) considered CONTROL.

        Notes:
        - Generates a violin plot where each dataset-condition pair's z-scores are shown.
        - Adds appropriate axis labels and titles.
        """

        plot_labels = []
        plot_data = []
        positions = []

        # Collect data in specified order
        for idx, short_label in enumerate(ordered_short_labels):
            subject = short_label.split("s")[-1]
            subject_label = f"s{subject}"
            group = "CONTROL" if subject_label in control_subjects else "TREATED"
            full_label = f"{group}_{short_label}"
            data = zscore_distributions.get(full_label, [])

            if data:
                plot_labels.append(full_label)
                plot_data.append(data)
                positions.append(idx)
            else:
                print(f"[SKIP] No z-scores for: {full_label}")

        # Plot violin distribution
        if not plot_data:
            print("No data available to plot.")
            return

        plt.figure(figsize=(16, 6))
        plt.violinplot(dataset=plot_data, positions=positions, showmedians=True, widths=0.7)
        plt.xticks(ticks=positions, labels=plot_labels, rotation=45)
        plt.ylabel("Spearman z-score (Rank Order Consistency)")
        plt.title("Burst Peak Rank Correlation Across Conditions")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_burst_similarity_matrix(sim_matrix, title="Burst Similarity (Cosine)"):
        """
        Plots the burst-to-burst similarity matrix.

        Args:
            sim_matrix (np.ndarray): Square matrix [n_bursts, n_bursts] of cosine similarity values.
            title (str): Plot title.
        """
        if sim_matrix is None:
            print("[INFO] Nothing to plot. Similarity matrix is None.")
            return

        plt.figure(figsize=(6, 5))
        plt.imshow(sim_matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Cosine similarity")
        plt.title(title)
        plt.xlabel("Burst Index")
        plt.ylabel("Burst Index")
        plt.xticks(np.arange(sim_matrix.shape[0]))
        plt.yticks(np.arange(sim_matrix.shape[1]))
        plt.tight_layout()
        plt.show()

    def plot_dual_similarity_matrices(results_dict):
        """
        Plots similarity matrix and corresponding average IFR waveform
        for one or more datasets.

        Args:
            results_dict (dict): 
                {
                    dataset_key: {
                        "sim_matrix": np.ndarray,
                        "burst_stack": np.ndarray,
                        "avg_matrix": np.ndarray,
                        "peak_times": np.ndarray
                    },
                    ...
                }
        """
        n_datasets = len(results_dict)
        fig, axes = plt.subplots(
            n_datasets, 2, 
            figsize=(10, 5 * n_datasets),
            squeeze=False
        )

        for i, (key, res) in enumerate(results_dict.items()):
            sim_matrix = res["sim_matrix"]
            avg_matrix = res["avg_matrix"]
            burst_stack = res["burst_stack"]
            peak_times = res["peak_times"]

            # ---- Plot 1: Similarity Matrix ----
            ax1 = axes[i, 0]
            im = ax1.imshow(sim_matrix, cmap='viridis', aspect='auto')
            ax1.set_title(f"{key} - Burst Similarity (Cosine)")
            ax1.set_xlabel("Burst Index")
            ax1.set_ylabel("Burst Index")
            
            # Optional: annotate burst times on ticks if available
            if peak_times is not None and len(peak_times) == sim_matrix.shape[0]:
                labels = [f"{t:.2f}s" for t in peak_times]
                ax1.set_xticks(range(len(labels)))
                ax1.set_yticks(range(len(labels)))
                ax1.set_xticklabels(labels, rotation=45, fontsize=8)
                ax1.set_yticklabels(labels, fontsize=8)

            fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

            # ---- Plot 2: Average IFR waveform ----
            ax2 = axes[i, 1]
            if avg_matrix is not None:
                time_axis = np.arange(avg_matrix.shape[1]) * 0.01  # 10 ms bins default
                ax2.imshow(avg_matrix, aspect='auto', cmap='hot', origin='lower')
                ax2.set_title(f"{key} - Mean IFR (Aligned Bursts)")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Units (sorted)")
                ax2.set_xticks(np.linspace(0, avg_matrix.shape[1]-1, 5))
                ax2.set_xticklabels([f"{t:.2f}" for t in np.linspace(time_axis[0], time_axis[-1], 5)])
            else:
                ax2.text(0.5, 0.5, "No average IFR data", ha='center', va='center')

        plt.tight_layout()
        plt.show()

    def plot_burst_aligned_ifr_matrix(avg_matrix, peak_times=None, title="Average Burst-Aligned IFR"):
        """
        Plots the average IFR matrix aligned to burst peaks.

        Args:
            avg_matrix (np.ndarray): [n_units, n_bins] average IFR across bursts.
            peak_times (np.ndarray, optional): Detected burst peak times (s).
            title (str): Title of the plot.
        """
        if avg_matrix is None:
            print("No data to plot.")
            return

        plt.figure(figsize=(12, 6))
        plt.imshow(
            avg_matrix,
            aspect='auto',
            origin='lower',
            cmap='hot'
        )
        plt.colorbar(label='Instantaneous FR (Hz)')
        plt.xlabel("Time bins (relative to burst peak)")
        plt.ylabel("Neuron (sorted)")
        plt.title(title)

        if peak_times is not None:
            plt.axvline(avg_matrix.shape[1] // 2, color='red', linestyle='--', label='Burst Peak')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_peak_centered_ifr_segments(stack, time_axis, title="Peak-Centered IFR Segments"):
        """
        Plots multiple IFR segments aligned to burst peaks.

        Args:
            stack (np.ndarray): [n_bursts, n_units, n_bins]
            time_axis (np.ndarray): Time axis for each segment (centered on 0).
            title (str): Plot title.
        """
        if stack is None or time_axis is None:
            print("No data to plot.")
            return

        plt.figure(figsize=(12, 6))
        mean_trace = np.mean(np.sum(stack, axis=1), axis=0)  # sum across units, average across bursts
        std_trace = np.std(np.sum(stack, axis=1), axis=0)

        for burst in stack:
            plt.plot(time_axis, np.sum(burst, axis=0), color="gray", alpha=0.3)

        plt.plot(time_axis, mean_trace, color="black", linewidth=2, label="Mean FR")
        plt.plot(time_axis, mean_trace + std_trace, linestyle="--", color="black", alpha=0.7, label="+1 STD")
        plt.plot(time_axis, mean_trace - std_trace, linestyle="--", color="black", alpha=0.7, label="-1 STD")
        plt.axvline(0, color='red', linestyle='--', label='Burst Peak')

        plt.xlabel("Time relative to burst peak (s)")
        plt.ylabel("Population FR (Hz)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_population_ifr_with_bursts(rate_matrix, times, bursts, title="Population IFR with Burst Windows"):
        """
        Plots IFR over time with burst windows highlighted.

        Args:
            rate_matrix (np.ndarray): [n_bins, n_units] IFR matrix.
            times (np.ndarray): Time axis corresponding to rate_matrix.
            bursts (list of tuple): Start and end indices for bursts.
            title (str): Plot title.
        """
        if rate_matrix is None or times is None:
            print("No data to plot.")
            return

        population_rate = np.sum(rate_matrix, axis=1)

        plt.figure(figsize=(14, 5))
        plt.plot(times, population_rate, label="Population IFR", color="black")

        for start, end in bursts:
            plt.axvspan(times[start], times[end], color='red', alpha=0.3)

        plt.xlabel("Time (s)")
        plt.ylabel("Population FR (Hz)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_pairwise_ifr_correlation(corr_matrix, title="Pairwise IFR Correlation Matrix"):
        """
        Plots a heatmap of pairwise IFR correlations between units.

        Args:
            corr_matrix (np.ndarray): Square matrix of unit-to-unit correlation values.
            title (str): Plot title.
        """
        if corr_matrix is None:
            print("No data to plot.")
            return

        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr_matrix, cmap="hot", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, label="Pearson r")
        plt.title(title)
        plt.xlabel("Unit Index")
        plt.ylabel("Unit Index")
        plt.tight_layout()
        plt.show()

    def plot_relative_unit_peak_times(rel_peaks, title="Unit Peak Times Relative to Burst Peaks"):
    """
    Plots histogram of unit peak times relative to population burst peaks.

    Args:
        rel_peaks (list): List of peak time offsets (ms).
        title (str): Plot title.
    """
    if not rel_peaks:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(rel_peaks, bins=40, color='gray', edgecolor='black', alpha=0.8)
    plt.axvline(0, color='red', linestyle='--', label='Burst Peak (0 ms)')
    plt.xlabel("Time Relative to Burst Peak (ms)")
    plt.ylabel("Number of Units")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
#           BURST ANALYSIS MICRO PLOTS
# --------------------------------------------------------
class BAMicPlots
    def plot_sorted_ifr_for_burst(sorted_burst_matrix, sorted_units, burst_time_axis, dataset_key=None, save_path=None):
        """
        Visualizes firing activity during a specific burst as a heatmap of IFR values.

        This function takes the IFR submatrix extracted and sorted for one burst
        and plots it as a time vs unit heatmap, where units are ordered by the time
        of their peak firing activity.

        Args:
            sorted_burst_matrix (np.ndarray):
                IFR data of shape [time_bins, sorted_units] for the burst.
            sorted_units (np.ndarray):
                Array of unit indices corresponding to the columns of the IFR matrix.
            burst_time_axis (np.ndarray):
                Time values (in seconds) relative to burst start for each bin.
            dataset_key (str, optional):
                Identifier for the dataset, displayed in the plot title.
            save_path (str, optional):
                If provided, saves the figure to this file path instead of displaying it.

        Output:
            fig (matplotlib.figure.Figure):
                Figure handle for the generated plot.
        """
        # Handle edge case where no data is available
        if sorted_burst_matrix is None or sorted_units is None or burst_time_axis is None:
            print("No burst activity to plot.")
            return None

        # Transpose matrix for plotting (units on y-axis, time on x-axis)
        data_to_plot = sorted_burst_matrix.T

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot IFR heatmap
        im = ax.imshow(
            data_to_plot,
            aspect='auto',
            origin='lower',
            extent=[burst_time_axis[0], burst_time_axis[-1], 0, len(sorted_units)],
            cmap='hot'
        )

        # Add colorbar for firing rate intensity
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Instantaneous Firing Rate (Hz)")

        # Label axes
        ax.set_xlabel("Time relative to burst start (s)")
        ax.set_ylabel("Units (sorted by peak firing time)")

        # Optional dataset title
        if dataset_key:
            ax.set_title(f"Burst IFR Heatmap - {dataset_key}")
        else:
            ax.set_title("Burst IFR Heatmap (Sorted Units)")

        # Adjust layout
        plt.tight_layout()

        # Save or display the figure
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()

        return fig