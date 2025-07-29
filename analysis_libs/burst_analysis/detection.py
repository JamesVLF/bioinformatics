import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import spearmanr
from typing import List
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.signal import find_peaks

"""
BurstDetection has three different detectors. Used together, they can provide a multi-layered view on how subclasses of 
putative neurons interact across time and experimental conditions. Each integrates with a stats 
toolkit for feature extraction and easy visualization via plots.py. 
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class BurstDetection:

    def __init__(self, trains, fs=10000, config=None):
        self.trains = trains
        self.fs = fs
        self.config = config or {}

# -----------------------------------------------------------------
# BURST DETECTION
# ----------------------------------------------------------------
    def compute_population_rate_and_bursts(self):
        """
        Computes a smoothed population firing rate and detects population bursts.
        Stores the following outputs as a list of tuples for each burst:
            times (np.ndarray): Time axis (in seconds), length = number of bins.
            smoothed (np.ndarray): Smoothed population rate over time.
            peaks (np.ndarray): Indices of peak burst times in the rate trace.
            bursts (List[Tuple[int, int]]): Start and end indices of each burst.

        Reverses the loop to match the edge and peak indices back to their absolute times in 'train'
        Outputs all stored tuples above, plus:
            peak_times (List[float]): Peak times in seconds for each detected burst.
            burst_windows (List[Tuple[float, float]]): Start and end times in seconds for each burst.

        Config Parameters for burst detection (from self.config):
            bin_size_ms (int): Time bin width in milliseconds for histogram (default 10 ms).
                - Affects temporal resolution of population rate.
            square_win_ms (int): Width of square window (in ms) for moving average smoothing.
                - Smooths fast fluctuations; combats noise.
            gauss_win_ms (int): Width of Gaussian smoothing window (in ms).
                - Smooths rate trace after square filter for refined shape.
            threshold_rms (float): Threshold multiplier (in RMS units) to define bursts.
                - Bursts are detected where population rate exceeds `threshold_rms * RMS(rate)`.
            min_dist_ms (int): Minimum distance (in ms) between detected burst peaks.
                - Prevents double-counting of close bursts.

        Method:
            (1) Concatenate all spike times into one array.
            (2) Bin spike counts across the whole population (bin_size_ms).
            (3) Smooth spike count using:
                - A square moving average (square_win_ms)
                - A Gaussian filter (gauss_win_ms)
            (4) Compute RMS of the smoothed signal.
            (5) Threshold = `threshold_rms * RMS` (threshold_rms).
            (6) Detect peaks using `scipy.signal.find_peaks`.
            (7) For each peak, walk backward / forward to clip at user-defined burst edges (edge_fraction).
        """

        bin_size_ms = self.config.get("bin_size_ms", 10)
        square_win_ms = self.config.get("square_win_ms", 20)
        gauss_win_ms = self.config.get("gauss_win_ms", 100)
        threshold_rms = self.config.get("threshold_rms", 4)
        min_dist_ms = self.config.get("min_dist_ms", 700)
        min_merge_separation_ms = self.config.get("min_merge_separation_ms", 200)
        bin_size_s = bin_size_ms / 1000.0

        # ---- Combine all spikes from all units into a single 1D array ----
        all_spikes = np.hstack([np.array(t) for t in self.trains if len(t) > 0])
        if all_spikes.size == 0:
            return None, None, None, None, None, None

        # ---- Histogram the population activity ----
        # Define full duration of the spike train (in seconds)
        duration = np.max(all_spikes)

        # Create bin edges for histogram (0 to max time, stepped in bin_size_s)
        bin_edges = np.arange(0, duration + bin_size_s, bin_size_s)

        # Bin all spikes into population spike counts per time bin
        spike_counts, _ = np.histogram(all_spikes, bins=bin_edges)

        # --- Smooth the population firing rate ---

        # Step 1: Square filter (fast smoothing)
        win_square = int(square_win_ms / bin_size_ms)
        smoothed = np.convolve(spike_counts, np.ones(win_square) / win_square, mode="same")

        # Step 2: Gaussian filter smoothing (for overall shape)

        # Gaussian sigma is scaled relative to bin size
        smoothed = gaussian_filter1d(smoothed, sigma=gauss_win_ms / bin_size_ms)

        # ---- Detect bursts using RMS threshold ----

        # Compute RMS of smoothed signal
        rms = np.sqrt(np.mean(smoothed ** 2))

        # Set detection threshold as multiple of RMS
        threshold = threshold_rms * rms

        # Convert minimum distance between peaks from ms → number of bins
        min_dist_bins = int(min_dist_ms / bin_size_ms)

        # ---- Detect peaks in population rate above threshold ----

        # Only one peak is kept within `min_dist_bins` of another
        peaks, _ = find_peaks(smoothed, height=threshold, distance=min_dist_bins)

        # ---- Store as lists of tuples ----
        bursts = []
        peak_times = []
        burst_windows = []

        # ---- Define burst start/end around each peak ----
        edge_fraction = self.config.get("burst_edge_fraction", 0.1)

        # ---- Construct time axis using one time bin per spike_count (len = num bins)----
        times = bin_edges[:-1]  # Time for each bin center

        for peak in peaks:
            # Peak value during 'time' segment
            peak_val = smoothed[peak]

            # Threshold for burst start/end is read from self.configs for fine-tuning
            thresh_edge = edge_fraction * peak_val

            # Initialize start and end indices to peak index
            start, end = peak, peak

            # Walk backward to find burst start where rate drops below edge threshold
            while start > 0 and smoothed[start] > thresh_edge:
                start -= 1

            # Walk forward to find burst end where rate drops below edge threshold
            while end < len(smoothed) and smoothed[end] > thresh_edge:
                end += 1

            # Clamp 'end' index to stay within bounds of the times array
            if end >= len(times):
                end = len(times) - 1

            # symmetric trimming so bursts have consistent durations around peak (not sure if this is helpful yet)
            #lead = peak - start
            #lag = end - peak
            #if lag > lead:
            #    end = peak + lead
            #if end >= len(times):
            #    end = len(times) - 1

            # Store (start_idx, end_idx) of burst
            bursts.append((start, end))

            # Match peak index back to absolute time in seconds
            peak_times.append(times[peak])

        # ---- Merge close bursts (needed so the very wide bursts in treated data don't get double counted) ----
        def merge_bursts(burst_list, min_sep_bins):
            if not burst_list:
                return []
            merged = [burst_list[0]]
            for start, end in burst_list[1:]:
                prev_start, prev_end = merged[-1]
                if start - prev_end <= min_sep_bins:
                    merged[-1] = (prev_start, max(prev_end, end))
                else:
                    merged.append((start, end))
            return merged

        min_merge_bins = int(min_merge_separation_ms / bin_size_ms)
        bursts = merge_bursts(bursts, min_merge_bins)

        # Match start and end indices back to absolute times and compute burst windows
        for start, end in bursts:
            if end >= len(times):
                end = len(times) - 1
            try:
                burst_windows.append((times[start], times[end]))
            except IndexError:
                print(f"[ERROR] Skipping burst at peak={peak} (start={start}, end={end}), times.shape={times.shape}")
                continue

        # Return all lists and arrays for downstream computations
        return times, smoothed, peaks, peak_times, bursts, burst_windows

    def compute_pop_burst_metrics(
            self,
            times,
            smoothed,
            peaks,
            bursts,
            time_start,
            time_window,
            peak_times=None,
            burst_windows=None
    ):
        """
        Compute metrics from population bursts and indices (the outputs of compute_population_rate_and_bursts).
        Works with RMS-based methods (needs: times, smoothed, peaks, bursts) for these proportionality stats.
        Expanding to compute all below metrics using real numbers from raw spike trains returned as peak_times and burst_windows.
        """
        bin_size_s = self.config.get("bin_size_ms", 10) / 1000.0
        spike_trains = self.trains
        n_neurons = len(spike_trains)
        total_spikes = sum(len(t) for t in spike_trains)
        duration_s = times[-1] if times is not None else 0

        # Core metrics using indices and binning of original trains
        n_bursts = len(bursts)
        durations = [(e - s) * bin_size_s for s, e in bursts]
        mean_dur = np.mean(durations) if durations else 0
        std_dur = np.std(durations) if durations else 0
        burst_rate = n_bursts / (duration_s / 60) if duration_s > 0 else 0
        mean_rate_per_neuron = (total_spikes / n_neurons / duration_s) if duration_s > 0 else 0
        mean_ibi = np.mean(np.diff([times[p] for p in peaks])) if len(peaks) > 1 else 0

        # Burst stats within window using indices
        window_bursts = [(s, e) for s, e in bursts if times[s] >= time_start and times[e] <= time_start + time_window]
        n_pop_bursts_window = len(window_bursts)

        # Peak FR per neuron in time window
        unit_peak_rates = []
        for train in spike_trains:
            spikes = np.array(train)
            spikes_in_win = spikes[(spikes >= time_start) & (spikes <= time_start + time_window)]
            if spikes_in_win.size > 0:
                hist, _ = np.histogram(spikes_in_win, bins=int(time_window / bin_size_s))
                unit_peak_rates.append(np.max(hist) / bin_size_s)
        avg_peakFR_unit_window = np.mean(unit_peak_rates) if unit_peak_rates else 0

        # Zero-aligned peak analysis using indices
        peak_vals, width_left, width_right, burst_durations_0 = [], [], [], []
        for peak in peaks:
            peak_time = times[peak]
            if time_start <= peak_time <= time_start + time_window:
                peak_vals.append(smoothed[peak])
                for s, e in bursts:
                    if s <= peak <= e:
                        burst_durations_0.append((e - s) * bin_size_s)
                        width_left.append((peak - s) * bin_size_s)
                        width_right.append((e - peak) * bin_size_s)
                        break

        return {
            "n_total_spikes": total_spikes,
            "duration_s": duration_s,
            "n_neurons": n_neurons,
            "mean_rate_per_neuron": mean_rate_per_neuron,
            "n_total_bursts": n_bursts,
            "mean_burst_dur": mean_dur,
            "std_burst_dur": std_dur,
            "burst_rate_per_min": burst_rate,
            "mean_IBI": mean_ibi,
            "n_pop_bursts_window": n_pop_bursts_window,
            "avg_peakFR_per_unit_window": avg_peakFR_unit_window,
            "avg_peakFR_per_unit_bin": avg_peakFR_unit_window,  # same as window here
            "n_bursts_at_time_0": len(peak_vals),
            "mean_peakFR_at_time_0": np.mean(peak_vals) if peak_vals else 0,
            "std_peakFR_at_time_0": np.std(peak_vals) if peak_vals else 0,
            "mean_burst_duration_at_time_0": np.mean(burst_durations_0) if burst_durations_0 else 0,
            "std_burst_duration_at_time_0": np.std(burst_durations_0) if burst_durations_0 else 0,
            "mean_width_lead": np.mean(width_left) if width_left else 0,
            "std_width_lead": np.std(width_left) if width_left else 0,
            "mean_width_lag": np.mean(width_right) if width_right else 0,
            "std_width_lag": np.std(width_right) if width_right else 0,
            "mean_total_width": np.mean([l + r for l, r in zip(width_left, width_right)]) if width_left else 0,

            "peak_times": peak_times,
            "burst_windows": burst_windows
        }

    def detect_dim_population_bursts(self, window_size=2.0, step_size=0.5,
                                      min_active_neurons=10, min_spikes_per_neuron=1):
        """
        Method:
            (1) Detects time windows of low-level but structured population activity.
            (2) Scans spike trains for time windows that have a minimum number of active units (min_active_neurons)
                that fire at least once (or more by changing min_spikes_per_neuron).
            (3) Moves across time in sliding windows (window_size, step_size)

        Args:
            trains (list of np.ndarray): List of spike time arrays, one array per neuron.
            window_size (float): How wide each time slice (sliding window) is (in seconds).
            step_size (float): How far to slide the window over each time (in seconds).
            min_active_neurons (int): Minimum number of different neurons that must spike during the window.
            min_spikes_per_neuron (int): Each of those neurons must fire this many times (default = 1).

        Outputs:
            List of tuples: Each tuple is (start_time, end_time) of detected windows.

        Notes:
            Use this for finding and drilling down into subtle coordinated firing events that get washed out with
            rms method. This method is ideal for perturbation assays or studying
            drugs that diminish coordinated network activity.
            Complementary to RMS- and ISI-based methods, and all are plugged into run_detection().
        """

        # ---- Combine all spikes from all units into a single 1D array and assign to all_spikes ----
        all_spikes = [t for t in self.trains if len(t) > 0]

        # ---- Remove units with zero spikes to avoid passing empty arrays (np.hstack(all_spikes)) to np.max(...) ----
        if not all_spikes:
            return [] # Return empty list if no spikes were found

        # ---- Merge all spike times into a 1D array and find the max spike time (end of recording) ----
        duration_s = np.max(np.hstack(all_spikes))

        # Store list of windows assigned to active_windows
        active_windows = []

        # Begin scanning when recording begins
        current_start = 0.0

        # Continue running until the window size is greater than remaining recording time
        while current_start + window_size <= duration_s:

            # Define the end time of the current scanning window by adding current start to the window size
            current_end = current_start + window_size

            # Counts the raw numbers of units that fired at least once within the user-defined window
            active_neuron_count = 0

            # Iterate over each neuron's spike train
            for spike_times in self.trains:

                # Make sure that train is a numpy array
                spike_times = np.array(spike_times)

                # Count how many spikes from the neuron are inside the window
                count = np.sum((spike_times >= current_start) & (spike_times < current_end))

                # And if the neuron fired enough times...
                if count >= min_spikes_per_neuron:

                    # Count it as an active unit
                    active_neuron_count += 1

            # And also, if enough different neurons were active in the window...
            if active_neuron_count >= min_active_neurons:

                # Save that window's time range.
                active_windows.append((round(current_start, 2), round(current_end, 2)))

            # Slide the window forward by one step size and restart the loop
            current_start += step_size

        return active_windows

    def extract_dim_burst_metrics(self, bursts, time_start, time_window):
        """
        Extracts relevant metrics from dim population burst detection output.

        Args:
            bursts (List[Tuple[float, float]]): List of dim activity windows (start, end in seconds).
            time_start (float): Start of user-defined analysis window.
            time_window (float): Duration of analysis window (in seconds).

        Outputs:
            dict of metrics
        """
        spike_trains = self.trains
        n_neurons = len(spike_trains)
        total_spikes = sum(len(t) for t in spike_trains)
        duration_s = max(np.max(np.array(t)) for t in spike_trains if len(t) > 0)

        durations = [end - start for start, end in bursts]
        window_end = time_start + time_window

        # Bursts inside the current analysis window
        bursts_in_window = [b for b in bursts if time_start <= b[0] <= window_end and b[1] <= window_end]

        return {
            "n_total_spikes": total_spikes,
            "duration_s": duration_s,
            "n_neurons": n_neurons,
            "mean_rate_per_neuron": (total_spikes / n_neurons / duration_s) if duration_s > 0 else 0,

            "n_total_bursts": len(bursts),
            "mean_burst_dur": np.mean(durations) if durations else 0,
            "std_burst_dur": np.std(durations) if durations else 0,
            "burst_rate_per_min": len(bursts) / (duration_s / 60) if duration_s > 0 else 0,

            "n_pop_bursts_window": len(bursts_in_window),
            "mean_burst_dur_in_window": np.mean([e - s for s, e in bursts_in_window]) if bursts_in_window else 0,

            # Still considering this: % time in bursts (duty cycle)
            "burst_duty_cycle": (sum(durations) / duration_s) if duration_s > 0 else 0
        }

    def detect_isi_bursts(self, isi_threshold_ms=10, min_spikes=3, aggregate=False):
        """
        Detects bursts within individual spike trains based on inter-spike intervals (ISI).

        Args:
            isi_threshold_ms (float): Max ISI (in ms) for spikes to be considered part of a burst.
            min_spikes (int): Minimum number of spikes in a burst.
            aggregate (bool): If True, returns a merged list of bursts across all neurons (sorted by start time).
                              If False, returns a list of burst lists, one per neuron.

        Outputs:
            - If aggregate=False:
                List of lists: Each sublist contains (start_time, end_time) tuples for bursts in that neuron.
            - If aggregate=True:
                List of (start_time, end_time) tuples across all neurons, sorted by start time.
        """
        isi_threshold_s = isi_threshold_ms / 1000.0
        all_bursts = []

        for neuron_spikes in self.trains:
            spikes = np.sort(np.array(neuron_spikes))
            if len(spikes) < min_spikes:
                all_bursts.append([] if not aggregate else [])
                continue

            isis = np.diff(spikes)
            in_burst = isis <= isi_threshold_s
            burst_starts = []
            burst = []

            for i, flag in enumerate(in_burst):
                if flag:
                    if not burst:
                        burst = [spikes[i], spikes[i + 1]]
                    else:
                        burst.append(spikes[i + 1])
                else:
                    if burst and len(burst) >= min_spikes:
                        burst_starts.append((burst[0], burst[-1]))
                    burst = []

            # Handle burst at the end
            if burst and len(burst) >= min_spikes:
                burst_starts.append((burst[0], burst[-1]))

            all_bursts.append(burst_starts)

        if aggregate:
            flat = [b for bursts in all_bursts for b in bursts]
            return sorted(flat, key=lambda x: x[0])
        else:
            return all_bursts

    def extract_isi_burst_metrics(self, bursts, time_start, time_window):
        """
        Extracts metrics from ISI-based neuron-local burst detection (aggregate view).

        Args:
            bursts (List[Tuple[float, float]]): Aggregated list of (start, end) ISI burst windows.
            time_start (float): Start of window of interest.
            time_window (float): Length of window (s).

        Outputs:
            dict of metrics
        """
        spike_trains = self.trains
        n_neurons = len(spike_trains)
        total_spikes = sum(len(t) for t in spike_trains)
        duration_s = max(np.max(np.array(t)) for t in spike_trains if len(t) > 0)

        durations = [end - start for start, end in bursts]
        window_end = time_start + time_window

        bursts_in_window = [b for b in bursts if time_start <= b[0] <= window_end and b[1] <= window_end]

        return {
            "n_total_spikes": total_spikes,
            "duration_s": duration_s,
            "n_neurons": n_neurons,
            "mean_rate_per_neuron": (total_spikes / n_neurons / duration_s) if duration_s > 0 else 0,

            "n_total_bursts": len(bursts),
            "mean_burst_dur": np.mean(durations) if durations else 0,
            "std_burst_dur": np.std(durations) if durations else 0,
            "burst_rate_per_min": len(bursts) / (duration_s / 60) if duration_s > 0 else 0,

            "n_isi_bursts_window": len(bursts_in_window),
            "mean_isi_burst_dur_window": np.mean([e - s for s, e in bursts_in_window]) if bursts_in_window else 0,

            # ISI-specific: how many neurons had at least 1 burst
            "n_bursting_neurons": len([
                train for train in spike_trains
                if any((s >= time_start and e <= window_end) for (s, e) in bursts)
            ])
        }
    

