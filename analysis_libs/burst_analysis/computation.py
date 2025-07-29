import pandas as pd
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d   
import warnings                   
from scipy.stats import spearmanr 
from burst_analysis.plotting import plot_raster
from burst_analysis.plotting import plot_population_rate
from burst_analysis.detection import BurstDetection


class BurstAnalysisMacro:
    def __init__(self, trains, fs=10000, config=None):
        self.trains = trains # each element is a list of lists (array) of spike times (in seconds) for a single neuron.
        self.fs = fs # int (default = 10000 Hz)
        self.config = config or {} # dict of config params 

    def compute_unit_burst_participation(self, burst_windows):
        """
        Computes fraction of bursts in which each neuron participates (≥2 spikes per burst).

        Params:
            burst_windows : list of start & end times [tuple(float, float)] for each detected burst in seconds

        Outputs: 
            participation_fractions : list of float where each entry corresponds to a single neuron's fraction of bursts
        """
        # Skip computation if no bursts were detected
        if not burst_windows:
            return []

        # Convert burst window times to milliseconds
        bursts_ms = [(start * 1000, end * 1000) for start, end in burst_windows]

        # Initialize list to store participation values for all neurons
        participation_fractions = []

        # Iterate through each neuron's spike train
        for spikes_s in self.trains:
            # Convert spike times from seconds to milliseconds
            spikes_ms = np.asarray(spikes_s) * 1000

            # Count how many bursts this neuron participated in (≥2 spikes)
            burst_counts = sum(
                ((spikes_ms >= start) & (spikes_ms <= end)).sum() >= 2
                for start, end in bursts_ms
            )

            # Compute fraction of total bursts that had sufficient spikes
            fraction = burst_counts / len(bursts_ms)
            participation_fractions.append(fraction)

        return participation_fractions

    def compute_instantaneous_firing_rate(trains, duration_ms=None, sigma=50):
        """
        Compute smoothed instantaneous firing rates (IFR) for multiple spike trains.

        Parameters
        ----------
        trains : list of arrays
            Each element is an array of spike times (in ms) for a single unit.
        duration_ms : int or None, optional
            The total duration of the time window to compute the IFR matrix over.
            If None, it is set to the maximum spike time across all units + 1.
        sigma : float, optional
            Standard deviation of the Gaussian kernel (in ms) for smoothing.

        Returns
        -------
        rate_mat : ndarray
            A 2D array of shape (duration_ms, n_units) where each column contains
            the smoothed IFR for that unit over time.

        Notes
        -----
        - Combines spike trains from multiple units into a firing rate matrix.
        - Each spike train is converted into instantaneous firing rate based on
        interspike intervals (ISI) and then smoothed with a Gaussian kernel.
        - Units with fewer than 2 spikes are skipped since ISI cannot be computed.
        """

        # Combine all spike trains into a single array to determine the overall maximum spike time
        all_spikes = np.hstack(trains)   # Concatenate all spike arrays into one long array
        all_spikes = all_spikes[~np.isnan(all_spikes)]   # Remove any NaN values from the spike data

        # If no spikes exist across all units, return a zero matrix with 1 time step and one column per unit
        if len(all_spikes) == 0:
            return np.zeros((1, len(trains)))

        # If no duration is provided, set it to the maximum spike time (rounded to int) + 1 millisecond
        if duration_ms is None:
            duration_ms = int(np.nanmax(all_spikes)) + 1

        # Determine how many units (spike trains) we are analyzing
        n_units = len(trains)

        # Initialize the IFR matrix with zeros; rows = time points (ms), columns = units
        rate_mat = np.zeros((duration_ms, n_units))

        # Loop through each unit's spike train to compute its instantaneous firing rate
        for unit, spk_times in enumerate(trains):
            spk_times = np.asarray(spk_times)   # Convert spike times to NumPy array
            if len(spk_times) < 2:   # Skip this unit if fewer than 2 spikes (cannot compute ISI)
                continue
            spk_times = spk_times.astype(int)   # Convert spike times to integer milliseconds

            # Compute interspike intervals (ISI) as differences between consecutive spikes
            isi = np.diff(spk_times).astype(float)

            # Insert NaN at the first position since no ISI exists before the first spike
            isi = np.insert(isi, 0, np.nan)

            # Convert ISIs to firing rate (1 / ISI), avoiding division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                isi_rate = np.where(isi == 0, np.nan, 1.0 / isi)

            # Initialize an array for this unit's IFR over the entire time window
            series = np.zeros(duration_ms)

            # For each consecutive spike pair, assign the ISI rate to all time bins between them
            for i in range(1, len(spk_times)):
                start = spk_times[i - 1]   # Start time of this ISI interval
                end = spk_times[i]         # End time of this ISI interval
                if end >= duration_ms:     # If the end exceeds the recording window, stop filling
                    break
                series[start:end] = isi_rate[i]   # Assign firing rate for this time range

            # Smooth the IFR time series using a Gaussian kernel
            # Multiply by 1000 to convert rate from spikes/ms to spikes/sec
            smoothed = 1000 * gaussian_filter1d(series, sigma=sigma)

            # Store the smoothed IFR in the appropriate column of the result matrix
            rate_mat[:, unit] = smoothed

        # Return the completed instantaneous firing rate matrix
        return rate_mat

    def compute_rank_corr_and_zscores(peak_time_matrix, num_shuffles=100, min_units=3):
        """
        Compute Spearman rank-order correlation and z-scores between bursts
        based on their peak firing times.

        Parameters
        ----------
        peak_time_matrix : ndarray
            2D array of shape (n_bursts, n_units), where each entry is the peak
            firing time (in ms) for a given burst and unit. NaNs indicate no data.
        num_shuffles : int, optional
            Number of shuffles to compute the null distribution of correlations.
        min_units : int, optional
            Minimum number of non-NaN units required per burst to include it.

        Returns
        -------
        rho_matrix : ndarray
            Spearman rank correlation matrix (bursts x bursts).
        zscore_matrix : ndarray
            Z-score matrix computed against shuffled null distributions.

        Notes
        -----
        - The input is first filtered to remove bursts with insufficient data.
        - Spearman correlation is computed across bursts (rows), comparing
        their rank orders of unit peak times.
        - A null distribution of correlations is created by shuffling unit
        peak times within bursts.
        - Z-scores indicate the deviation of observed correlations from the null.
        """

        # Identify which bursts (rows) have at least 'min_units' valid (non-NaN) values
        valid_bursts = np.sum(~np.isnan(peak_time_matrix), axis=1) >= min_units

        # Filter out bursts with insufficient valid data
        peak_time_matrix = peak_time_matrix[valid_bursts, :]

        # If fewer than 2 bursts remain after filtering, correlation is not possible
        if peak_time_matrix.shape[0] < 2:
            print("[SKIP] Not enough valid bursts for correlation.")
            return None, None

        # Compute the Spearman correlation matrix across bursts (comparing their rank orders of peak times)
        with warnings.catch_warnings():                   # Catch potential scipy warnings about NaNs
            warnings.simplefilter("ignore", category=UserWarning)  # Ignore user warnings temporarily
            rho_matrix, _ = spearmanr(peak_time_matrix, axis=1, nan_policy='omit') 
            # 'axis=1' computes correlation between rows (bursts)
            # 'nan_policy=omit' ensures NaNs are ignored during correlation

        # Initialize a list to hold shuffled correlation matrices for the null distribution
        shuffled_rhos = []

        # Perform shuffling multiple times to build the null distribution
        for _ in range(num_shuffles):
            shuffled = np.empty_like(peak_time_matrix)  # Create an empty array to store shuffled data

            # Shuffle the peak times within each burst independently
            for i in range(peak_time_matrix.shape[0]):
                row = peak_time_matrix[i]               # Extract the current burst row
                valid = ~np.isnan(row)                  # Identify valid (non-NaN) entries
                shuffled_row = row.copy()               # Copy row values to preserve original
                shuffled_row[valid] = np.random.permutation(row[valid])  
                # Randomly permute valid peak times to destroy rank order
                shuffled[i] = shuffled_row              # Store the shuffled version

            # Compute Spearman correlation on the shuffled data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                shuff_rho, _ = spearmanr(shuffled, axis=1, nan_policy='omit')
                # Each shuffle generates a new correlation matrix
            shuffled_rhos.append(shuff_rho)             # Add the shuffled correlation matrix to the list

        # Convert the list of shuffled correlation matrices into a 3D NumPy array
        shuffled_rhos = np.array(shuffled_rhos)

        # Compute the mean correlation across shuffles (null expectation)
        mean_shuff = np.nanmean(shuffled_rhos, axis=0)

        # Compute the standard deviation of correlations across shuffles (null variability)
        std_shuff = np.nanstd(shuffled_rhos, axis=0)

        # Compute z-scores as (observed correlation - mean shuffled correlation) / std deviation
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppress divide-by-zero and invalid warnings
            zscore_matrix = (rho_matrix - mean_shuff) / std_shuff

        # Replace infinite or undefined z-scores with NaN to avoid invalid results
        zscore_matrix[~np.isfinite(zscore_matrix)] = np.nan

        # Return both the observed Spearman correlation matrix and the z-score matrix
        return rho_matrix, zscore_matrix

    def build_peak_times_matrix(self, trains, burst_windows, threshold=0.5):
        """
        Construct a matrix of peak firing times for units across bursts.

        Parameters
        ----------
        trains : list of arrays
            Each element contains spike times (in seconds) for a single unit.
        burst_windows : list of tuples
            Each tuple (start, end) specifies a burst window in seconds.
        threshold : float, optional
            Minimum fraction of bursts in which a unit must fire at least 2 spikes
            to be considered valid.

        Returns
        -------
        peak_times_matrix : ndarray
            2D array of shape (n_bursts, n_units) containing the time (ms)
            of peak firing rate for each burst-unit combination.
            NaNs are filled where no peak is detected.

        Notes
        -----
        - Units firing less than 2 spikes in fewer than (threshold * total_bursts)
        bursts are discarded.
        - Burst and spike times are converted to milliseconds.
        - The output matrix is transposed to have bursts as rows and units as columns.
        """

        # Convert all burst windows from seconds to milliseconds
        bursts_ms = [(s * 1000, e * 1000) for s, e in burst_windows]

        # Count how many bursts exist (used for thresholding valid units)
        n_bursts = len(bursts_ms)

        # Convert all spike trains from seconds to milliseconds
        trains_ms = [np.asarray(spikes) * 1000 for spikes in trains]

        # Initialize a list to hold indices of valid units that meet the threshold
        valid_units = []

        # Iterate over each unit and check if it fires in enough bursts
        for i, spikes in enumerate(trains_ms):
            # Count how many bursts have at least 2 spikes for this unit
            count = sum(((spikes >= s) & (spikes <= e)).sum() >= 2 for s, e in bursts_ms)
            # Check if this unit meets the required threshold fraction of bursts
            if count / n_bursts >= threshold:
                valid_units.append(i)

        # If fewer than 3 units are valid, return None because analysis requires at least 3
        if len(valid_units) < 3:
            return None

        # Initialize a matrix (units x bursts) filled with NaNs to store peak firing times
        peak_times_matrix = np.full((len(valid_units), n_bursts), np.nan)

        # Iterate over each burst to compute unit-specific peak times
        for b_idx, (s_ms, e_ms) in enumerate(bursts_ms):
            # Calculate burst duration in ms
            duration_ms = int(e_ms - s_ms)

            # Skip bursts that are too short (<10 ms) for meaningful rate computation
            if duration_ms < 10:
                continue

            # Extract spike trains for valid units and align them to the burst start time
            burst_trains = [trains_ms[j] - s_ms for j in valid_units]

            # Compute instantaneous firing rate (IFR) for each unit within this burst
            rate = self.compute_instantaneous_firing_rate(burst_trains, duration_ms=duration_ms)

            # Find the time point of maximum firing rate for each unit
            for u_idx in range(len(valid_units)):
                # Get the index of the maximum IFR value for this unit
                peak_idx = np.argmax(rate[:, u_idx])
                # Convert index back to an absolute time in ms relative to entire recording
                peak_times_matrix[u_idx, b_idx] = s_ms + peak_idx

        # Transpose the matrix so that bursts are rows and units are columns
        return peak_times_matrix.T


    def compute_burst_aligned_ifr_matrix(self,
                                         bin_size_ms=10,
                                         square_win_ms=20,
                                         gauss_win_ms=100,
                                         threshold_rms=4,
                                         min_dist_ms=700,
                                         burst_window_s=1.5,
                                         max_bursts=30):
        """
        Extracts instantaneous firing rate (IFR) matrices for multiple bursts,
        aligned around their peak population activity, and returns their average.

        Returns:
            burst_stack (np.ndarray): All aligned/sorted bursts [n_bursts, n_units, n_bins].
            avg_matrix (np.ndarray): Mean IFR matrix across bursts [n_units, n_bins].
            peak_times (np.ndarray): Times of burst peaks (in seconds).
            peak_indices (np.ndarray): Indices into population rate where peaks occurred.

        Parameters:
            burst_window_s (float): Duration of window around each burst (centered on peak), in seconds.
                - Total window is `burst_window_s * 2` centered on peak.
            max_bursts (int): Maximum number of bursts to align and average.

        Config Parameters (from self.config):
            bin_size_ms (int): Binning resolution.
            square_win_ms (int): Width of square smoothing window.
            gauss_win_ms (int): Width of Gaussian smoothing window.
            threshold_rms (float): RMS multiple for burst detection.
            min_dist_ms (int): Minimum time between bursts.

        Method:
            1. Calls `compute_population_rate_and_bursts()` to get burst peak indices.
            2. For each peak:
                - Defines a window before/after the peak.
                - Bins spikes from each neuron in that window.
                - Computes IFR and smooths it.
                - Sorts units by peak IFR time.
            3. Averages sorted matrices across bursts.
        """
        bin_size_ms = self.config.get("bin_size_ms", 10)
        bin_size_s = bin_size_ms / 1000
        square_win_ms = self.config.get("square_win_ms", 20)
        gauss_win_ms = self.config.get("gauss_win_ms", 100)

        # Step 1: Get burst peaks from population rate
        times, _, peaks, _ = self.compute_population_rate_and_bursts()
        if times is None or not len(peaks):
            return None, None, None

        duration = times[-1]
        bin_edges = np.arange(0, duration + bin_size_s, bin_size_s)
        spike_counts, _ = np.histogram(np.hstack(self.trains), bins=bin_edges)

        # Step 2: Smooth population rate (can adjust smoothing kernel here)
        square_kernel = np.ones(int(square_win_ms / bin_size_ms)) / (square_win_ms / bin_size_ms)
        smoothed = np.convolve(spike_counts, square_kernel, mode='same')
        # Optionally add Gaussian smoothing here again

        n_units = len(self.trains)
        half_window_bins = int(burst_window_s / bin_size_s)
        burst_matrices = []

        for peak in peaks[:max_bursts]:
            start_bin = peak - half_window_bins
            end_bin = peak + half_window_bins
            if start_bin < 0 or end_bin > len(bin_edges) - 1:
                continue

            window_start = float(bin_edges[start_bin])
            window_end = float(bin_edges[end_bin])
            n_bins = end_bin - start_bin

            mat = np.zeros((n_units, n_bins))

            # Step 3: Compute smoothed IFR for each unit in window
            for i, spikes in enumerate(self.trains):
                spk_in_win = spikes[(spikes >= window_start) & (spikes < window_end)]
                binned, _ = np.histogram(spk_in_win, bins=n_bins, range=(window_start, window_end))
                rate = binned / bin_size_s
                rate = np.convolve(rate, np.ones(2)/2, mode='same')
                rate = gaussian_filter1d(rate, sigma=gauss_win_ms / bin_size_ms)
                mat[i] = rate

            # Step 4: Sort units by peak firing time
            sort_idx = np.argsort(np.argmax(mat, axis=1))
            mat_sorted = mat[sort_idx]

            burst_matrices.append(mat_sorted)

        # Step 5: Average across bursts
        if burst_matrices:
            burst_stack = np.stack(burst_matrices)  # shape: [n_bursts, n_units, n_bins]
            avg_matrix = np.mean(burst_stack, axis=0)
            return burst_stack, avg_matrix, bin_edges[peaks[:len(burst_matrices)]], peaks[:len(burst_matrices)]
        else:
            return None, None, None, None

    def get_peak_centered_ifr_segments(self, window_s=3.0):
        """
        Extract fixed-duration IFR segments centered on each burst peak (time = 0).

        Parameters:
            window_s (float): Total duration in seconds for each segment (e.g., 3.0 = ±1.5s)

        Returns:
            stack (np.ndarray): Shape (n_bursts, n_units, n_time_bins)
            time_axis (np.ndarray): Real time axis, centered at 0
        """
        rate_matrix = self.compute_ifr_matrix().T  # shape: (n_units, n_time_bins)
        times, _, peaks, _ = self.compute_population_rate_and_bursts()

        bin_size_ms = self.config.get("bin_size_ms", 10)
        bin_size_s = bin_size_ms / 1000.0
        half_window_bins = int((window_s / 2) / bin_size_s)

        aligned_segments = []

        for peak_idx in peaks:
            start = peak_idx - half_window_bins
            end = peak_idx + half_window_bins

            if start < 0 or end > rate_matrix.shape[1]:
                continue  # skip if segment would go out of bounds

            segment = rate_matrix[:, start:end]
            aligned_segments.append(segment)

        if not aligned_segments:
            print("No valid peak-centered segments found.")
            return None, None

        stack = np.stack(aligned_segments)  # shape: (n_bursts, n_units, n_time_bins)
        time_axis = np.linspace(-window_s / 2, window_s / 2, stack.shape[2])

        return stack, time_axis

    def compute_population_ifr_with_bursts(self):
        """
        Convenience method that returns:
        - IFR matrix [time x units]
        - Time axis
        - Detected burst windows
        """
        times, _, _, bursts = self.compute_population_rate_and_bursts()
        rate_matrix = self.compute_ifr_matrix()
        return rate_matrix.T, times, bursts

    def compute_ifr_matrix(self,
                           duration_s=None,
                           bin_size_ms=None,
                           square_win_ms=None,
                           gauss_win_ms=None):
        """
        Computes instantaneous firing rate (IFR) matrix for all units over the full recording.

        Args:
            duration_s (float, optional): Total duration of recording (in seconds).
                If None, inferred from max spike time across all units.
            bin_size_ms (int, optional): Time resolution of bins (e.g., 10 ms).
            square_win_ms (int, optional): Width of square smoothing window.
            gauss_win_ms (int, optional): Width of Gaussian smoothing window.

        Returns:
            rate_mat (np.ndarray): IFR matrix with shape [n_units, n_bins].

        Notes:
            - Each spike train is binned into a shared time base.
            - Two-stage smoothing: square (fast average) + Gaussian (jitter reduction).
        """
        # Use method parameters, or fallback to config defaults
        bin_size_ms = bin_size_ms if bin_size_ms is not None else self.config.get("bin_size_ms", 10)
        square_win_ms = square_win_ms if square_win_ms is not None else self.config.get("square_win_ms", 20)
        gauss_win_ms = gauss_win_ms if gauss_win_ms is not None else self.config.get("gauss_win_ms", 100)

        bin_size_s = bin_size_ms / 1000
        square_win_bins = int(square_win_ms / bin_size_ms)
        gauss_win_bins = gauss_win_ms / bin_size_ms

        if duration_s is None:
            duration_s = max(np.max(t) if len(t) > 0 else 0 for t in self.trains)

        n_bins = int(np.ceil(duration_s / bin_size_s))
        rate_mat = np.zeros((len(self.trains), n_bins))

        for i, spikes in enumerate(self.trains):
            if len(spikes) == 0:
                continue
            binned, _ = np.histogram(spikes, bins=n_bins, range=(0, duration_s))
            rate = binned / bin_size_s
            rate = np.convolve(rate, np.ones(square_win_bins) / square_win_bins, mode='same')
            rate = gaussian_filter1d(rate, sigma=gauss_win_bins)
            rate_mat[i] = rate

        return rate_mat

    def prepare_sorted_ifr_matrix_for_burst(self, rate_matrix, times, burst_idx, burst_windows, min_spikes=2):
        """
        Extracts and sorts unit firing activity during a specific burst.

        Returns:
            sorted_burst_matrix (np.ndarray): IFR [time_bins, sorted_units].
            sorted_units (np.ndarray): Unit indices sorted by peak time.
            burst_time_axis (np.ndarray): Time axis relative to burst start.

        Parameters:
            rate_matrix (np.ndarray): [time_bins, n_units] IFR matrix.
            times (np.ndarray): Time axis corresponding to rate_matrix.
            burst_idx (int): Index of burst to extract (from `burst_windows`).
            burst_windows (List[Tuple[int, int]]): Start/end indices of each burst in time bins.
            min_spikes (int): Minimum number of spikes to include unit.

        Method:
            1. Selects burst time window from global IFR matrix.
            2. Filters units with low firing activity during the burst.
            3. Sorts remaining units by their peak firing time within the window.
        """
        burst_start, burst_end = burst_windows[burst_idx]
        burst_t_start, burst_t_end = times[burst_start], times[burst_end]
        burst_matrix = rate_matrix[burst_start:burst_end, :]  # [time, units]

        firing_units = []
        for i, spks in enumerate(self.trains):
            n_spikes = np.sum((spks >= burst_t_start) & (spks <= burst_t_end))
            if n_spikes >= min_spikes:
                firing_units.append(i)

        if not firing_units:
            return None, None, None

        burst_submatrix = burst_matrix[:, firing_units]  # restrict to active units
        peak_times = np.argmax(burst_submatrix, axis=0)  # peak time for each unit
        sort_idx = np.argsort(peak_times)

        sorted_burst_matrix = burst_submatrix[:, sort_idx]
        sorted_units = np.array(firing_units)[sort_idx]
        burst_time_axis = times[burst_start:burst_end] - burst_t_start

        return sorted_burst_matrix, sorted_units, burst_time_axis

    def get_relative_unit_peak_times(self, rate_matrix, times, bursts, peak_times):
        """
        Computes each unit's firing peak time relative to its burst's population peak.

        Returns:
            rel_peaks (List[float]):
                List of time differences (in ms) between each unit's peak IFR
                and the population burst peak it occurred in.

        Parameters:
            rate_matrix (np.ndarray): Full IFR matrix [n_time_bins, n_units].
            times (np.ndarray): Time axis corresponding to rate_matrix.
            bursts (List[Tuple[int, int]]): Start and end indices of each burst.
            peak_times (np.ndarray): Time values (in seconds) of burst peaks.

        Method:
            1. For each burst:
                - Extract the window of activity for that burst.
                - For each unit, find its IFR peak time in that window.
                - Subtract the population peak time to get a relative offset.
            2. Stores the relative peak time (in milliseconds) for all units that fired.
        """
        rel_peaks = []

        for (start_idx, end_idx), pop_peak in zip(bursts, peak_times):
            window = rate_matrix[start_idx:end_idx, :]  # time window of IFR
            if window.shape[0] == 0:
                continue

            time_window = times[start_idx:end_idx]

            for unit in range(window.shape[1]):
                trace = window[:, unit]
                if np.count_nonzero(trace) == 0:
                    continue

                unit_peak_time = time_window[np.argmax(trace)]
                rel_peaks.append((unit_peak_time - pop_peak) * 1000.0)  # convert to ms

        return rel_peaks

    def compute_pairwise_ifr_correlation(self, rate_matrix, units=None, min_activity=1e-6):
        """
        Computes pairwise Pearson correlation between unit IFRs over time.

        Returns:
            corr_matrix (np.ndarray): Symmetric [n_units, n_units] matrix of correlations.

        Parameters:
            rate_matrix (np.ndarray): IFR matrix [n_time_bins, n_units].
            units (List[int], optional): If given, only compute correlations for these units.
            min_activity (float): Skip units with near-zero activity.

        Method:
            1. Select valid units.
            2. For each pair of units:
                - Compute Pearson correlation between their IFR traces.
                - Skip units with near-zero variance.
        """

        if units is not None:
            rate_matrix = rate_matrix[:, units]
        n_units = rate_matrix.shape[1]
        corr_matrix = np.full((n_units, n_units), np.nan)

        for i in range(n_units):
            for j in range(i, n_units):
                trace_i = rate_matrix[:, i]
                trace_j = rate_matrix[:, j]

                if np.std(trace_i) < min_activity or np.std(trace_j) < min_activity:
                    continue

                r, _ = pearsonr(trace_i, trace_j)
                corr_matrix[i, j] = corr_matrix[j, i] = r

        return corr_matrix

    def compute_burst_similarity_matrix_rms(self, burst_matrices):
        """
        Computes cosine similarity between each pair of bursts.
        For each burst, this method:
            1. Finds the peak index (in the smoothed population rate).
            2. Converts that to a center in bin-space.
            3. Extracts a fixed number of bins before and after that peak.

            win_bins = int(burst_window_s / bin_size_s) # e.g. 150 bins for 1.5s @ 10ms bins on each side of the peak_idx/time
            segment = ifr_matrix[:, peak_idx - win_bins : peak_idx + win_bins]

        The window size around each burst is therefore the same (~1.5 seconds here):
        |<------ 3.0 sec window ------>|
        |<-- 1.5 sec -->|<-- 1.5 -->|
             [ centered at burst peak ]

        It then creates an IFR slice centered at its peak with a shape of (n_units, 2 * win_bins).
        These uniform slices (the same duration, peak times, and number of bins) are then stacked into a 3D matrix
        with shape (n_bursts, n_units, 2 * win_bins).

        Limitations: This is an all-purpose method for a rough pass, but not good for analyzing
                        bursts with widely varying durations or shapes. It does not capture actual burst onset
                        or offset times, only a symmetrical window around the peak.

        Parameters:
            burst_matrices (list[np.ndarray] or np.ndarray):
                List of 2D arrays or a single 3D array [n_bursts, n_units, n_bins].

        Returns:
            np.ndarray: [n_bursts, n_bursts] matrix of cosine similarity.
        """
        if isinstance(burst_matrices, np.ndarray):
            if burst_matrices.ndim == 3:
                burst_matrices = [burst_matrices[i] for i in range(burst_matrices.shape[0])]
            elif burst_matrices.ndim == 1 and all(isinstance(m, np.ndarray) and m.ndim == 2 for m in burst_matrices):
                # Handle object array of 2D arrays
                burst_matrices = list(burst_matrices)
            else:
                raise ValueError(f"Expected 3D array or list of 2D arrays, got shape: {burst_matrices.shape}")

        if not isinstance(burst_matrices, list) or not all(isinstance(m, np.ndarray) and m.ndim == 2 for m in burst_matrices):
            raise ValueError("burst_matrices must be a list of 2D numpy arrays.")

        vecs = [m.flatten() for m in burst_matrices]
        return cosine_similarity(vecs)




# -------------------------------
#           HELPERS
# -------------------------------

def update_parkinsons_exp_metrics(trains_by_condition, current_df=None):
    new_data = pd.DataFrame([
        {"Sample": cond, "Active Units": len(trains)}
        for cond, trains in trains_by_condition.items()
    ])

    if current_df is None:
        return new_data

    missing_cols = [col for col in new_data.columns if col not in current_df.columns]
    for col in missing_cols:
        current_df[col] = None

    missing_cols = [col for col in current_df.columns if col not in new_data.columns]
    for col in missing_cols:
        new_data[col] = None

    combined = pd.concat([current_df, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset="Sample", keep="first")
    combined = combined.sort_values("Sample").reset_index(drop=True)
    return combined

def extract_condition_components(df, condition_col="Sample"):
    df = df.copy()

    def parse_condition(name):
        if not isinstance(name, str):
            return pd.Series({
                "SampleID": None,
                "Age of Organoid": None,
                "Treatment": None,
                "Phase": None,
                "Time of Treatment": None
            })

        parts = name.split("_")
        return pd.Series({
            "SampleID": parts[0] if len(parts) > 0 else None,
            "Age of Organoid": parts[1] if len(parts) > 1 else None,
            "Treatment": parts[2] if len(parts) > 2 else None,
            "Phase": parts[3] if len(parts) > 3 else None,
            "Time of Treatment": parts[4] if len(parts) > 4 else None
        })

    components = df[condition_col].apply(parse_condition)
    df = pd.concat([df, components], axis=1)
    return df

def refresh_metrics_csv(datasets, csv_path):
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = None

    trains = {k: sd.train for k, sd in datasets.items()}
    df = update_parkinsons_exp_metrics(trains, df)
    df = extract_condition_components(df, condition_col="Sample")  
    df.to_csv(csv_path, index=False)
    return df

def summarize_population_burst_metrics(datasets, config=None, time_start=0, time_window=180, plot=False):
    all_metrics = []
    per_dataset = {}

    for key, sd in datasets.items():
        print(f"Processing {key}...")
        analysis = BurstDetection(sd.train, config=config)
        result = analysis.compute_population_rate_and_bursts()

        if result[0] is None:
            print(f"[WARNING] Burst detection failed for '{key}'")
            continue

        times, smoothed, peaks, peak_times, bursts, burst_windows = result
        metrics = analysis.compute_pop_burst_metrics(
            times, smoothed, peaks, bursts,
            time_start=time_start,
            time_window=time_window,
            peak_times=peak_times,
            burst_windows=burst_windows
        )

        if plot:
            plot_raster(sd.train, title=f"{key} - Raster Plot")
            plot_population_rate(times, smoothed, bursts, title=f"{key} - Population Rate")

        if metrics:
            all_metrics.append(metrics)
            per_dataset[key] = metrics

    if not all_metrics:
        print("No datasets produced valid metrics.")
        return None

    def average_stat(stat_key):
        values = [m[stat_key] for m in all_metrics if stat_key in m]
        return {
            "mean": np.mean(values) if values else None,
            "std": np.std(values) if values else None
        }

    summary_keys = list(all_metrics[0].keys())
    summary = {k: average_stat(k) for k in summary_keys if isinstance(all_metrics[0][k], (int, float))}

    return {"overall": summary, "per_dataset": per_dataset}

def append_burst_metrics(datasets, metrics_func, csv_path, config=None):
    """
    Updates the Parkinsons experiment metrics CSV with new burst metrics.

    Parameters:
        datasets (dict): Dictionary of {dataset_key: SpikeDataset}
        metrics_func (callable): Function that computes metrics (e.g., compute_pop_burst_metrics)
        csv_path (str or Path): Path to the metrics CSV file
        config (dict): Optional config to pass to analysis function

    Returns:
        pd.DataFrame: Updated DataFrame with new metrics merged in
    """
    if Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    new_entries = []

    for key, sd in datasets.items():
        detector = BurstDetection(sd.train, fs=sd.metadata.get("fs", 10000), config=config or {})
        times, smoothed, peaks, peak_times, bursts, burst_windows = detector.compute_population_rate_and_bursts()

        if times is None or not bursts:
            print(f"[WARNING] No bursts detected for '{key}'")
            continue

        metrics = detector.compute_pop_burst_metrics(
            times=times,
            smoothed=smoothed,
            peaks=peaks,
            bursts=bursts,
            time_start=config.get("time_start", 0),
            time_window=config.get("time_window", times[-1]),
            peak_times=peak_times,
            burst_windows=burst_windows
        )

        metrics["Sample"] = key
        new_entries.append(metrics)

    # Combine and merge
    if not new_entries:
        print("No metrics computed.")
        return df

    new_df = pd.DataFrame(new_entries)
    df = update_parkinsons_exp_metrics(
        trains_by_condition={row["Sample"]: None for _, row in new_df.iterrows()},
        current_df=df
    )

    # Merge in the new metrics
    df = pd.merge(df, new_df, on="Sample", how="outer")

    df.to_csv(csv_path, index=False)
    print(f"Updated burst metrics written to {csv_path}")
    return df