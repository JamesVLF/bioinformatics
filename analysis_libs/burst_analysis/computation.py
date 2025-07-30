import pandas as pd
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d   
from scipy.signal import correlate
from scipy.signal import fftconvolve
import warnings                   
from scipy.stats import spearmanr, pearsonr 
from burst_analysis.plotting import BurstDetectionPlots, BAMacPlots, BAMicPlots
from burst_analysis.detection import BurstDetection


class BurstAnalysisMacro:
    def __init__(self, trains, fs=10000, config=None):
        self.trains = trains # each element is a list of lists (array) of spike times (in seconds) for a single neuron.
        self.fs = fs # int (default = 10000 Hz)
        self.config = config or {} # dict of config params 

    def compute_unit_burst_participation(self, burst_windows):
        """
        Computes fraction of bursts in which each neuron participates (≥2 spikes per burst).

        Args:
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
        Computes smoothed instantaneous firing rates (IFR) for multiple spike trains.

        Argss:
            trains : list of arrays
                Each element is an array of spike times (in ms) for a single unit.
            duration_ms : int or None, optional
                The total duration of the time window to compute the IFR matrix over.
                If None, it is set to the maximum spike time across all units + 1.
            sigma : float, optional
                Standard deviation of the Gaussian kernel (in ms) for smoothing.

        Outputs:
            rate_mat : ndarray
                A 2D array of shape (duration_ms, n_units) where each column contains
                the smoothed IFR for that unit over time.

        Notes:
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
        Computes Spearman rank-order correlation and z-scores between bursts
        based on their peak firing times.

        Args:
            peak_time_matrix : ndarray
                2D array of shape (n_bursts, n_units), where each entry is the peak
                firing time (in ms) for a given burst and unit. NaNs indicate no data.
            num_shuffles : int, optional
                Number of shuffles to compute the null distribution of correlations.
            min_units : int, optional
                Minimum number of non-NaN units required per burst to include it.

        Outputs:
            rho_matrix : ndarray
                Spearman rank correlation matrix (bursts x bursts).
            zscore_matrix : ndarray
                Z-score matrix computed against shuffled null distributions.

        Notes:
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

        Args:
            trains : list of arrays
                Each element contains spike times (in seconds) for a single unit.
            burst_windows : list of tuples
                Each tuple (start, end) specifies a burst window in seconds.
            threshold : float, optional
                Minimum fraction of bursts in which a unit must fire at least 2 spikes
                to be considered valid.

        Returns:
            peak_times_matrix : ndarray
                2D array of shape (n_bursts, n_units) containing the time (ms)
                of peak firing rate for each burst-unit combination.
                NaNs are filled where no peak is detected.

        Notes:
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

        Method:
            1. Calls `compute_population_rate_and_bursts()` to get burst peak indices.
            2. For each peak:
                - Defines a window before/after the peak.
                - Bins spikes from each neuron in that window.
                - Computes IFR and smooths it.
                - Sorts units by peak IFR time.
            3. Averages sorted matrices across bursts.

        Args:
            burst_window_s (float): Duration of window around each burst (centered on peak), in seconds.
                - Total window is `burst_window_s * 2` centered on peak.
            max_bursts (int): Maximum number of bursts to align and average.

        Configs (from self.config):
            bin_size_ms (int): Binning resolution.
            square_win_ms (int): Width of square smoothing window.
            gauss_win_ms (int): Width of Gaussian smoothing window.
            threshold_rms (float): RMS multiple for burst detection.
            min_dist_ms (int): Minimum time between bursts.

        Outputs:
            burst_stack (np.ndarray): All aligned/sorted bursts [n_bursts, n_units, n_bins].
            avg_matrix (np.ndarray): Mean IFR matrix across bursts [n_units, n_bins].
            peak_times (np.ndarray): Times of burst peaks (in seconds).
            peak_indices (np.ndarray): Indices into population rate where peaks occurred.

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
        Extracts fixed-duration IFR (Instantaneous Firing Rate) segments centered on each 
        detected population burst peak, aligning the time axis such that the peak occurs 
        at time = 0 seconds.

        Args:
            window_s (float): Total duration of each extracted segment, in seconds. 
                            For example, 3.0 will extract a window of ±1.5 seconds 
                            around each burst peak.

        Returns:
            stack (np.ndarray): 3D array of shape (n_bursts, n_units, n_time_bins),
                                where each entry is an IFR segment centered on a burst peak.
            time_axis (np.ndarray): 1D array representing the time axis for each segment, 
                                    aligned such that 0 corresponds to the burst peak.
        """

        # Compute the full IFR matrix for all units and transpose it, so shape is (n_units(rows), n_time_bins(columns))
        rate_matrix = self.compute_ifr_matrix().T

        # Retrieve population firing rate times, additional metadata (ignored here), 
        # detected burst peak indices, and other burst-related info.
        # 'peaks' are the indices in the global IFR time axis where burst peaks occur.
        times, _, peaks, _ = self.compute_population_rate_and_bursts()

        # Get the bin size (temporal resolution) from configuration in milliseconds,
        # and convert it to seconds for consistent calculations.
        bin_size_ms = self.config.get("bin_size_ms", 10)
        bin_size_s = bin_size_ms / 1000.0

        # Find how many time bins = half the window size (± half-window)...define how many bins taken before / after peak
        half_window_bins = int((window_s / 2) / bin_size_s)

        # Initialize a list to store IFR segments extracted around each burst peak
        aligned_segments = []

        # Loop through each detected burst peak index in the population IFR
        for peak_idx in peaks:
            # Define start and end indices of the segment window centered on the peak
            start = peak_idx - half_window_bins
            end = peak_idx + half_window_bins

            # If the window would extend outside the available time bins, skip this peak
            if start < 0 or end > rate_matrix.shape[1]:
                continue

            # Extract the segment for all units across the defined time window
            segment = rate_matrix[:, start:end]

            # Append the segment to the list of aligned segments
            aligned_segments.append(segment)

        # If no valid segments were found (all skipped), return None to indicate failure
        if not aligned_segments:
            print("No valid peak-centered segments found.")
            return None, None

        # Stack all collected segments into a single NumPy array with shape (n_bursts, n_units, n_time_bins)
        stack = np.stack(aligned_segments)

        # Create time axis for segments, evenly spaced between -window/2 and +window/2, with length = # of time bins per segment
        time_axis = np.linspace(-window_s / 2, window_s / 2, stack.shape[2])

        # Return both the aligned IFR segment data and the centered time axis.
        return stack, time_axis

    def compute_population_ifr_with_bursts(self):
        """
        Convenience method that outputs:
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
        Computes the instantaneous firing rate (IFR) matrix for all units over the full
        recording duration. The IFR represents how frequently neurons are firing, 
        computed in fixed time bins and optionally smoothed.

        Args:
            duration_s (float, optional):
                Total duration of the recording in seconds.
                - If None, the duration is inferred from the latest spike time across all units.
            bin_size_ms (int, optional):
                The width of each time bin in milliseconds (e.g., 10 ms bins).
            square_win_ms (int, optional):
                Width of the square smoothing window in milliseconds (applied first).
            gauss_win_ms (int, optional):
                Width of the Gaussian smoothing window in milliseconds (applied second).

        Outputs:
            rate_mat (np.ndarray):
                A 2D array of shape [n_units, n_bins] containing the IFR of each unit
                across time, after binning and smoothing.

        Notes:
            - Step 1: Spike times are binned into a common time base.
            - Step 2: Rates are smoothed using a square kernel and a Gaussian filter
                    to reduce noise and capture general firing patterns.
        """

        #-------- (1) Determine binning and smoothing parameters --------
        # Use passed parameters if provided, otherwise fall back to class config defaults.
        bin_size_ms = bin_size_ms if bin_size_ms is not None else self.config.get("bin_size_ms", 10)
        square_win_ms = square_win_ms if square_win_ms is not None else self.config.get("square_win_ms", 20)
        gauss_win_ms = gauss_win_ms if gauss_win_ms is not None else self.config.get("gauss_win_ms", 100)

        # Convert time units from milliseconds to seconds (for binning)
        bin_size_s = bin_size_ms / 1000

        # Convert smoothing window sizes from milliseconds to number of bins
        square_win_bins = int(square_win_ms / bin_size_ms)
        gauss_win_bins = gauss_win_ms / bin_size_ms  # Gaussian sigma in bins

        # -------- (2) Determine recording duration if not explicitly provided ----------
        # Look for the maximum spike time across all spike trains
        if duration_s is None:
            duration_s = max(np.max(t) if len(t) > 0 else 0 for t in self.trains)

        # ------- (3) Compute the number of bins spanning the recording -----------
        # This ensures that the entire duration is covered with the chosen bin size
        n_bins = int(np.ceil(duration_s / bin_size_s))

        # Initialize an empty IFR matrix with zeros
        # Rows = units, Columns = time bins
        rate_mat = np.zeros((len(self.trains), n_bins))

        # ------- (4) Process each unit’s spike train individually ----------
        for i, spikes in enumerate(self.trains):
            # Skip units with no spikes to avoid computation errors
            if len(spikes) == 0:
                continue

            # Bin the spike times into discrete bins along the total duration ()'binned' = spike counts per bin)
            binned, _ = np.histogram(spikes, bins=n_bins, range=(0, duration_s))

            # Convert spike counts to firing rates (spikes per second)
            rate = binned / bin_size_s

            # Apply square smoothing (moving average) to reduce bin-to-bin variability
            rate = np.convolve(rate, np.ones(square_win_bins) / square_win_bins, mode='same')

            # Apply Gaussian smoothing to further smooth jitter while preserving peak structure
            rate = gaussian_filter1d(rate, sigma=gauss_win_bins)

            # Store the processed rate vector for this unit in the IFR matrix
            rate_mat[i] = rate

        return rate_mat
    

# ----------------------------------------------------------- 
#           LEVEL 3: Single Burst and Unit-level Analysis
# -------------------------------------------------------------

    def prepare_sorted_ifr_matrix_for_burst(self, rate_matrix, times, burst_idx, burst_windows, min_spikes=2):
        """
        Extracts and sorts unit firing activity for a specific population burst.

        This function takes a global instantaneous firing rate (IFR) matrix and isolates
        the activity of neurons during a particular burst, filtering out low-activity
        units and sorting the remaining units by their peak firing time within the burst window.

        Method:
            1. Identify the time window for the selected burst.
            2. Filter units based on the number of spikes they fired during this burst.
            3. Sort the remaining units by the time of their maximum firing rate.

        Args:
            rate_matrix (np.ndarray):
                A 2D array of shape [time_bins, n_units] representing IFR values across all units.
            times (np.ndarray):
                1D array containing the global time axis corresponding to rate_matrix.
            burst_idx (int):
                Index of the target burst within the provided list of burst windows.
            burst_windows (List[Tuple[int, int]]):
                A list of (start_idx, end_idx) pairs representing burst intervals in time bins.
            min_spikes (int):
                Minimum spike count for a unit to be considered active during this burst.

        Returns:
            sorted_burst_matrix (np.ndarray or None):
                IFR submatrix [time_bins, sorted_units] for the selected burst,
                sorted by peak firing time. Returns None if no active units are found.
            sorted_units (np.ndarray or None):
                Array of unit indices included in the burst activity, sorted by peak time.
            burst_time_axis (np.ndarray or None):
                Time axis for this burst, zeroed relative to burst start.
        """

        # ----- (1) Identify start and end indices of the selected burst ---------
        burst_start, burst_end = burst_windows[burst_idx]

        # Get the corresponding real-time values for the burst window
        burst_t_start, burst_t_end = times[burst_start], times[burst_end]

        # Extract only the rows (time bins) corresponding to this burst... shape = [time_bins within burst, n_units]
        burst_matrix = rate_matrix[burst_start:burst_end, :]

        # ------- (2) Determine which units are active enough during this burst -----------
        # Initialize a list to store indices of units meeting the activity threshold
        firing_units = []

        # Loop through each unit's spike times
        for i, spks in enumerate(self.trains):
            # Count spikes that occurred within the burst time window
            n_spikes = np.sum((spks >= burst_t_start) & (spks <= burst_t_end))

            # Keep only units with enough spikes
            if n_spikes >= min_spikes:
                firing_units.append(i)

        # If no units meet the criteria, return None values
        if not firing_units:
            return None, None, None

        # -------- (3) Extract IFR data only for active units ---------
        burst_submatrix = burst_matrix[:, firing_units]  # Shape: [time_bins, active_units]

        # Find time bin of the firing rate peak for each active unit
        peak_times = np.argmax(burst_submatrix, axis=0)

        # Sort units by their peak firing times (ascending)
        sort_idx = np.argsort(peak_times)

        # Apply the sorting order to the IFR submatrix and unit indices
        sorted_burst_matrix = burst_submatrix[:, sort_idx]
        sorted_units = np.array(firing_units)[sort_idx]

        # Create time axis relative to the burst start time (time zero = first time point of the burst)
        burst_time_axis = times[burst_start:burst_end] - burst_t_start

        # Return sorted IFR matrix, unit indices, and relative time axis
        return sorted_burst_matrix, sorted_units, burst_time_axis

    def get_relative_unit_peak_times(self, dataset_key=None, config=None):
        """
        Computes the relative timing of each unit's firing rate peak compared to the 
        detected population burst peak across all bursts in a dataset.

        Method:
            1. Load spike data and initialize the burst analysis object.
            2. Compute the population IFR matrix and detect burst start/end windows.
            3. Compute the timing of each population burst peak.
            4. For each burst:
                - Extract the IFR traces for all units within the burst window.
                - Identify the peak firing time of each active unit.
                - Compute the difference between the unit's peak and the population peak.
            5. Return all differences (ms) as a list.

        Args:
            dataset_key (str, optional):
                Name of the dataset to analyze. If None, defaults to a primary dataset.
            config (dict, optional):
                Configuration parameters passed to BurstAnalysisMacro for IFR and 
                burst detection computations.

        Intermediates:
            rate_matrix: firing rate values for all units [time_bins, n_units]
            times: time axis corresponding to each bin
            bursts: list of (start_idx, end_idx) for each detected burst
            peak_times: array of burst peak times (in seconds)
                
        Outputs:
            rel_peaks (List[float]):
                A list of time offsets (in milliseconds) where:
                    - Each entry corresponds to a single unit in a single burst.
                    - Positive values indicate the unit's firing peak occurred *after* 
                    the population burst peak.
                    - Negative values indicate the unit's firing peak occurred *before* 
                    the population burst peak.

        Notes: 
            This function measures how early or late individual neurons reach their 
            peak firing rate relative to the overall population burst peak, providing 
            insights into temporal firing coordination within bursts.
        """

        # Normalize dataset key (handles cases where it's a NumPy scalar)
        dataset_key = self._normalize_dataset_key(dataset_key)

        # Retrieve spike dataset object
        sd = self.spike_data[dataset_key]

        # Initialize burst analysis class with spike trains and sampling frequency
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        # ----- (1) Compute the population IFR matrix and detect burst time windows --------
        rate_matrix, times, bursts = analysis.compute_population_ifr_with_bursts()

        # ---- (2) Compute detected population burst peak times --------
        _, _, peak_times, _ = analysis.compute_burst_aligned_ifr_matrix()

        # ---- (3) Initialize list to hold relative timing results -------
        rel_peaks = []

        # ---- (4)  Iterate over all bursts and their population peaks -------
        for (start_idx, end_idx), pop_peak in zip(bursts, peak_times):
            # Extract IFR activity window for this burst
            window = rate_matrix[start_idx:end_idx, :]
            if window.shape[0] == 0:
                continue  # Skip empty bursts

            # Extract corresponding time values for this burst window
            time_window = times[start_idx:end_idx]

            # ----- (5) For each unit, find the time of its peak firing activity -------
            for unit in range(window.shape[1]):
                trace = window[:, unit]  # IFR trace for this unit
                if np.count_nonzero(trace) == 0:
                    continue  # Skip silent units with no firing activity

                # Identify the time bin of the firing rate peak for this unit
                unit_peak_time = time_window[np.argmax(trace)]

                # Compute relative timing difference to population peak (convert s → ms)
                rel_peaks.append((unit_peak_time - pop_peak) * 1000.0)

        # Get list of timing offsets
        return rel_peaks
    
    def compute_burst_aligned_ifr_trials(self, burst_windows, burst_peaks,
                                     bin_size=0.001, window=(-0.25, 0.5)):
        n_units = len(self.trains)
        n_bursts = len(burst_windows)
        time_bins = np.arange(window[0], window[1] + bin_size, bin_size)
        n_bins = len(time_bins) - 1

        ifr_trials = {u: np.zeros((n_bursts, n_bins)) for u in range(n_units)}

        for burst_idx, (burst_window, peak_time) in enumerate(zip(burst_windows, burst_peaks)):
            start, end = burst_window

            for unit_idx, spikes in enumerate(self.trains):
                spikes_in_burst = spikes[(spikes >= start) & (spikes <= end)]
                if len(spikes_in_burst) < 2:
                    continue

                aligned_spikes = spikes_in_burst - peak_time
                counts, _ = np.histogram(aligned_spikes, bins=time_bins)
                ifr_trials[unit_idx][burst_idx, :] = counts / bin_size

        time_axis = time_bins[:-1] + (bin_size / 2)
        return time_axis, ifr_trials

    
    def compute_pairwise_ifr_correlation(self, rate_matrix, units=None, min_activity=1e-6):
        """
        Computes pairwise Pearson correlation between unit IFRs over time.

        Method:
            1. Select valid units.
            2. For each pair of units:
                - Compute Pearson correlation between their IFR traces.
                - Skip units with near-zero variance.

        Args:
            rate_matrix (np.ndarray): IFR matrix [n_time_bins, n_units].
            units (List[int], optional): If given, only compute correlations for these units.
            min_activity (float): Skip units with near-zero activity.
        
        Outputs:
            corr_matrix (np.ndarray): Symmetric [n_units, n_units] matrix of correlations.
        
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
    

    def compute_burst_similarity(self, dataset_key, window_size=3.0):
        """
        Computes cosine similarity between bursts in a given dataset.

        Args:
            dataset_key (str): Key identifying the dataset in `self.spike_data`.
            window_size (float): Total duration (in seconds) of IFR window around burst peaks (default=3.0).

        Outputs:
            sim_matrix (np.ndarray): [n_bursts, n_bursts] cosine similarity matrix.
            burst_stack (np.ndarray): IFR slices aligned to burst peaks [n_bursts, n_units, n_bins].
            avg_matrix (np.ndarray): Mean IFR across bursts [n_units, n_bins].
            peak_times (np.ndarray): Detected burst peak times (seconds).
        """
        if dataset_key not in self.spike_data:
            raise ValueError(f"Dataset '{dataset_key}' not found in spike_data.")
        
        sd = self.spike_data[dataset_key]
        fs = sd.metadata.get("fs", 10000)

        bamac = BurstAnalysisMacro(sd.train, fs, config=self.config)

        half_window = window_size / 2.0
        burst_stack, avg_matrix, peak_times, _ = bamac.compute_burst_aligned_ifr_matrix(
            burst_window_s=half_window
        )

        if burst_stack is None or len(burst_stack) == 0:
            print(f" No bursts detected in dataset '{dataset_key}'.")
            return None, None, None, None

        sim_matrix = bamac.compute_similarity_matrix(burst_stack)

        return sim_matrix, burst_stack, avg_matrix, peak_times

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
    

    def compute_unit_ifr_for_burst_window(self, burst_window, bin_size=0.001):
        """
        Compute the instantaneous firing rates (IFRs) for all units within a specified burst window.

        Args:
            burst_window (tuple of floats):
                (start_time, end_time) in seconds defining the burst period.
            bin_size (float, optional):
                Width of time bins in seconds for IFR calculation (default = 0.001 → 1 ms bins).

        Returns:
            time_axis (np.ndarray):
                Array of time bin centers spanning the burst window (in seconds).
            ifr_matrix (np.ndarray):
                Shape = (n_bins, n_units). Instantaneous firing rate (Hz) for each unit over the burst window.

        Notes:
            - IFR = spike counts per bin / bin width (Hz).
            - Only spikes falling inside the burst window are considered.
        """

        start_time, end_time = burst_window
        time_bins = np.arange(start_time, end_time + bin_size, bin_size)
        n_bins = len(time_bins) - 1
        n_units = len(self.trains)

        ifr_matrix = np.zeros((n_bins, n_units))

        for unit_idx, spikes in enumerate(self.trains):
            if len(spikes) == 0:
                continue
            # Select spikes inside this burst window
            spikes_in_burst = spikes[(spikes >= start_time) & (spikes <= end_time)]
            # Compute histogram and convert to firing rate (Hz)
            counts, _ = np.histogram(spikes_in_burst, bins=time_bins)
            ifr_matrix[:, unit_idx] = counts / bin_size

        # Bin centers for plotting
        time_axis = time_bins[:-1] + (bin_size / 2)

        return time_axis, ifr_matrix
    

# ---------------------------------------------------------------------------------
#           LEVEL 3: BURST ANALYSIS MICRO - SINGLE BURST AND UNIT CHARACTERIZATION
# ---------------------------------------------------------------------------------
    
class BurstAnalysisMicro:
    def __init__(self, trains, fs=10000, config=None):
        self.trains = trains # each element is a list of lists (array) of spike times (in seconds) for a single neuron.
        self.fs = fs # int (default = 10000 Hz)
        self.config = config or {} # dict of config params 

    def get_backbone_units(self, burst_windows, 
                       min_spikes_per_burst=2, 
                       min_fraction_bursts=1.0,
                       min_total_spikes=30):
        """
        Classifies units into backbone and non-rigid groups based on spike participation across bursts.

        Stores the following outputs as two lists of unit indices:
            backbone_units (List[int]): Units that consistently meet activity thresholds.
            non_rigid_units (List[int]): Units failing to meet backbone criteria.

        Config Parameters for backbone detection:
            min_spikes_per_burst (int): Minimum spikes a unit must fire within a burst 
                for that burst to count as active participation. Default=2 spikes.
            min_fraction_bursts (float): Fraction of bursts in which a unit must be active
                to qualify as backbone. Default=1.0 (active in every burst).
            min_total_spikes (int): Minimum number of spikes across all bursts
                for a unit to be considered backbone. Default=30 spikes.

        Method:
            (1) Iterate through each unit's spike train.
            (2) For every detected burst window:
                - Count spikes within that window.
                - If spikes ≥ min_spikes_per_burst → mark burst as "active".
                - Add to total spikes across all bursts.
            (3) Compute the fraction of active bursts for each unit:
                fraction_active = active_bursts / total_bursts
            (4) Classify the unit as backbone if:
                - total_spikes ≥ min_total_spikes AND
                - fraction_active ≥ min_fraction_bursts
            (5) All other units are labeled as non-rigid.

        Outputs:
            backbone_units (List[int]): List of indices for backbone units.
            non_rigid_units (List[int]): List of indices for non-rigid units.
        """
        n_units = len(self.trains)  # Count how many neurons are in the recording (each train = 1 unit)
        n_bursts = len(burst_windows)  # Count how many bursts were detected
        backbone_units = []  # Prepare an empty list to collect indices of backbone units
        non_rigid_units = []  # Prepare an empty list to collect indices of non-rigid units

        # If no bursts were detected, return all units as non-rigid
        if n_bursts == 0:
            print("Warning: No bursts provided for backbone detection.")
            return [], list(range(n_units))

        # Loop over every unit (each unit is a spike train)
        for unit_idx, spike_train in enumerate(self.trains):
            bursts_with_activity = 0  # How many bursts had enough spikes for this unit
            total_spikes_in_bursts = 0  # Total spikes across all bursts for this unit

            # Go through each burst window (start and end time in seconds)
            for (start, end) in burst_windows:
                # Select spikes that happened during this burst
                spikes_in_burst = spike_train[(spike_train >= start) & (spike_train <= end)]

                # Count how many spikes this unit fired during the burst
                n_spikes = len(spikes_in_burst)

                # Add this count to the total number of spikes across bursts
                total_spikes_in_bursts += n_spikes

                # If this burst reached the minimum required spikes, mark it as active
                if n_spikes >= min_spikes_per_burst:
                    bursts_with_activity += 1

            # Calculate fraction of bursts where this unit was active
            fraction_active = bursts_with_activity / n_bursts

            # Check if the unit qualifies as a backbone unit
            if (total_spikes_in_bursts >= min_total_spikes) and (fraction_active >= min_fraction_bursts):
                backbone_units.append(unit_idx)  # Add index to backbone list
            else:
                non_rigid_units.append(unit_idx)  # Otherwise classify as non-rigid

        # Return the two groups of units
        return backbone_units, non_rigid_units

    def compute_pairwise_ifr_cross_correlation(self, units=None, fs=10000, max_lag=0.35,
                                           burst_windows=None, bin_size=0.005):
        """
        Optimized computation of pairwise cross-correlation coefficients and lag times
        between units' instantaneous firing rates (IFRs). Limits analysis to burst periods,
        uses 5 ms bins, and vectorized FFT-based cross-correlation for speed.

        Args:
            units (list[int], optional): Indices of units to include. Defaults to all units.
            fs (int): Original recording sampling frequency in Hz (used for reference).
            max_lag (float): Maximum lag in seconds for correlation search. Default = 0.35 s.
            burst_windows (list[tuple], optional): Time intervals (start, end) in seconds
                where bursts occurred; analysis is restricted to these windows.
            bin_size (float): Time bin width in seconds for IFR histogram. Default = 0.005 s (5 ms).

        Outputs:
            cc_matrix (np.ndarray): Max correlation coefficients (n_units x n_units).
            lag_matrix (np.ndarray): Lag times (ms) where peak correlation occurs (n_units x n_units).
        """

        # -----------------------------
        # 1) Determine analysis window
        # -----------------------------
        # If burst windows are provided, only analyze spike times within detected bursts (+0.5s padding)
        if burst_windows and len(burst_windows) > 0:
            start_time = min(s for s, _ in burst_windows) - 0.5  # earliest burst start - buffer
            end_time = max(e for _, e in burst_windows) + 0.5    # latest burst end + buffer
        else:
            # Otherwise, include entire recording duration
            all_spikes = np.hstack([t for t in self.trains if len(t) > 0])
            start_time, end_time = 0, np.max(all_spikes)

        # Ensure start_time is not negative
        start_time = max(0, start_time)

        # -----------------------------
        # 2) Build time bins for IFR
        # -----------------------------
        # Define bin edges from start to end time using chosen bin_size
        bin_edges = np.arange(start_time, end_time + bin_size, bin_size)

        # Compute number of time bins for the analysis
        time_bins = len(bin_edges) - 1

        # If units not specified, include all neurons
        if units is None:
            units = list(range(len(self.trains)))

        # Count selected units
        n_units = len(units)

        # Initialize IFR matrix: rows = time bins, columns = units
        ifr_matrix = np.zeros((time_bins, n_units), dtype=float)

        # -----------------------------
        # 3) Bin spikes for each unit
        # -----------------------------
        for i, u in enumerate(units):
            spikes = self.trains[u]  # retrieve spike times for this unit
            if len(spikes) > 0:
                # Restrict spikes to analysis window
                spikes = spikes[(spikes >= start_time) & (spikes <= end_time)]
                # Count spikes in each bin
                counts, _ = np.histogram(spikes, bins=bin_edges)
                # Convert counts to firing rate in Hz (spikes per second)
                ifr_matrix[:, i] = counts / bin_size

        # -----------------------------
        # 4) Normalize IFR per unit
        # -----------------------------
        # Compute mean firing rate for each unit
        mean_rates = ifr_matrix.mean(axis=0)
        # Compute standard deviation for each unit
        std_rates = ifr_matrix.std(axis=0)
        # Prevent division by zero for silent units
        std_rates[std_rates == 0] = 1
        # Normalize IFRs to zero mean and unit variance
        norm_ifr = (ifr_matrix - mean_rates) / std_rates

        # -----------------------------
        # 5) Prepare correlation matrices
        # -----------------------------
        # Convert maximum lag from seconds to number of time bins
        max_lag_bins = int(max_lag / bin_size)
        # Initialize matrix for correlation coefficients
        cc_matrix = np.zeros((n_units, n_units), dtype=float)
        # Initialize matrix for lag times in ms
        lag_matrix = np.zeros((n_units, n_units), dtype=float)

        # -----------------------------
        # 6) Compute cross-correlation for each pair of units
        # -----------------------------
        for i in range(n_units):
            xi = norm_ifr[:, i]  # IFR time series for unit i
            for j in range(i, n_units):
                yj = norm_ifr[:, j]  # IFR time series for unit j

                # Compute cross-correlation via FFT (fast convolution)
                corr = fftconvolve(xi, yj[::-1], mode='full')

                # Build array of lag values corresponding to correlation indices
                lags = np.arange(-len(xi) + 1, len(xi)) * bin_size

                # Identify center index representing zero lag
                mid = len(corr) // 2
                # Define indices corresponding to ±max_lag window
                low, high = mid - max_lag_bins, mid + max_lag_bins + 1

                # Extract correlation values within window
                corr_window = corr[low:high]
                # Extract lag times for this window
                lags_window = lags[low:high]

                # Find index of maximum absolute correlation value
                idx = np.argmax(np.abs(corr_window))

                # Store normalized correlation coefficient (divide by series length)
                max_corr = corr_window[idx] / len(xi)
                cc_matrix[i, j] = max_corr
                cc_matrix[j, i] = max_corr  # symmetric matrix

                # Store lag time in milliseconds for this pair
                max_lag_time = lags_window[idx] * 1000
                lag_matrix[i, j] = max_lag_time
                lag_matrix[j, i] = -max_lag_time  # opposite sign for reversed pair

        # -----------------------------
        # 7) Return results
        # -----------------------------
        return cc_matrix, lag_matrix

    def compute_ifr_matrix(self, bin_size=0.001):
        """
        Computes an instantaneous firing rate (IFR) matrix for all units
        by binning spike times into fixed time windows.

        Stores:
            time_axis (np.ndarray): Center of each time bin in seconds.
            ifr_matrix (np.ndarray): Firing rate values in Hz, shape = (time_bins, n_units).

        Method:
            (1) Determine full recording duration from last spike in all units.
            (2) Create equal-width time bins from 0 to duration.
            (3) Count spikes for each unit in each bin using histogram.
            (4) Convert counts to Hz by dividing by bin width.
            (5) Return the time axis and IFR matrix for downstream analyses.

        Args:
            bin_size (float): Time bin width in seconds. Default=0.001 s (1 ms bins).

        Outputs:
            time_axis (np.ndarray): Bin centers in seconds.
            ifr_matrix (np.ndarray): Firing rate in Hz, shape = (time_bins, n_units).
        """
        import numpy as np

        # Gather all spikes from all units into one array
        all_spikes = np.hstack([t for t in self.trains if len(t) > 0])

        # If no spikes present, return empty IFR matrix
        if all_spikes.size == 0:
            return np.array([]), np.zeros((0, len(self.trains)))

        # Find the total recording duration (last spike time)
        duration = np.max(all_spikes)

        # Build time bins (edges from 0 to duration, spaced by bin_size)
        bin_edges = np.arange(0, duration + bin_size, bin_size)

        # Compute bin centers for time axis
        time_axis = bin_edges[:-1] + bin_size / 2

        # Initialize IFR matrix (time_bins x n_units)
        n_units = len(self.trains)
        ifr_matrix = np.zeros((len(time_axis), n_units))

        # Loop through each unit to compute spike counts per bin
        for i, train in enumerate(self.trains):
            if len(train) > 0:
                counts, _ = np.histogram(train, bins=bin_edges)
                ifr_matrix[:, i] = counts / bin_size  # Convert counts to Hz

        # Return time axis and full IFR matrix
        return time_axis, ifr_matrix

    def separate_unit_pair_types(self, cc_matrix, backbone_units):
        """
        Separates pairwise correlation coefficients into three categories:
        (1) Backbone-Backbone (BB-BB)
        (2) Backbone-NonRigid (BB-NR)
        (3) NonRigid-NonRigid (NR-NR)

        Args:
            cc_matrix (np.ndarray): Pairwise correlation coefficients (n_units x n_units).
            backbone_units (list[int]): Indices of backbone units within analyzed units.

        Outputs:
            bb_pairs (list[float]): Correlation coefficients for backbone-backbone pairs.
            bn_pairs (list[float]): Correlation coefficients for backbone-nonrigid pairs.
            nn_pairs (list[float]): Correlation coefficients for nonrigid-nonrigid pairs.
        """
        import numpy as np

        # Identify all units
        all_units = list(range(cc_matrix.shape[0]))
        non_rigid_units = [u for u in all_units if u not in backbone_units]

        bb_pairs, bn_pairs, nn_pairs = [], [], []

        # Iterate over unique pairs (upper triangle only)
        for i in range(len(all_units)):
            for j in range(i + 1, len(all_units)):
                val = cc_matrix[i, j]
                if i in backbone_units and j in backbone_units:
                    bb_pairs.append(val)
                elif ((i in backbone_units and j in non_rigid_units) or
                    (j in backbone_units and i in non_rigid_units)):
                    bn_pairs.append(val)
                else:
                    nn_pairs.append(val)

        return bb_pairs, bn_pairs, nn_pairs


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


def separate_unit_pair_types(corr_matrix, backbone_units):
    """
    Separates pairwise cross-correlation values into three categories:
    1) Backbone-to-Backbone (BB-BB)
    2) Backbone-to-Non-Rigid (BB-NR)
    3) Non-Rigid-to-Non-Rigid (NR-NR)

    Args:
        corr_matrix (np.ndarray): Symmetric matrix [n_units x n_units] containing pairwise 
            maximum cross-correlation coefficients between units.
        backbone_units (list or np.ndarray): Indices of units classified as backbone units.

    Returns:
        bb_bb (list): Correlation values for pairs where both units are backbone units.
        bb_nr (list): Correlation values for pairs where one is backbone, one is non-rigid.
        nr_nr (list): Correlation values for pairs where both are non-rigid units.

    Notes:
        - Only upper triangle (excluding diagonal) of the matrix is considered 
          to avoid duplicate pairs and self-correlations.
        - Non-rigid units are all units not listed in `backbone_units`.
    """
    n_units = corr_matrix.shape[0]
    backbone_units = set(backbone_units)
    all_units = set(range(n_units))
    non_rigid_units = all_units - backbone_units

    bb_bb, bb_nr, nr_nr = [], [], []

    # Iterate over unique unit pairs (upper triangle only)
    for i in range(n_units):
        for j in range(i+1, n_units):
            val = corr_matrix[i, j]
            if i in backbone_units and j in backbone_units:
                bb_bb.append(val)
            elif (i in backbone_units and j in non_rigid_units) or \
                 (j in backbone_units and i in non_rigid_units):
                bb_nr.append(val)
            else:
                nr_nr.append(val)

    return bb_bb, bb_nr, nr_nr

def compute_backbone_means(corr_matrix, backbone_units):
    """
    Computes the mean pairwise correlation values for Backbone-to-Backbone unit pairs.

    Args:
        corr_matrix (np.ndarray): Symmetric matrix [n_units x n_units] with pairwise
            maximum cross-correlation coefficients between units.
        backbone_units (list or np.ndarray): Indices of units classified as backbone units.

    Returns:
        mean_corr (float): Average correlation value across all BB-BB unit pairs.
        individual_corrs (list): List of correlation values for all BB-BB pairs.

    Notes:
        - This helper is primarily used in Panel F to mark the distribution mean 
          of backbone pairs on the histogram.
        - Only upper triangle is considered to avoid duplicates.
    """
    backbone_units = list(backbone_units)
    bb_bb_corrs = []

    for i in range(len(backbone_units)):
        for j in range(i+1, len(backbone_units)):
            u1 = backbone_units[i]
            u2 = backbone_units[j]
            bb_bb_corrs.append(corr_matrix[u1, u2])

    if not bb_bb_corrs:
        return np.nan, []

    mean_corr = np.mean(bb_bb_corrs)
    return mean_corr, bb_bb_corrs