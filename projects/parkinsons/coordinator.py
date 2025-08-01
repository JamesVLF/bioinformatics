import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from collections import defaultdict
from burst_analysis.loading import SpikeDataLoader
from burst_analysis.detection import BurstDetection
from burst_analysis.computation import BurstAnalysisMacro, BurstAnalysisMicro
from burst_analysis.plotting import BurstDetectionPlots, BAMacPlots, BAMicPlots 


DEFAULT_PATHS = {
    "d0s2_Control": "/Users/main_mac/bioinformatics/data/extracted/maxtwo_newconfig1/MO6359s2_D53_Control_BASELINE_0hr.npz",
    "d0s6_Treated": "/Users/main_mac/bioinformatics/data/extracted/maxtwo_newconfig1/MO6359s6_D53_175µM_BASELINE_0hr.npz",
    "d6s2_Control": "/Users/main_mac/bioinformatics/data/extracted/maxtwo_newconfig1/MO6359s2_D60_Control_T2_D6.npz",
    "d6s6_Treated": "/Users/main_mac/bioinformatics/data/extracted/maxtwo_newconfig1/MO6359s6_D60_175µM_T2_D6.npz"
}
class OrchestratorPDx2:
    def __init__(self, spike_paths=None):
        """
        Initializing:
            - loads data in `spike_paths` as SpikeData objects
            - makes all loaded data 'active'
            - sets `sd_main` as default dataset
            - activates analysis, plotting, and tracking tools
        """
            
        self.spike_paths = spike_paths or DEFAULT_PATHS
        self.loader = SpikeDataLoader(self.spike_paths)
        self.loader.spike_data = self.loader.load() 
        self.burst_detection_metrics_df = pd.DataFrame()
        self.groups = {}            # group datasets, burst classes, and unit types
        self.active_group = None    # track currently selected group
        self.burst_metrics_cache = {}  # transient container
        self.burst_analyzer = None # instance holder for BurstDetection
        self.active_datasets = list(self.loader.spike_data.keys()) # Set all loaded datasets as active

        # Set default dataset for sd_main
        if "d0s2_Control" in self.active_datasets:
            self.loader.set_dataset("d0s2_Control")
            print("Set d0s2_Control as default dataset.")
        elif self.active_datasets:
            fallback = self.active_datasets[0]
            self.loader.set_dataset(fallback)
            print(f"Set '{fallback}' as default dataset (first available).")
        else:
            raise ValueError("No datasets found during loading.")

        

    @property
    def sd_main(self):
        return self.loader.sd_main

    @property
    def spike_data(self):
        return self.loader.spike_data

    @property
    def metadata_df(self):
        return self.loader.metadata_df

    @property
    def neuron_df(self):
        return self.loader.neuron_df

    def set_dataset(self, name):
        """Updates sd_main so that all methods will default to 'name' if no other key is passed."""
        self.loader.set_dataset(name)

    def set_active_datasets(self, keys):
        if not isinstance(keys, list):
            keys = [keys]
        missing = [k for k in keys if k not in self.spike_data]
        if missing:
            raise ValueError(f"Dataset(s) not found: {missing}")
        self.active_datasets = keys
        self.loader.set_dataset(keys[0])  # Set first as main for compatibility
        print(f"Activated datasets: {self.active_datasets}")

    def list_datasets(self):
        """Returns a list of loaded dataset keys for quick inspection."""
        return list(self.spike_data.keys())

    def _init_burst_analyzer(self, config=None):
        self.burst_analyzer = BurstDetection(
            self.sd_main.train,
            self.sd_main.metadata.get("fs", 10000),
            config = config if config is not None else {}
        )

    def _normalize_dataset_key(self, key):
        if isinstance(key, np.ndarray) and key.size == 1:
            return key.item()
        return key
    
    def resolve_group_input(self, group_input):
        return self.loader.resolve_group_input(group_input)

    def define_dataset_group(self, group_name, filter_func):
        return self.loader.define_dataset_group(group_name, filter_func)

    def get_unit_info(self, dataset_name=None):
        return self.loader.get_unit_info(dataset_name)

    def get_active_dataset_key(self):
        for k, v in self.spike_data.items():
            if v is self.sd_main:
                return k
        return None

    def build_neuron_df(self):
        return self.loader.build_neuron_df()

    def build_metadata_df(self):
        return self.loader.build_metadata_df()  

    def get_burst_detector(self, dataset_key, config=None):
        if dataset_key not in self.spike_data:
            raise ValueError(f"Dataset '{dataset_key}' not found.")
        sd = self.spike_data[dataset_key]
        fs = sd.metadata.get("fs", 10000)
        return BurstDetection(sd.train, fs=fs, config=config or {})

    def list_datasets(self):
        return list(self.spike_data.keys())
    
    def set_dataset(self, name):
        """Updates sd_main so that all methods will default to 'name' if no other key is passed."""
        self.loader.set_dataset(name)

    def set_active_datasets(self, keys):
        if not isinstance(keys, list):
            keys = [keys]
        missing = [k for k in keys if k not in self.spike_data]
        if missing:
            raise ValueError(f"Dataset(s) not found: {missing}")
        self.active_datasets = keys
        self.loader.set_dataset(keys[0])  # Set first as main for compatibility
        print(f"Activated datasets: {self.active_datasets}")

    def list_datasets(self):
        """Returns a list of loaded dataset keys for quick inspection."""
        return list(self.spike_data.keys())

    def _normalize_dataset_key(self, key): 
        # for numpy array --> scalar conversion
        if isinstance(key, np.ndarray) and key.size == 1:
            return key.item()
        return key
    
    def _resolve_dataset_keys(self, dataset_key):
        # Normalize dataset selection to a list of keys
        if dataset_key is None:
            return list(self.spike_data.keys())
        elif isinstance(dataset_key, str):
            return [dataset_key]
        elif isinstance(dataset_key, (list, tuple)):
            return list(dataset_key)
        else:
            raise ValueError("dataset_key must be None, str, or list/tuple of strings.")
        
    def get_unit_info(self, dataset_name=None):
        return self.loader.get_unit_info(dataset_name)

    def get_dataset_keys(self, dataset_keys=None):
        if dataset_keys:
            return [k for k in dataset_keys if k in self.spike_data]
        return list(self.spike_data.keys())

    # -------------------------------------------------------------------------
    #           LEVEL 1: BURST DETECTION + FEATURE EXTRACTION
    # -------------------------------------------------------------------------

    def show_neuron_count_plot(self, group_input=None):
        BurstDetectionPlots.plot_neuron_counts(self.metadata_df, group_input, self.groups)

    def compute_and_plot_population_bursts(self, dataset_keys=None, config=None, time_range=(0, 180), save=False, output_dir=None):
        if dataset_keys is None:
            dataset_keys = list(self.spike_data.keys())
        all_metrics = {}

        for key in dataset_keys:
            detector = self.get_burst_detector(key, config)
            result = detector.compute_population_rate_and_bursts()
            if result[0] is None:
                print(f"Burst detection unsuccessful for '{key}'")
                continue
            times, smoothed, peaks, peak_times, bursts, burst_windows = result

            time_start = config.get("time_start", 0.0) if config else 0.0
            time_window = config.get("time_window", times[-1]) if config else times[-1]
            metrics = detector.compute_pop_burst_metrics(times=times, smoothed=smoothed, peaks=peaks, bursts=bursts,
                            time_start=time_start, time_window=time_window, peak_times=peak_times, burst_windows=burst_windows)

            sd = self.spike_data[key]
            BurstDetectionPlots.plot_overlay_raster_population(trains=sd.train, times=times, smoothed=smoothed, bursts=bursts,
                dataset_label=key, time_range=time_range, save=save, output_dir=output_dir)
            all_metrics[key] = metrics
        return all_metrics
            
    def print_burst_summary(self, dataset_key, duration_s, total_spikes, n_neurons, bin_size_s, bursts, peaks, times):
        print(f"\nBurst Extraction Summary — {dataset_key}")
        print(f"Recording duration: {duration_s:.2f} s")
        print(f"Total spikes: {total_spikes}")
        print(f"Number of neurons: {n_neurons}")
        print(f"Mean firing rate per neuron: {(total_spikes / n_neurons / duration_s):.2f} Hz" if duration_s > 0 else "N/A")

        if bursts:
            durations = [(e - s) * bin_size_s for s, e in bursts]
            print(f"\nBursts detected: {len(bursts)}")
            print(f"Mean burst duration: {np.mean(durations):.3f} s")
            print(f"Std of burst durations: {np.std(durations):.3f} s")
            print(f"Burst rate: {len(bursts) / (duration_s / 60):.2f} bursts/min" if duration_s > 0 else "N/A")

            if len(peaks) > 1:
                peak_times_arr = np.array([times[p] for p in peaks])
                ibis = np.diff(peak_times_arr)
                print(f"Mean inter-burst interval (IBI): {np.mean(ibis):.2f} s")
        else:
            print("\n No bursts detected.")

    # ----------------------------------------------------------
    #           LEVEL 2 ANALYSIS 
    # ----------------------------------------------------------

    def compute_and_plot_unit_participation(self, sample_configs, control_subjects=None,
                                            dataset_keys=None, ordered_short_labels=None,
                                            save=False, output_path=None):
        """
        Method:
            1. Computes per-unit burst participation fractions across multiple datasets;
            2. Generates a violin plot comparing control and drugged groups.

        Args:
            sample_configs : dict that  maps dataset identifiers (like 'S1', 'S2'...) to analysis configuration objects
                for burst detection computations.
            control_subjects : set (optional) of subject IDs (like {'s1','s2','s3'}) for the CONTROL group. 
                Remaining datasets get labeled TREATED. If Default=None → no CONTROL grouping.
            dataset_keys : list (optional) of keys to analyze. Defaults to all loaded datasets.
            ordered_short_labels : list of str (optional, but in explicit order) of x-axis sample labels (like ['d-6s1','d1s1',...]).
                If None → ordering follows dataset key sorting.
            save : bool, optional
                If True → saves the plot as PNG.
            output_path : str (used only if save=True).

        Outputs:
            grouped_participation : dict mapping of 'GROUP_dataset' → list of per-unit participation fractions.
        """

        # Default args
        dataset_keys = dataset_keys or list(self.spike_data.keys())
        control_subjects = control_subjects or set()
        grouped_participation = defaultdict(list)

        # --- COMPUTATION LOOP ---
        for dataset_key in sorted(dataset_keys):
            if dataset_key not in self.spike_data:
                print(f"Dataset '{dataset_key}' not found in loaded data. Skipping.")
                continue

            # Normalize naming and extract subject ID
            parts = dataset_key.split("_")[0]  # e.g., "d-6s4"
            sample_num = parts.split("s")[-1]
            subject_label = f"s{sample_num}"

            # Get configuration
            config = sample_configs.get(f"S{sample_num}")
            if config is None:
                print(f"No config for dataset {dataset_key}. Skipping.")
                continue

            # Initialize burst detector and run population burst detection
            detector = self.get_burst_detector(dataset_key, config)
            result = detector.compute_population_rate_and_bursts()
            if result is None or result[-1] is None:
                print(f"Burst computation failed for {dataset_key}. Skipping.")
                continue

            burst_windows = result[-1]
            if not burst_windows:
                print(f"No bursts found in {dataset_key}")
                continue

            # Compute unit participation using BurstAnalysisMacro
            trains = self.spike_data[dataset_key].train
            analysis = BurstAnalysisMacro(trains, fs=detector.fs, config=config)
            fractions = analysis.compute_unit_participation(burst_windows)

            # Group CONTROL vs TREATED
            group = "CONTROL" if subject_label in control_subjects else "TREATED"
            label = f"{group}_{parts}"
            grouped_participation[label].extend(fractions)
            print(f"DONE: {dataset_key}: {len(burst_windows)} bursts, {len(fractions)} units")

        # --- PLOTTING ---
        BAMacPlots.plot_violin_burst_participation(
            grouped_participation=grouped_participation,
            ordered_short_labels=ordered_short_labels,
            save=save,
            output_path=output_path
        )
        return grouped_participation
    
    def run_burst_rank_order_analysis(orc, sample_configs, ordered_short_labels, control_subjects):
        """
       Method: 
            1. Builds peak time matrices for each dataset
            2. Computes Spearman rank correlations and z-scores
            3. Collects z-score distributions by condition
            4. Generates violin plots of the results

        Args:
            orc: object with loaded spike data and orchestrator attributes
            sample_configs: dict mapping of configurations for datasets (keyed by sample label)
            ordered_short_labels: list of str dataset labels ordered for plotting
            control_subjects: set of IDs designated as CONTROL for grouping

        Notes:
            - Iterates through all datasets in `orc.loader.spike_data`
            - Filters bursts and units in accord with threshold rules
            - Uses 3 methods and plotting functions for the notebook UI function
        """

        zscore_distributions = defaultdict(list)

        for dataset_key in sorted(orc.loader.spike_data.keys()):
            orc.loader.set_dataset(dataset_key)

            parts = dataset_key.split("_")[0]
            sample_num = parts.split("s")[-1]
            sample_label = f"S{sample_num}"
            config = sample_configs.get(sample_label)

            if config is None:
                continue

            try:
                analysis = orc._get_analysis(config=config)
                result = analysis.compute_population_rate_and_bursts()
                times, _, _, _, _, burst_windows = result
            except Exception:
                continue

            if not burst_windows or len(burst_windows) < 3:
                continue

            trains = orc.sd_main.train
            peak_times_matrix = BurstAnalysisMacro.build_peak_times_matrix(trains, burst_windows, threshold=0.5)
            if peak_times_matrix is None:
                continue

            _, zscores = BurstAnalysisMacro.compute_rank_corr_and_zscores(peak_times_matrix)
            if zscores is None:
                continue

            flat_zscores = zscores[np.triu_indices(zscores.shape[0], k=1)]
            condition = "CONTROL" if "s2" in parts else "TREATED"
            label = f"{condition}_{parts}"
            zscore_distributions[label].extend(flat_zscores.tolist())

        BAMacPlots.plot_zscore_distributions(zscore_distributions, ordered_short_labels, control_subjects)

    def run_similarity_analysis(self, dataset_key=None, config=None, window_size=3.0):
        """
                        spike trains ──► detect bursts ──► extract burst windows
                                            │
                                            ▼
                                extract_bursts_from_train
                                            │
                                            ▼
                        build IFR slices (aligned to burst peak, same length)
                                            │
                                            ▼
                    compute_similarity_matrix (cosine similarity of IFR slices)
                                            │
                                            ▼
                            plot_burst_similarity_matrix
        """
        # ------- (1) Load dataset -------------
        if dataset_key not in self.spike_data:
            raise ValueError(f"Dataset '{dataset_key}' not found.")
        sd = self.spike_data[dataset_key]

        # -------- (2) Instantiate analysis object ----------
        bamac = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        # ---------- (3) Extract IFR matrices for bursts (aligned to peaks) -------
        burst_stack, avg_matrix, peak_times, _ = bamac.compute_burst_aligned_ifr_matrix(
            burst_window_s=window_size / 2  # half-window = 1.5s for a 3s total window
        )
        if burst_stack is None:
            print(f" No bursts detected for '{dataset_key}'.")
            return

        # ------- (4) Compute similarity between bursts ----------
        sim_matrix = bamac.compute_similarity_matrix(burst_stack)

        # ----- (5) Plot similarity matrix ---------
        BAMacPlots.plot_burst_similarity_matrix(
            sim_matrix, title=f"Burst Similarity (Cosine) - {dataset_key}"
        )
        return sim_matrix, burst_stack, peak_times
    
    def plot_combined_similarity_analysis(self, dataset_key=None):
        """
        Generates dual plots (similarity matrix + mean IFR) 
        for one, multiple, or all datasets using precomputed results
        """
        keys = [self._normalize_dataset_key(dataset_key)] if dataset_key is not None else list(self.spike_data.keys())
        results_dict = {}

        for key in keys:
            analysis_res = self.analysis_results.get(key, None)
            if analysis_res is None:
                print(f" No precomputed similarity results for dataset '{key}'. Skipping.")
                continue
            sim_matrix, burst_stack, avg_matrix, peak_times = analysis_res
            results_dict[key] = {
                "sim_matrix": sim_matrix,
                "burst_stack": burst_stack,
                "avg_matrix": avg_matrix,
                "peak_times": peak_times
            }
        if not results_dict:
            print(" No valid datasets found for plotting.")
            return
        BAMacPlots.plot_dual_similarity_matrices(results_dict)

    def run_burst_aligned_ifr_analysis(self, dataset_key=None, config=None, plot=True):
        """
        Computes and (optionally) plots burst-aligned IFR matrices
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        burst_stack, avg_matrix, peak_times = analysis.compute_burst_aligned_ifr_matrix(**(config or {}))

        if plot:
            BAMacPlots.plot_burst_aligned_ifr_matrix(avg_matrix, peak_times)

        return burst_stack, avg_matrix, peak_times

    def run_peak_centered_ifr_segments(self, dataset_key=None, config=None, window_s=3.0, plot=True):
        """
        Extracts IFR segments centered on each detected burst peak and (optionally) plots them
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        stack, time_axis = analysis.get_peak_centered_ifr_segments(window_s=window_s)

        if plot:
            BAMacPlots.plot_peak_centered_ifr_segments(stack, time_axis)

        return stack, time_axis

    def run_population_ifr_with_bursts(self, dataset_key=None, config=None, plot=True):
        """
        Computes population IFR and highlights burst windows in plot
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        rate_matrix, times, bursts = analysis.compute_population_ifr_with_bursts()

        if plot:
            BAMacPlots.plot_population_ifr_with_bursts(rate_matrix, times, bursts)
        return rate_matrix, times, bursts

    def run_pairwise_ifr_correlation(self, dataset_key=None, config=None, units=None, plot=True):
        """
        Computes pairwise IFR correlation matrix and optional heatmap
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        rate_matrix = analysis.compute_ifr_matrix()
        corr_matrix = analysis.compute_pairwise_ifr_correlation(rate_matrix.T, units=units)

        if plot:
            BAMacPlots.plot_pairwise_ifr_correlation(corr_matrix)

        return corr_matrix

    def run_relative_unit_peak_times(self, dataset_key=None, config=None, plot=True):
        """
        Computes relative unit peak times for bursts and (optionally) plots histogram
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000), config)

        # ------ (1) Get population IFR with bursts --------
        rate_matrix, times, bursts = analysis.compute_population_ifr_with_bursts()

        # ------ (2) Get burst peak times ---------
        _, _, peak_times, _ = analysis.compute_burst_aligned_ifr_matrix()

        # ----- (3) Compute relative peaks --------
        rel_peaks = analysis.get_relative_unit_peak_times(rate_matrix, times, bursts, peak_times)

        if plot:
            BAMacPlots.plot_relative_unit_peak_times(rel_peaks)

        return rel_peaks

# -----------------------------------------------------------------------------------
#           LEVEL 3: Single Burst + Unit-Level Analysis
# -----------------------------------------------------------------------------------

    def visualize_sorted_burst_ifr(self, dataset_key=None, burst_idx=0, min_spikes=2, save_path=None):
        """
        Computes / graphs sorted IFR activity for a single burst

        Method:
            1. Computes the global IFR matrix and burst windows for the dataset
            2. Extracts and sorts units active during the specified burst
            3. Plots the resulting burst IFR heatmap

        Args:
            dataset_key: str (optional) Defaults to the main dataset if None
            burst_idx: int index of the burst to visualize (0-based)
            min_spikes: int minimum spike threshold by which to include a unit in the plot
            save_path: str (optional) saves plot to that location instead of displaying it

        Outputs:
            matplotlib.figure.Figure handle if plotting is successful, otherwise None
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMacro(sd.train, sd.metadata.get("fs", 10000))

        # Compute full IFR matrix and burst windows
        rate_matrix, times, bursts = analysis.compute_population_ifr_with_bursts()

        # Extract and sort IFR activity for the selected burst
        sorted_burst_matrix, sorted_units, burst_time_axis = analysis.prepare_sorted_ifr_matrix_for_burst(
            rate_matrix,
            times,
            burst_idx,
            bursts,
            min_spikes=min_spikes
        )
        # Generate heatmap
        fig = BAMicPlots.plot_sorted_ifr_for_burst(
            sorted_burst_matrix,
            sorted_units,
            burst_time_axis,
            dataset_key=dataset_key,
            save_path=save_path
        )
        return fig

    def run_backbone_detection(self, dataset_key=None,
                           burst_results=None,
                           metrics_df=None,
                           min_spikes_per_burst=2,
                           min_fraction_bursts=1.0,
                           min_total_spikes=30):
        """
        Classifies units into backbone and non-rigid groups using cached burst data.

        Method:
            (1) Retrieves burst_windows from either:
                    - burst_results dict (session cache), or
                    - metrics DataFrame (loaded from previous runs).
            (2) Uses BurstAnalysisMacro.get_backbone_units() to classify units.
            (3) Prints summary of backbone vs non-rigid units.

        Args:
            dataset_key (str): Dataset key to analyze.
            burst_results (dict, optional): Cached burst detection outputs.
            metrics_df (pd.DataFrame, optional): DataFrame with 'dataset_key' and 'burst_windows' columns.
            min_spikes_per_burst (int): Minimum spikes per burst to count as active.
            min_fraction_bursts (float): Fraction of bursts required to classify as backbone.
            min_total_spikes (int): Minimum total spikes across all bursts.

        Outputs:
            backbone_units (List[int]): Unit indices classified as backbone.
            non_rigid_units (List[int]): Unit indices classified as non-rigid.
        """
        dataset_key = self._normalize_dataset_key(dataset_key)

        # --- Retrieve cached burst windows ---
        burst_windows = None

        if burst_results and dataset_key in burst_results:
            burst_windows = burst_results[dataset_key].get("burst_windows")

        elif metrics_df is not None:
            row = metrics_df[metrics_df["dataset_key"] == dataset_key]
            if not row.empty:
                burst_windows = row["burst_windows"].values[0]

        if not burst_windows:
            print(f"No cached bursts found for dataset '{dataset_key}'.")
            return [], []

        # --- Load spike trains for dataset ---
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMicro(sd.train, fs=sd.metadata.get("fs", 10000))

        # --- Classify units ---
        backbone_units, non_rigid_units = analysis.get_backbone_units(
            burst_windows,
            min_spikes_per_burst=min_spikes_per_burst,
            min_fraction_bursts=min_fraction_bursts,
            min_total_spikes=min_total_spikes
        )

        # --- Print summary ---
        print(f"\n[Backbone Detection] Dataset: {dataset_key}")
        print(f"Detected {len(backbone_units)} backbone units: {backbone_units}")
        print(f"Detected {len(non_rigid_units)} non-rigid units: {non_rigid_units}")

        return backbone_units, non_rigid_units


    
# ------------------------------
# Figure 3 Panel Wrapper Methods
# ------------------------------
    
    def plot_figure3_panel_a(self, dataset_key=None, unit_indices=[0, 1, 2], burst_idx=0, 
                         burst_windows=None, bin_size=0.001, save_path=None, config=None):
        """
        Plot firing rate traces and spike raster for selected units during a single burst window,
        with proper spike time scaling, smoothing, non-overlapping titles, and improved axis scaling.
        """

        # --- Dataset selection ---
        dataset_key = self._normalize_dataset_key(dataset_key) or self.list_datasets()[0]
        if dataset_key not in self.spike_data:
            print(f"Dataset '{dataset_key}' not found.")
            return None
        sd = self.spike_data[dataset_key]
        fs = sd.metadata.get("fs", 10000)  # Sampling frequency for conversion

        # --- Retrieve burst windows ---
        if burst_windows is None:
            detector = self.get_burst_detector(dataset_key, config)
            result = detector.compute_population_rate_and_bursts()
            if result is None or len(result) < 6 or not result[-1]:
                print(f"No bursts detected for dataset '{dataset_key}'.")
                return None
            burst_windows = result[-1]

        if burst_idx >= len(burst_windows):
            print(f"Burst index {burst_idx} out of range. Dataset has {len(burst_windows)} bursts.")
            return None

        burst_start, burst_end = burst_windows[burst_idx]

        # --- Compute IFR for this burst ---
        analysis = BurstAnalysisMacro(sd.train, fs=fs, config=config)
        time_axis, ifr_matrix = analysis.compute_unit_ifr_for_burst_window(
            (burst_start, burst_end),
            bin_size=bin_size
        )

        # --- Plot figure ---
        plt.figure(figsize=(8, 4))
        colors = ['darkorange', 'purple', 'mediumvioletred']

        # Convert time axis to ms relative to burst start
        burst_time_ms = (time_axis - burst_start) * 1000
        burst_start_ms = burst_start * 1000
        burst_end_ms = burst_end * 1000

        # --- Plot IFR traces and spike rasters ---
        from scipy.ndimage import gaussian_filter1d

        for i, unit in enumerate(unit_indices):
            if unit >= ifr_matrix.shape[1]:
                continue

            # Smooth IFR trace for readability
            smoothed_ifr = gaussian_filter1d(ifr_matrix[:, unit], sigma=1)

            # Plot IFR trace
            plt.plot(burst_time_ms, smoothed_ifr,
                    color=colors[i % len(colors)],
                    label=f'Unit {unit}')

            # Convert spikes from samples → seconds → ms (relative to burst start)
            spikes_sec = sd.train[unit] / fs
            spikes_in_window = spikes_sec[(spikes_sec >= burst_start) & (spikes_sec <= burst_end)]
            if len(spikes_in_window) > 0:
                plt.vlines((spikes_in_window - burst_start) * 1000,
                        ymin=-5 * (i + 1),
                        ymax=-1 * (i + 1),
                        color=colors[i % len(colors)],
                        linewidth=1)

        # Axis scaling limited to burst window
        plt.xlim(0, (burst_end - burst_start) * 1000)

        plt.xlabel("Time (ms)")
        plt.ylabel("Firing Rate (Hz)")
        plt.title("Representative Unit IFRs in Single Burst", fontsize=12, pad=20)
        plt.legend()
        plt.tight_layout()

        # Move burst index title up to avoid overlap
        y_max = plt.gca().get_ylim()[1]
        plt.text(x=(burst_end_ms - burst_start_ms) / 2,
                y=y_max * 1.1,
                s=f"Burst {burst_idx + 1}",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            return plt.gcf()

    def plot_figure3_panel_b(self, dataset_key=None, unit_indices=[0, 1, 2],
                         burst_windows=None, burst_peaks=None,
                         bin_size=0.005, save_path=None, config=None):
        """
        Creates an overlay plot of selected units with IFR traces for all bursts.

        Args:
            dataset_key (str): Dataset key.
            unit_indices (list[int]): Units to plot.
            burst_windows (list[tuple], optional): Precomputed burst windows.
            burst_peaks (list[float], optional): Precomputed burst peak times (s).
            bin_size (float): Time bin width in seconds (default=5 ms).
            save (bool): If True, saves the figure.
            save_path (str): Path for saving the figure.
            config (dict): Burst detection config if detection needs to be run.

        Returns:
            matplotlib.figure.Figure or None
        """

        dataset_key = self._normalize_dataset_key(dataset_key) or self.list_datasets()[0]
        if dataset_key not in self.spike_data:
            print(f"Dataset '{dataset_key}' not found.")
            return None
        sd = self.spike_data[dataset_key]

        # Use cached detection results 
        if burst_windows is None or burst_peaks is None:
            detector = self.get_burst_detector(dataset_key, config)
            result = detector.compute_population_rate_and_bursts()
            if result is None or len(result) < 6 or not result[-1]:
                print(f"No bursts detected for dataset '{dataset_key}'.")
                return None
            burst_windows = result[-1]
            burst_peaks = result[3]  # list of peak times

        # Compute IFR trials for all bursts 
        analysis = BurstAnalysisMacro(sd.train, fs=sd.metadata.get("fs", 10000), config=config)
        time_axis, ifr_trials = analysis.compute_burst_aligned_ifr_trials(
            burst_windows, burst_peaks, bin_size=bin_size
        )

        # Keep selected units; remove empty trials
        trials_clean = {u: ifr_trials[u][~(ifr_trials[u] == 0).all(axis=1)]
                        for u in unit_indices if u in ifr_trials}

        fig = plt.figure(figsize=(8, 4))
        plt.close(fig)

        BAMicPlots.plot_average_burst_aligned_ifr(
            time_axis,
            trials_clean,
            selected_units=unit_indices,
            save_path=save_path
        )
        return fig

    def plot_figure3_panel_c(self, dataset_key=None,
                         burst_results=None,
                         config=None,
                         min_spikes_per_burst=2,
                         min_fraction_bursts=0.8,
                         min_total_spikes=30,
                         max_lag=0.35,
                         save_path=None):
        """
        Optimized wrapper for Figure 3 Panel C.

        Method:
            (1) Retrieve cached bursts.
            (2) Identify backbone and non-rigid units.
            (3) Compute FFT-based pairwise IFR correlations (5 ms bins, burst-limited).
            (4) Plot heatmap (backbone units first, then non-rigid units) with grayscale colormap
                and red separators, matching published Panel C style.

        Args:
            dataset_key (str): Dataset key for analysis.
            burst_results (dict): Cached burst detection results.
            config (dict, optional): Analysis configuration.
            min_spikes_per_burst (int): Spike threshold for activity per burst.
            min_fraction_bursts (float): Fraction of bursts required for backbone classification.
            min_total_spikes (int): Minimum total spikes to classify as backbone.
            max_lag (float): Maximum lag (s) for cross-correlation search.
            save_path (str, optional): Path to save Panel C plot.

        Outputs:
            fig (matplotlib.figure.Figure): Handle to generated heatmap figure.
        """
        import os
        dataset_key = self._normalize_dataset_key(dataset_key)
        if dataset_key not in self.spike_data:
            print(f"Dataset '{dataset_key}' not found.")
            return None

        # Retrieve cached bursts
        if burst_results is None or dataset_key not in burst_results:
            print(f"No cached bursts found for dataset '{dataset_key}'.")
            return None
        burst_windows = burst_results[dataset_key]["burst_windows"]
        if not burst_windows:
            print("No bursts detected for this dataset.")
            return None

        # Load spike data
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMicro(sd.train, fs=sd.metadata.get("fs", 10000), config=config)

        # Identify backbone vs non-rigid units
        backbone_units, non_rigid_units = analysis.get_backbone_units(
            burst_windows,
            min_spikes_per_burst=min_spikes_per_burst,
            min_fraction_bursts=min_fraction_bursts,
            min_total_spikes=min_total_spikes
        )
        all_units = backbone_units + non_rigid_units
        if not all_units:
            print("No units found for correlation analysis.")
            return None

        print(f"[Panel C Optimized] Backbone: {len(backbone_units)}, Non-rigid: {len(non_rigid_units)}")

        # Compute FFT-based cross-correlation
        cc_matrix, _ = analysis.compute_pairwise_ifr_cross_correlation(
            units=all_units,
            fs=sd.metadata.get("fs", 10000),
            max_lag=max_lag,
            burst_windows=burst_windows,
            bin_size=0.005  # 5 ms bins
        )

        # Ensure output directory exists
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Plot heatmap with target figure styling
        fig = BAMicPlots.plot_panel_c(
            cc_matrix=cc_matrix,
            backbone_units=list(range(len(backbone_units))),  # pass backbone indices only
            save_path=save_path
        )

        return fig

    def plot_figure3_panel_d(self, dataset_key=None,
                         burst_results=None,
                         config=None,
                         min_spikes_per_burst=2,
                         min_fraction_bursts=0.8,
                         min_total_spikes=30,
                         max_lag=0.35,
                         save_path=None):
        """
        Optimized wrapper for Figure 3 Panel D (lag times).

        Method:
            (1) Retrieve cached bursts.
            (2) Identify backbone and non-rigid units.
            (3) Compute FFT-based lag time matrix (burst-limited, 5 ms bins).
            (4) Plot diverging red-blue heatmap with black separators.

        Args:
            dataset_key (str): Dataset key for analysis.
            burst_results (dict): Cached burst detection results.
            config (dict, optional): Analysis configuration.
            min_spikes_per_burst (int): Spike threshold for burst participation.
            min_fraction_bursts (float): Fraction of bursts required for backbone classification.
            min_total_spikes (int): Total spikes threshold for backbone units.
            max_lag (float): Maximum lag window in seconds for CC.
            save_path (str, optional): Path to save figure.

        Outputs:
            fig (matplotlib.figure.Figure): Heatmap figure handle.
        """
        import os
        dataset_key = self._normalize_dataset_key(dataset_key)
        if dataset_key not in self.spike_data:
            print(f"Dataset '{dataset_key}' not found.")
            return None

        if burst_results is None or dataset_key not in burst_results:
            print(f"No cached bursts found for dataset '{dataset_key}'.")
            return None
        burst_windows = burst_results[dataset_key]["burst_windows"]
        if not burst_windows:
            print("No bursts detected for this dataset.")
            return None

        # Load spike data
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMicro(sd.train, fs=sd.metadata.get("fs", 10000), config=config)

        # Identify backbone and non-rigid units
        backbone_units, non_rigid_units = analysis.get_backbone_units(
            burst_windows,
            min_spikes_per_burst=min_spikes_per_burst,
            min_fraction_bursts=min_fraction_bursts,
            min_total_spikes=min_total_spikes
        )
        all_units = backbone_units + non_rigid_units
        if not all_units:
            print("No units found for lag correlation analysis.")
            return None

        print(f"[Panel D Optimized] Backbone: {len(backbone_units)}, Non-rigid: {len(non_rigid_units)}")

        # Compute optimized cross-correlation lag matrix
        _, lag_matrix = analysis.compute_pairwise_ifr_cross_correlation(
            units=all_units,
            fs=sd.metadata.get("fs", 10000),
            max_lag=max_lag,
            burst_windows=burst_windows,
            bin_size=0.005
        )

        # Ensure output directory exists
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Plot heatmap in target style
        fig = BAMicPlots.plot_panel_d(
            lag_matrix=lag_matrix,
            backbone_units=list(range(len(backbone_units))),
            save_path=save_path
        )

        return fig
    
    def plot_figure3_panel_e(self, dataset_key=None,
                         burst_results=None,
                         config=None,
                         min_spikes_per_burst=2,
                         min_fraction_bursts=0.8,
                         min_total_spikes=30,
                         max_lag=0.35,
                         save_path=None):
        """
        Wrapper for Figure 3 Panel E.

        Method:
            (1) Retrieve cached burst windows.
            (2) Identify backbone and non-rigid units.
            (3) Compute full pairwise cross-correlation matrix (backbone + nonrigid).
            (4) Separate CC values into BB-BB, BB-NR, NR-NR categories.
            (5) Plot violin plot comparing distributions.

        Args:
            dataset_key (str): Dataset key for analysis.
            burst_results (dict): Cached burst detection results.
            config (dict, optional): Analysis configuration.
            min_spikes_per_burst (int): Minimum spikes per burst threshold.
            min_fraction_bursts (float): Fraction of bursts required for backbone.
            min_total_spikes (int): Total spikes threshold for backbone.
            max_lag (float): Maximum lag window in seconds.
            save_path (str, optional): Path to save violin plot figure.

        Outputs:
            fig (matplotlib.figure.Figure): Handle to violin plot figure.
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        if dataset_key not in self.spike_data:
            print(f"Dataset '{dataset_key}' not found.")
            return None

        if burst_results is None or dataset_key not in burst_results:
            print(f"No cached bursts found for dataset '{dataset_key}'.")
            return None
        burst_windows = burst_results[dataset_key]["burst_windows"]
        if not burst_windows:
            print("No bursts detected for this dataset.")
            return None

        # Load spike data
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMicro(sd.train, fs=sd.metadata.get("fs", 10000), config=config)

        # Identify backbone and non-rigid units
        backbone_units, non_rigid_units = analysis.get_backbone_units(
            burst_windows,
            min_spikes_per_burst=min_spikes_per_burst,
            min_fraction_bursts=min_fraction_bursts,
            min_total_spikes=min_total_spikes
        )

        print(f"[Panel E] Backbone units: {len(backbone_units)}, Non-rigid units: {len(non_rigid_units)}")

        # Compute full CC matrix for all units (backbone + nonrigid)
        cc_matrix, _ = analysis.compute_pairwise_ifr_cross_correlation(
            units=backbone_units + non_rigid_units,
            fs=sd.metadata.get("fs", 10000),
            max_lag=max_lag
        )

        # Separate pair types
        bb_pairs, bn_pairs, nn_pairs = analysis.separate_unit_pair_types(
            cc_matrix=cc_matrix,
            backbone_units=list(range(len(backbone_units)))  # indices in reordered list
        )

        # Plot violin plot
        fig = BAMicPlots.plot_panel_e(bb_pairs, bn_pairs, nn_pairs, save_path=save_path)

        return fig

    def plot_figure3_panel_f(self, dataset_key=None,
                         burst_results=None,
                         config=None,
                         min_spikes_per_burst=2,
                         min_fraction_bursts=0.8,
                         min_total_spikes=30,
                         max_lag=0.35,
                         save_path=None):
        """
        Wrapper for Figure 3 Panel F.

        Method:
            (1) Retrieve cached bursts.
            (2) Identify backbone vs non-rigid units.
            (3) Compute pairwise CC matrix for all units.
            (4) Extract BB-BB pairs and all pairs.
            (5) Plot histogram of CC values (log scale) and mark mean BB-BB score.

        Args:
            dataset_key (str): Dataset key for analysis.
            burst_results (dict): Cached burst detection results.
            config (dict, optional): Analysis configuration.
            min_spikes_per_burst (int): Spike threshold for activity per burst.
            min_fraction_bursts (float): Fraction of bursts for backbone classification.
            min_total_spikes (int): Total spike count for backbone classification.
            max_lag (float): Max lag in seconds for CC computation.
            save_path (str, optional): Path to save Panel F plot.

        Outputs:
            fig (matplotlib.figure.Figure): Histogram figure for Panel F.
        """
        dataset_key = self._normalize_dataset_key(dataset_key)
        if dataset_key not in self.spike_data:
            print(f"Dataset '{dataset_key}' not found.")
            return None

        if burst_results is None or dataset_key not in burst_results:
            print(f"No cached bursts found for dataset '{dataset_key}'.")
            return None
        burst_windows = burst_results[dataset_key]["burst_windows"]
        if not burst_windows:
            print("No bursts detected for this dataset.")
            return None

        # Load spike data
        sd = self.spike_data[dataset_key]
        analysis = BurstAnalysisMicro(sd.train, fs=sd.metadata.get("fs", 10000), config=config)

        # Identify backbone units
        backbone_units, non_rigid_units = analysis.get_backbone_units(
            burst_windows,
            min_spikes_per_burst=min_spikes_per_burst,
            min_fraction_bursts=min_fraction_bursts,
            min_total_spikes=min_total_spikes
        )

        print(f"[Panel F] Backbone units: {len(backbone_units)}, Non-rigid units: {len(non_rigid_units)}")

        # Compute full CC matrix
        all_units = backbone_units + non_rigid_units
        cc_matrix, _ = analysis.compute_pairwise_ifr_cross_correlation(
            units=all_units,
            fs=sd.metadata.get("fs", 10000),
            max_lag=max_lag
        )

        # Extract all unique pairwise values (upper triangle)
        import numpy as np
        all_pairs = cc_matrix[np.triu_indices(len(all_units), k=1)]

        # Extract BB-BB pairs only
        bb_pairs, _, _ = analysis.separate_unit_pair_types(
            cc_matrix=cc_matrix,
            backbone_units=list(range(len(backbone_units)))
        )

        # Plot histogram
        fig = BAMicPlots.plot_panel_f(bb_pairs, all_pairs, save_path=save_path)

        return fig
    


















































# --------------------------------------------------------------------------------------
#                               HOLD HERE FOR NOW 
# --------------------------------------------------------------------------------------

    def plot_dim_overlay(self, dataset_keys=None, config=None, time_range=(0, 180), save=False, output_dir=None):
        dataset_keys = dataset_keys or list(self.spike_data.keys())
        for key in dataset_keys:
            detector = self.get_burst_detector(key, config)
            dim_bursts = detector.detect_dim_population_bursts()
            if not dim_bursts:
                print(f"No dim bursts found in '{key}'")
                continue
            sd = self.spike_data[key]
            BurstDetectionPlots.plot_overlay_raster_dim_bursts(trains=sd.train, bursts=dim_bursts, dataset_label=key, time_range=time_range, save=save,
                output_dir=output_dir)

    def plot_isi_overlay(self, dataset_keys=None, config=None, time_range=(0, 180), save=False, output_dir=None):
        dataset_keys = dataset_keys or list(self.spike_data.keys())
        for key in dataset_keys:
            detector = self.get_burst_detector(key, config)
            isi_bursts = detector.detect_isi_bursts(aggregate=True)
            if not isi_bursts:
                print(f" No ISI bursts found in '{key}'")
                continue
            sd = self.spike_data[key]
            BurstDetectionPlots.plot_overlay_raster_isi_bursts(trains=sd.train, bursts=isi_bursts, dataset_label=key, time_range=time_range, save=save,
                output_dir=output_dir)

    def highlight_dim_bursts_on_plot(self, dataset_key, window_size=2.0, step_size=0.5, min_active_neurons=10, min_spikes_per_neuron=1,
        overlay_color='cyan', overlay_alpha=0.3, ax=None):
        if dataset_key not in self.spike_data:
            raise ValueError(f"Dataset '{dataset_key}' not found.")

        detector = self.get_burst_detector(dataset_key)
        burst_windows = detector.detect_dim_population_bursts(window_size=window_size, step_size=step_size, 
                        min_active_neurons=min_active_neurons, min_spikes_per_neuron=min_spikes_per_neuron)

        ax = ax or plt.gca()
        for start, end in burst_windows:
            ax.axvspan(start, end, color=overlay_color, alpha=overlay_alpha)
        return burst_windows
