import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from collections import defaultdict
from burst_analysis.loading import SpikeDataLoader
from burst_analysis.detection import BurstDetection
from burst_analysis.computation import BurstAnalysisMacro
from burst_analysis.plotting import BurstDetectionPlots, BAMacPlots, BAMicPlots 

class OrchestratorPDx2:
    def __init__(self, spike_paths, track_results=True):
        self.spike_paths = spike_paths
        self.loader = SpikeDataLoader(self.spike_paths)
        self.spike_data = self.loader.load()  
        self.track_results = track_results
        self.burst_detection_metrics_df = pd.DataFrame()
        self.BAMac = None 

    def _init_BAMac(self, config=None):
        self.BAMac = BurstAnalysisMacro(
            self.sd_main.train,
            self.sd_main.metadata.get("fs", 10000),
            config = config if config is not None else {}
        )

    def get_burst_detector(self, dataset_key, config=None):
        if dataset_key not in self.spike_data:
            raise ValueError(f"Dataset '{dataset_key}' not found.")
        sd = self.spike_data[dataset_key]
        fs = sd.metadata.get("fs", 10000)
        return BurstDetection(sd.train, fs=fs, config=config or {})

    def list_datasets(self):
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

    def store_burst_results(self, results, dataset_key=None, overwrite=False):
        if not self.track_results:
            print("Tracking is disabled. Set `track_results=True` in constructor.")
            return
        if dataset_key is None:
            raise ValueError("Must provide dataset_key when tracking results.")
        if isinstance(results, dict):
            results = [results]
        for entry in results:
            entry["dataset_key"] = dataset_key
        self.burst_detection_metrics_df = pd.concat(
            [self.burst_detection_metrics_df, pd.DataFrame(results)],
            ignore_index=True)

    def get_dataset_keys(self, dataset_keys=None):
        if dataset_keys:
            return [k for k in dataset_keys if k in self.spike_data]
        return list(self.spike_data.keys())

    # -------------------------------------------------------------------------
    #           LEVEL 1: BURST DETECTION + FEATURE EXTRACTION
    # -------------------------------------------------------------------------

    def show_neuron_count_plot(self, group_input=None):
        BurstDetectionPlots.plot_neuron_counts(self.metadata_df, group_input, self.groups)

    def compute_and_plot_population_bursts(self, dataset_keys=None, config=None, time_range=(0, 180), save=False, output_dir=None,
                                           store_results=False):
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

            if store_results: # store metrics to cache or df
                self.store_burst_results(metrics, dataset_key=key)

            sd = self.spike_data[key]
            BurstDetectionPlots.plot_overlay_raster_population(trains=sd.train, times=times, smoothed=smoothed, bursts=bursts,
                dataset_label=key, time_range=time_range, save=save, output_dir=output_dir)
            all_metrics[key] = metrics
        return all_metrics

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

    # ----------------------------------------------------------
    #           LEVEL 2: BURST ANALYSIS MACRO
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
                print(f"[SKIP] No config for dataset {dataset_key}")
                continue

            # Initialize burst detector and run population burst detection
            detector = self.get_burst_detector(dataset_key, config)
            result = detector.compute_population_rate_and_bursts()
            if result is None or result[-1] is None:
                print(f"[SKIP] Burst computation failed for {dataset_key}")
                continue

            burst_windows = result[-1]
            if not burst_windows:
                print(f"[NOTE] No bursts found in {dataset_key}")
                continue

            # Compute unit participation using BurstAnalysisMacro
            trains = self.spike_data[dataset_key].train
            analysis = BurstAnalysisMacro(trains, fs=detector.fs, config=config)
            fractions = analysis.compute_unit_participation(burst_windows)

            # Group CONTROL vs TREATED
            group = "CONTROL" if subject_label in control_subjects else "TREATED"
            label = f"{group}_{parts}"
            grouped_participation[label].extend(fractions)
            print(f"[DONE] {dataset_key}: {len(burst_windows)} bursts, {len(fractions)} units")

        # --- PLOTTING ---
        BAMacPlots.plot_violin_burst_participation(
            grouped_participation=grouped_participation,
            ordered_short_labels=ordered_short_labels,
            save=save,
            output_path=output_path
        )
        return grouped_participation
    
    def run_burst_rank_order_analysis(bb, sample_configs, ordered_short_labels, control_subjects):
        """
       Method: 
            1. Builds peak time matrices for each dataset
            2. Computes Spearman rank correlations and z-scores
            3. Collects z-score distributions by condition
            4. Generates violin plots of the results

        Args:
            orch: object of backend analysis  with loaded spike data and helper methods
            sample_configs: dict mapping of configurations for datasets (keyed by sample label)
            ordered_short_labels: list of str dataset labels ordered for plotting
            control_subjects: set of IDs designated as CONTROL for grouping

        Notes:
            - Iterates through all datasets in `orch.loader.spike_data`
            - Filters bursts and units in accord with threshold rules
            - Uses 3 methods and plotting functions for the notebook UI function
        """

        zscore_distributions = defaultdict(list)

        for dataset_key in sorted(bb.loader.spike_data.keys()):
            bb.loader.set_dataset(dataset_key)

            parts = dataset_key.split("_")[0]
            sample_num = parts.split("s")[-1]
            sample_label = f"S{sample_num}"
            config = sample_configs.get(sample_label)

            if config is None:
                continue

            try:
                analysis = bb._get_analysis(config=config)
                result = analysis.compute_population_rate_and_bursts()
                times, _, _, _, _, burst_windows = result
            except Exception:
                continue

            if not burst_windows or len(burst_windows) < 3:
                continue

            trains = bb.sd_main.train
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

        burst_stack, avg_matrix, peak_times, peak_indices = analysis.compute_burst_aligned_ifr_matrix(
            **(config or {})
        )

        if plot:
            BAMacPlots.plot_burst_aligned_ifr_matrix(avg_matrix, peak_times)

        return burst_stack, avg_matrix, peak_times, peak_indices

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
#           LEVEL 3: BURST ANALYSIS MICRO- Single Bursts + Unit Relationships
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
