import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from collections import defaultdict
from burst_analysis.loading import SpikeDataLoader
from burst_analysis.detection import BurstDetection
from burst_analysis.computation import BurstAnalysisMacro
from burst_analysis.plotting import BurstDetectionPlots, BAMacPlots

class OrchestratorPDx2:
    def __init__(self, spike_paths, track_results=True):
        self.spike_paths = spike_paths
        self.loader = SpikeDataLoader(self.spike_paths)
        self.spike_data = self.loader.load()  
        self.track_results = track_results
        self.burst_detection_metrics_df = pd.DataFrame()


    def get_burst_detector(self, dataset_key, config=None):
        if dataset_key not in self.spike_data:
            raise ValueError(f"[ERROR] Dataset '{dataset_key}' not found.")
        
        sd = self.spike_data[dataset_key]
        fs = sd.metadata.get("fs", 10000)
        return BurstDetection(sd.train, fs=fs, config=config or {})

    def list_datasets(self):
        return list(self.spike_data.keys())

    def _normalize_dataset_key(self, key):
        if isinstance(key, np.ndarray) and key.size == 1:
            return key.item()
        return key

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
    #           LEVEL 1 - BURST DETECTION + FEATURE EXTRACTION
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
                print(f"[WARNING] Burst detection failed for '{key}'")
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
                print(f"[WARNING] No DIM bursts found in '{key}'")
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
            print("\nNo bursts detected.")

    def plot_isi_overlay(self, dataset_keys=None, config=None, time_range=(0, 180), save=False, output_dir=None):
        dataset_keys = dataset_keys or list(self.spike_data.keys())
        for key in dataset_keys:
            detector = self.get_burst_detector(key, config)
            isi_bursts = detector.detect_isi_bursts(aggregate=True)
            if not isi_bursts:
                print(f"[WARNING] No ISI bursts found in '{key}'")
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
    #           LEVEL 2 – BURST ANALYSIS MACRO
    # ----------------------------------------------------------

    def compute_and_plot_unit_participation(self, sample_configs, control_subjects=None,
                                            dataset_keys=None, ordered_short_labels=None,
                                            save=False, output_path=None):
        """
        Computes per-unit burst participation fractions across multiple datasets
        and generates a violin plot comparing CONTROL vs TREATED groups.

        Parameters
        ----------
        sample_configs : dict
            Dictionary mapping dataset identifiers (e.g., 'S1', 'S2') to analysis configuration objects
            used for burst detection computations.
        control_subjects : set, optional
            Set of subject IDs (e.g., {'s1','s2','s3'}) considered CONTROL group. 
            Remaining datasets are labeled TREATED. Default=None → no CONTROL grouping.
        dataset_keys : list, optional
            Specific dataset keys to analyze. Defaults to all loaded datasets.
        ordered_short_labels : list of str, optional
            Explicit order of x-axis sample labels (e.g., ['d-6s1','d1s1',...]).
            If None, ordering follows dataset key sorting.
        save : bool, optional
            If True, saves the resulting violin plot as PNG.
        output_path : str, optional
            File path to save the plot (used only if save=True).

        Returns
        -------
        grouped_participation : dict
            Mapping of 'GROUP_dataset' → list of per-unit participation fractions.
        """

        # Default args
        dataset_keys = dataset_keys or list(self.spike_data.keys())
        control_subjects = control_subjects or set()
        grouped_participation = defaultdict(list)

        # --- COMPUTATION LOOP ---
        for dataset_key in sorted(dataset_keys):
            if dataset_key not in self.spike_data:
                print(f"[SKIP] Dataset '{dataset_key}' not found in loaded data.")
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
        Wrapper function that orchestrates the full analysis pipeline:
        1. Builds peak time matrices for each dataset.
        2. Computes Spearman rank correlations and z-scores.
        3. Collects z-score distributions by condition.
        4. Generates violin plots of the results.

        Parameters
        ----------
        bb : object
            Analysis backend object with loaded spike data and helper methods.
        sample_configs : dict
            Configuration mapping for datasets (keyed by sample label).
        ordered_short_labels : list of str
            Dataset label order for plotting.
        control_subjects : set
            Subject IDs considered CONTROL for grouping.

        Notes
        -----
        - Iterates through all datasets in `bb.loader.spike_data`.
        - Filters bursts and units as per threshold rules.
        - Uses the three computational methods and plotting function
        to replicate the notebook analysis in structured form.
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

        # Generate final plot
        BAMacPlots.plot_zscore_distributions(zscore_distributions, ordered_short_labels, control_subjects)

    def run_similarity_analysis(self, dataset_key, config=None, window_size=3.0):
        if dataset_key not in self.spike_data:
            raise ValueError(f"Dataset '{dataset_key}' not found.")
        sd = self.spike_data[dataset_key]
        detector = self.get_burst_detector(dataset_key, config)
        _, _, _, peak_times, _, burst_windows = detector.compute_population_rate_and_bursts()
        if not burst_windows:
            print(f"[WARNING] No bursts detected for '{dataset_key}'.")
            return
        aligned_trains = detector.extract_bursts_from_raw_train(sd.train, burst_windows)
        sim_matrix = detector.compute_similarity_matrix(burst_windows=aligned_trains, peak_times=peak_times, window_size=window_size)
        BAMacPlots.plot_burst_similarity_matrix(sim_matrix, title="Burst Similarity (Cosine)")
