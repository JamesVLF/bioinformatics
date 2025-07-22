import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from analysis_libs.burst_analysis.loader import SpikeDataLoader
from analysis_libs.burst_analysis.detector import BurstDetection
from analysis_libs.burst_analysis.plots import plot_overlay_raster_population 
from analysis_libs.burst_analysis.plots import (plot_neuron_counts, 
                                                plot_overlay_raster_dim_bursts, 
                                                plot_overlay_raster_isi_bursts, 
                                                plot_burst_similarity_matrix) 

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

    # -----------------------------------------------
    # Level 1 – Population Bursts + Raster Overview
    # -----------------------------------------------

    def show_neuron_count_plot(self, group_input=None):
        plot_neuron_counts(self.metadata_df, group_input, self.groups)

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
            plot_overlay_raster_population(trains=sd.train, times=times, smoothed=smoothed, bursts=bursts,
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
            plot_overlay_raster_dim_bursts(trains=sd.train, bursts=dim_bursts, dataset_label=key, time_range=time_range, save=save,
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
            plot_overlay_raster_isi_bursts(trains=sd.train, bursts=isi_bursts, dataset_label=key, time_range=time_range, save=save,
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

    # -----------------------------------------------
    # Level 2 – Population Burst Analysis
    # -----------------------------------------------

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
        plot_burst_similarity_matrix(sim_matrix, title="Burst Similarity (Cosine)")
