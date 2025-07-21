from burst_analysis.loader import SpikeDataLoader
from burst_analysis.detector import detect_population_bursts


class BurstDetection:
    def __init__(self, zip_dir, config=None):
        """
        Handles loading and burst detection for acqm.zip spike datasets.

        Args:
            zip_dir (str or Path): Directory containing acqm.zip files.
            config (dict, optional): Detection parameters.
        """
        self.zip_dir = zip_dir
        self.config = config or {}
        self.datasets = {}  # {stem: SpikeData}

    def load(self):
        """Unzip and load all acqm.zip files from the directory."""
        loader = SpikeDataLoader(self.zip_dir)
        self.datasets = loader.load_and_extract_zip_stems()

    def run(self):
        """
        Run burst detection on all loaded SpikeData instances.

        Returns:
            dict[str, list[dict]]: Mapping of file stem to burst list.
        """
        results = {}
        for stem, sd in self.datasets.items():
            bursts = detect_population_bursts(
                sd,
                bin_size_ms=self.config.get("bin_size_ms", 10),
                smooth_sigma=self.config.get("smooth_sigma", 2),
                threshold_std=self.config.get("threshold_std", 2.0),
                min_duration_ms=self.config.get("min_duration_ms", 30),
            )
            results[stem] = bursts
        return results
