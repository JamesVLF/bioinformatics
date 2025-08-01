import os
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import boto3
from botocore.client import Config  
from spikedata.spikedata import SpikeData
import scipy.io as sio
from burst_analysis.attributes import NeuronAttributes

class SpikeDataLoader:
    def __init__(self, npz_dir, spike_paths=None):
        if isinstance(npz_dir, dict):
            self.spike_paths = npz_dir
            self.npz_dir = None
        else:
            self.npz_dir = Path(npz_dir)
            self.spike_paths = spike_paths or {}
        self.spike_data = {}
        self.sd_main = None
        # self.metadata_df = pd.DataFrame()
        # self.neuron_df = pd.DataFrame()
        self.groups = {}


    def load(self):
        """Load all .npz files either from self.spike_paths or directory."""
        extracted = {}
        if self.spike_paths:  
            for key, path in self.spike_paths.items():
                sd = self._load_spike_data(Path(path))
                if sd:
                    extracted[key] = sd
        elif self.npz_dir:  
            for npz_path in sorted(self.npz_dir.glob("*.npz")):
                stem = npz_path.stem
                sd = self._load_spike_data(npz_path)
                if sd:
                    extracted[stem] = sd
        return extracted

    def _load_spike_data(self, npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            train = [times / data["fs"] for times in data["train"].item().values()]
            return SpikeData(train)
        except Exception as e:
            print(f"[ERROR] Failed to load {npz_path.name}: {e}")
            return None

    @staticmethod
    def fetch_zips_from_s3(uuid_list, local_dir, bucket='braingeneers'):
        s3 = boto3.client(
            's3',
            endpoint_url='https://s3.braingeneers.gi.ucsc.edu',
            config=Config(signature_version='s3v4')  # ← this line requires `from botocore.client import Config`
        )

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        for uuid in uuid_list:
            prefix = f"ephys/{uuid}/"
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    filename = Path(key).name

                    if not filename.endswith('_acqm.zip'):
                        continue

                    dest = local_dir / filename
                    if dest.exists():
                        print(f"[SKIP] {filename} already exists")
                        continue

                    print(f"[DL] {key} → {dest}")
                    s3.download_file(bucket, key, str(dest))
    
    @staticmethod
    def list_zips_on_s3(uuid_list, bucket='braingeneers'):
        s3 = boto3.client(
            's3',
            endpoint_url='https://s3.braingeneers.gi.ucsc.edu',
            config=Config(signature_version='s3v4')
        )

        all_keys = []
        for uuid in uuid_list:
            prefix = f"ephys/{uuid}/"
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('_acqm.zip'):
                        all_keys.append(key)

        return all_keys    

    def set_dataset(self, name):
        if name in self.spike_data:
            self.sd_main = self.spike_data[name]
            print(f"Switched to dataset: {name}")
        else:
            raise ValueError(f"Dataset '{name}' not found.")

    def sd_main_key(self):
        for k, v in self.spike_data.items():
            if v is self.sd_main:
                return k
        return None

    def resolve_group_input(self, group_input):
        if not hasattr(self, "metadata_df") or self.metadata_df.empty:
            print("Metadata not available.")
            return None

        df = self.metadata_df.copy()

        if group_input is None:
            return df

        if isinstance(group_input, str):
            if group_input not in self.groups:
                print(f"Group '{group_input}' not found.")
                return None
            keys = self.groups[group_input]
            return df[df["dataset_key"].isin(keys)]

        if isinstance(group_input, list):
            return df[df["dataset_key"].isin(group_input)]

        if isinstance(group_input, pd.DataFrame):
            if "dataset_key" in group_input.columns:
                keys = group_input["dataset_key"].unique()
                return df[df["dataset_key"].isin(keys)]
            else:
                print("Provided DataFrame lacks 'dataset_key' column.")
                return None

        print("Invalid group_input type.")
        return None

    def load_data(self, spike_paths=None):
        paths = spike_paths or self.spike_paths
        self.spike_data = self.load_and_align_spike_data(paths)

    def load_and_align_spike_data(self, paths):
        spike_data_dict = {}

        for file_id, path in paths.items():
            try:
                data = np.load(path, allow_pickle=True)
                spike_trains = data["train"].item()
                neuron_data = data["neuron_data"].item()
                config = data.get("config", None)
                fs = data["fs"].item()
            except Exception as e:
                print(f"Error loading {file_id}: {e}")
                continue

            aligned_trains = []
            neuron_attrs = []

            for unit_key, meta in neuron_data.items():
                cluster_id = meta.get("cluster_id")
                if cluster_id is None or cluster_id not in spike_trains:
                    print(f"Skipping unit in {file_id}: missing or unmatched cluster_id {cluster_id}")
                    continue

                aligned_trains.append(np.array(spike_trains[cluster_id]) / fs)  # convert to seconds

                attr = NeuronAttributes(
                    cluster_id=meta["cluster_id"],
                    channel=meta["channel"],
                    position=meta["position"],
                    template=meta["template"],
                    amplitudes=meta["amplitudes"],
                    waveforms=meta["waveforms"],
                    neighbor_channels=meta["neighbor_channels"],
                    neighbor_positions=meta["neighbor_positions"],
                    neighbor_templates=meta["neighbor_templates"]
                )
                neuron_attrs.append(attr)

            if not aligned_trains:
                print(f"No aligned units found in {file_id}")
                continue

            spike_data = SpikeData(
                aligned_trains,
                N=len(aligned_trains),
                neuron_attributes=neuron_attrs,
                metadata={
                    "cluster_ids": [attr.cluster_id for attr in neuron_attrs],
                    "fs": fs,
                    "config": config
                }
            )

            spike_data_dict[file_id] = spike_data
            print(f"Loaded spike data for {file_id} with {len(aligned_trains)} units.")

        return spike_data_dict

    def build_metadata_df(self):
        rows = []
        time_lookup = {"baseline": 0, "1hr": 1, "3hr": 3, "24hr": 24, "48hr": 48}

        for key, sd in self.spike_data.items():
            parts = key.split("_")
            key_lower = key.lower()

            try:
                sample = parts[0]
                group = parts[1]
                time_part = next(p for p in parts if 'hr' in p or 'baseline' in p.lower())
                time_hr = time_lookup[time_part.lower()]
                config = 'connected' if 'connected' in key_lower else 'newconfig'

                rows.append({
                    "sample": sample,
                    "group": group,
                    "time_hr": time_hr,
                    "config": config,
                    "dataset_key": key,
                    "neuron_count": sd.N
                })

            except Exception as e:
                print(f"Skipping {key}: {e}")
                continue

        self.metadata_df = pd.DataFrame(rows)

    def define_dataset_group(self, group_name, filter_func):
        if not hasattr(self, 'metadata_df'):
            raise RuntimeError("metadata_df not built yet.")
        group_keys = self.metadata_df[filter_func(self.metadata_df)]["dataset_key"].tolist()
        self.groups[group_name] = group_keys

    def build_neuron_df(self):
        rows = []
        for ds_key, sd in self.spike_data.items():
            for i, attr in enumerate(sd.neuron_attributes or []):
                rows.append({
                    "dataset_key": ds_key,
                    "neuron_index": i,
                    "cluster_id": attr.cluster_id,
                    "channel": attr.channel,
                    "position": attr.position,
                })

        df = pd.DataFrame(rows)

        # Enrich with metadata from metadata_df
        if hasattr(self, "metadata_df") and not self.metadata_df.empty:
            df = pd.merge(df, self.metadata_df, on="dataset_key", how="left")
        else:
            print("Warning: metadata_df not available. neuron_df will lack context info.")

        self.neuron_df = df

    def get_unit_info(self, dataset_name=None):
        sd = self.sd_main if dataset_name is None else self.spike_data.get(dataset_name)
        if not sd:
            print(f"Dataset '{dataset_name}' not found.")
            return

        for i, attr in enumerate(sd.neuron_attributes or []):
            print(f"Neuron {i}: cluster_id={attr.cluster_id}, channel={attr.channel}")

    def define_group(self, group_name, dataset_to_units):
        self.groups[group_name] = dataset_to_units

    def extract_and_label_zips(base_folder, target_subdir="extracted_data", label_parts=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
        """
        Extracts all .zip files in a given base folder, renames the extracted .npz files using
        a label derived from the zip filename, and returns a mapping of labels to extracted file paths.

        Parameters:
        - base_folder (str or Path): Directory containing the .zip files.
        - target_subdir (str): Subfolder name under base_folder to store extracted files.
        - label_parts (tuple): Indices of the filename parts (split by '_') to use for labeling.

        Returns:
        - extracted_files (dict): Mapping from label to Path of extracted .npz file.
        """
        base_folder = Path(base_folder)
        target_folder = base_folder / target_subdir
        zip_paths = sorted(base_folder.glob("*.zip"))

        if not zip_paths:
            print(f"No .zip files found in {base_folder}")
            return {}

        zip_label_map = {}
        for zp in zip_paths:
            parts = zp.stem.split("_")
            if max(label_parts) < len(parts):
                label = "_".join(parts[i] for i in label_parts)
                zip_label_map[zp.name] = label
            else:
                print(f"[WARN] Skipping {zp.name}, not enough parts for label.")

        extracted_files = {}
        for zip_name, label in zip_label_map.items():
            out = extract_zip(
                zip_filename  = zip_name,
                base_folder   = base_folder,
                target_folder = target_folder,
                label         = label
            )
            if out:
                extracted_files[label] = out

        print("Extraction complete")
        return extracted_files

    def load_pickle(path):
        """Load a pickle file from the given path."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    def add_label(obj, label):
        """Attach a label to a dataset (as attribute or dict key)."""
        if hasattr(obj, '__dict__'):
            setattr(obj, 'label', label)
        elif isinstance(obj, dict):
            obj['__label__'] = label
        return obj

    def load_datasets_key(paths):
        """Load generic datasets from a dictionary of {label: path}."""
        datasets = {}
        for key, path in paths.items():
            obj = load_pickle(path)
            datasets[key] = add_label(obj, key)
        return datasets

    def load_datasets(paths):
        """Load datasets from {label: path} and attach label."""
        datasets = {}
        for label, path in paths.items():
            obj = load_pickle(path)
            datasets[label] = add_label(obj, label)
        return datasets



    def load_spike_data(spike_paths):
        """Load spike data with labels and type info."""
        spike_data = {}
        for label, path in spike_paths.items():
            if os.path.exists(path):
                data = load_pickle(path)
                spike_data[label] = add_label(data, label)
                print(f"Loaded spike data for {label}: {type(data)}")
            else:
                print(f"Missing spike file for {label}")
        return spike_data

    def load_curation(qm_path):
        """
        Load spike data from a curation zip file (acqm).

        Parameters:
        qm_path (str): Path to the .zip file containing spike data.

        Returns:
        tuple: (train, neuron_data, config, fs)
            train (list): List of spike times arrays (seconds).
            neuron_data (dict): Neuron metadata.
            config (dict or None): Configuration dictionary if present.
            fs (float): Sampling frequency.
        """
        with zipfile.ZipFile(qm_path, 'r') as f_zip:
            qm = f_zip.open("qm.npz")
            data = np.load(qm, allow_pickle=True)
            spike_times = data["train"].item()
            fs = data["fs"]
            train = [times / fs for _, times in spike_times.items()]
            config = data["config"].item() if "config" in data else None
            neuron_data = data["neuron_data"].item()
        return train, neuron_data, config, fs


        acqm_path = "./23126c_D44_KOLFMO_5272025_acqm.zip"

        # Load data from acqm file
        train_acqm, neuron_data, config, fs = load_curation(acqm_path)
        sd_acqm = SpikeData(train_acqm)

        embed()

    def extract_zip(zip_filename, base_folder, target_folder, label):
        """
        Extracts ONLY the first .npz file from a ZIP archive and renames it to match the experiment label.
        """
        from pathlib import Path
        import zipfile

        base_folder = Path(base_folder)
        target_folder = Path(target_folder)
        zip_filename = Path(zip_filename)
        zip_path = base_folder / zip_filename

        if not zip_path.exists():
            print(f"File not found: {zip_path}")
            return None

        target_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            # Find all .npz files in the zip
            npz_members = [m for m in z.namelist() if m.endswith(".npz")]
            if not npz_members:
                print(f"[WARN] No .npz file found in {zip_path.name}")
                return None

            # Extract only the first .npz file
            npz_name = npz_members[0]
            z.extract(npz_name, path=target_folder)

        src = target_folder / npz_name
        dst = target_folder / f"{label}.npz"
        src.replace(dst)

        print(f"Extracted and renamed {npz_name} → {dst.name}")
        return dst

    def normalize_conditions(conditions, data_dict):
        if conditions is None:
            return list(data_dict.keys())
        elif isinstance(conditions, str):
            return [conditions] if conditions in data_dict else []
        elif isinstance(conditions, list):
            return [c for c in conditions if c in data_dict]
        else:
            return []

    def load_npz_data(file_path):
        data = np.load(file_path, allow_pickle=True)
        spike_times_sec = {neuron_id: spikes / data["fs"] for neuron_id, spikes in data["train"].item().items()}
        return spike_times_sec, data["neuron_data"].item(), data["fs"]

    def load_all_data(file_dict_or_folder, conditions=None):
        """
        Load multiple labeled .npz spike datasets.

        Accepts either a {label: path} dict or a folder path of labeled .npz files.
        """
        from pathlib import Path

        # If a folder was passed, build a file_dict
        if isinstance(file_dict_or_folder, (str, Path)):
            folder = Path(file_dict_or_folder)
            if not folder.exists():
                print(f"[ERROR] Folder does not exist: {folder}")
                return {}
            npz_files = sorted(folder.glob("*.npz"))
            file_dict = {f.stem: f for f in npz_files}
        elif isinstance(file_dict_or_folder, dict):
            file_dict = file_dict_or_folder
        else:
            print("[ERROR] Invalid input type to load_all_data")
            return {}

        selected_conditions = normalize_conditions(conditions, file_dict)

        if not selected_conditions:
            print("No valid dataset(s) found. Available options:", list(file_dict.keys()))
            return {}

        loaded_data = {}
        for condition in selected_conditions:
            print(f"Loading {condition} ...")
            spike_times_sec, neuron_data, fs = load_npz_data(file_dict[condition])
            loaded_data[condition] = {
                "spike_times_sec": spike_times_sec,
                "neuron_data": neuron_data,
                "fs": fs
            }
            print(f"Loaded {len(spike_times_sec)} neurons for {condition}.")

        print("\nSelected datasets successfully loaded.\n")
        return loaded_data


    def mat_to_spikeData(mat_path):
        mat = sio.loadmat(mat_path)
        units = [i[0][0]*1e3 for i in mat['spike_times']]
        sd = SpikeData(units)
        return sd

    def inspect_dataset(condition, data_dict):
        """
        Prints basic information about a spike dataset.

        Parameters:
        - condition (str): Dataset key to inspect (e.g., 'Baseline').
        - data_dict (dict): Dictionary containing SpikeData instances.
        """
        if condition not in data_dict:
            print(f"[ERROR] Condition '{condition}' not found in provided data.")
            return

        sd = data_dict[condition]
        try:
            unit_ids, spike_times = sd.idces_times()
        except Exception as e:
            print(f"[WARNING] Could not extract unit IDs or spike times: {e}")
            unit_ids, spike_times = [], []

        print("\n--- Dataset Inspection ---")
        print(f"Condition           : {condition}")
        print(f"Number of Neurons   : {sd.N}")
        print(f"Recording Length    : {sd.length / 1000:.2f} seconds")
        print(f"Sample Unit IDs     : {unit_ids[:10]}")
        print(f"Sample Spike Times  : {spike_times[:10]}")
        print(f"Available Attributes: {dir(sd)}")
        print("-----------------------------")


    def inspect_datasets(conditions=None, data_dict=None):
        """
        Inspects one, multiple, or all datasets.

        Parameters:
        - conditions (str, list, or None): Dataset condition(s) to inspect.
        - data_dict (dict): Dictionary containing all loaded datasets.
        """
        if not data_dict:
            print("No datasets loaded.")
            return

        # Normalize input
        conditions = normalize_conditions(conditions, data_dict)

        for condition in conditions:
            inspect_dataset(condition, data_dict)
            print("-" * 50)