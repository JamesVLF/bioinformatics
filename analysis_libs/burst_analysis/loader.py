import zipfile
from pathlib import Path
import numpy as np
import os
import boto3
from spikedata.spikedata import SpikeData


class SpikeDataLoader:
    def __init__(self, zip_dir):
        self.zip_dir = Path(zip_dir)

    def load_and_extract_zip_stems(self):
        """
        Extract all acqm.zip files, label by filename stem, and return SpikeData objects.

        Returns:
            dict[str, SpikeData]: Mapping of stem → SpikeData instance
        """
        extracted = {}
        for zip_path in sorted(self.zip_dir.glob("*.zip")):
            stem = zip_path.stem
            out_path = self._extract_first_npz(zip_path, stem)
            if out_path:
                sd = self._load_spike_data(out_path)
                if sd:
                    extracted[stem] = sd
        return extracted

    def _extract_first_npz(self, zip_path, label):
        target_folder = zip_path.parent / "extracted"
        target_folder.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as z:
            npz_files = [f for f in z.namelist() if f.endswith(".npz")]
            if not npz_files:
                print(f"[WARN] No .npz in {zip_path.name}")
                return None
            z.extract(npz_files[0], path=target_folder)

        src = target_folder / npz_files[0]
        dst = target_folder / f"{label}.npz"
        src.rename(dst)
        return dst

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
            config=Config(signature_version='s3v4')
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