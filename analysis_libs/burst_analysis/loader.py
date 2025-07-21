import zipfile
from pathlib import Path
import numpy as np
import os
import boto3
from botocore.client import Config  
from spikedata.spikedata import SpikeData


from pathlib import Path
import numpy as np

class SpikeDataLoader:
    def __init__(self, npz_dir):
        self.npz_dir = Path(npz_dir)

    def load(self):
        """Load all .npz files from the directory and return {stem: SpikeData}"""
        extracted = {}

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
