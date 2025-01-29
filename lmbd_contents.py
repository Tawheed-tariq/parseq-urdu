import lmdb
import numpy as np
from PIL import Image
import io

def inspect_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        num_samples = int(txn.get(b'num-samples').decode())
        print(f"Number of samples in LMDB: {num_samples}")

        # Inspect first few entries
        for i in range(1, min(num_samples + 1, 6)):  # Inspect first 5 sample
            image_key = f'image-{i:09d}'.encode()
            label_key = f'label-{i:09d}'.encode()

            image_bin = txn.get(image_key)
            label = txn.get(label_key)

            if image_bin is None or label is None:
                print(f"Sample {i} is missing or invalid.")
                continue

            try:
                img = Image.open(io.BytesIO(image_bin)).convert('RGB')
                print(f"Sample {i}: Label = {label.decode()}, Image Size = {img.size}")
            except Exception as e:
                print(f"Error loading image {i}: {e}")

inspect_lmdb('/DATA/Tawheed/parseq_data/train/train_trdg10')
