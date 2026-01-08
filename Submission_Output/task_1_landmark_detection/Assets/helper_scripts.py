import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_missing_images(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    df['exists'] = df['image_name'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x)))
    missing_imgs = df[~df['exists']]
    print(f"Missing images: {len(missing_imgs)}")
    return missing_imgs

def compute_image_stats(image_dir, image_names):
    means = []
    stds = []
    for img_name in tqdm(image_names, desc="Computing stats"):
        img = cv2.imread(os.path.join(image_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            means.append(img.mean())
            stds.append(img.std())
    return means, stds

def plot_landmark_density(df, image_dir, img_size=256):
    heatmaps = {
        "OFD1": np.zeros((img_size, img_size)),
        "OFD2": np.zeros((img_size, img_size)),
        "BPD1": np.zeros((img_size, img_size)),
        "BPD2": np.zeros((img_size, img_size)),
    }

    for row in df.itertuples():
        img = cv2.imread(os.path.join(image_dir, row.image_name), 0)
        if img is None: continue
        h, w = img.shape

        scale_x = img_size / w
        scale_y = img_size / h

        # Note: Using corrected columns if available, else raw
        # Assuming raw CSV columns here for EDA
        heatmaps["OFD1"][int(row.ofd_1_y * scale_y), int(row.ofd_1_x * scale_x)] += 1
        heatmaps["OFD2"][int(row.ofd_2_y * scale_y), int(row.ofd_2_x * scale_x)] += 1
        heatmaps["BPD1"][int(row.bpd_1_y * scale_y), int(row.bpd_1_x * scale_x)] += 1
        heatmaps["BPD2"][int(row.bpd_2_y * scale_y), int(row.bpd_2_x * scale_x)] += 1

    plt.figure(figsize=(10,10))
    for i, (k, hm) in enumerate(heatmaps.items()):
        plt.subplot(2,2,i+1)
        plt.imshow(hm, cmap="hot")
        plt.title(f"{k} Density")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
