import os, glob
import numpy as np
import tifffile as tf
from PIL import Image
import csv

ROOT = "/home/claude/oscd/Onera Satellite Change Detection dataset - Images"
OUT  = "/home/claude/work/landcover_dataset"
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/images_rgb", exist_ok=True)
os.makedirs(f"{OUT}/images_multiband_npy", exist_ok=True)
os.makedirs(f"{OUT}/masks", exist_ok=True)
os.makedirs(f"{OUT}/masks_colored", exist_ok=True)

CLASS_NAMES = {0:"Other/Unclassified",1:"Water",2:"Urban",3:"Forest",4:"Agriculture",5:"Barren land"}
CLASS_COLORS = {
    0:(60,60,60),      # dark gray
    1:(0,90,255),       # water - blue
    2:(230,30,30),       # urban - red
    3:(0,130,0),         # forest - dark green
    4:(190,230,60),      # agriculture - light green/yellow
    5:(180,140,90),       # barren - tan/brown
}

def read_band(path):
    return tf.imread(path).astype(np.float32)

def load_aoi_bands(aoi_dir):
    # returns dict bandname -> array (float, TOA reflectance scale ~0-1)
    bands = {}
    for b in ["B02","B03","B04","B08","B11"]:
        p = os.path.join(aoi_dir, f"{b}.tif")
        bands[b] = read_band(p) / 10000.0
    return bands

def safe_div(a, b):
    return a / np.where(b==0, 1e-6, b)

def classify(bands):
    B02, B03, B04, B08, B11 = bands["B02"], bands["B03"], bands["B04"], bands["B08"], bands["B11"]

    ndvi = safe_div(B08 - B04, B08 + B04)
    ndwi = safe_div(B03 - B08, B03 + B08)      # McFeeters
    ndbi = safe_div(B11 - B08, B11 + B08)      # Xu
    bsi  = safe_div((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02))

    h, w = ndvi.shape
    label = np.zeros((h, w), dtype=np.uint8)  # default 0 = other

    water_mask = ndwi > 0.10
    forest_mask = (~water_mask) & (ndvi > 0.50)
    agri_mask   = (~water_mask) & (~forest_mask) & (ndvi > 0.20)
    urban_mask  = (~water_mask) & (~forest_mask) & (~agri_mask) & (ndbi > 0.05) & (ndvi < 0.15)
    barren_mask = (~water_mask) & (~forest_mask) & (~agri_mask) & (~urban_mask) & (bsi > 0.10)

    label[water_mask]  = 1
    label[urban_mask]  = 2
    label[forest_mask] = 3
    label[agri_mask]   = 4
    label[barren_mask] = 5
    return label

def to_rgb_uint8(bands):
    R, G, B = bands["B04"], bands["B03"], bands["B02"]
    rgb = np.stack([R, G, B], axis=-1)
    # simple percentile stretch for visualization
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)

def colorize(label):
    h, w = label.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS.items():
        out[label == cid] = color
    return out

aois = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
summary_rows = []

for aoi in aois:
    for t, subfolder in [("1", "imgs_1_rect"), ("2", "imgs_2_rect")]:
        aoi_dir = os.path.join(ROOT, aoi, subfolder)
        if not os.path.isdir(aoi_dir):
            continue
        bands = load_aoi_bands(aoi_dir)
        label = classify(bands)
        rgb = to_rgb_uint8(bands)
        nir = np.clip(bands["B08"] * 3.0, 0, 1)  # stretch NIR a bit for viz-friendly npy

        sample_id = f"{aoi}_t{t}"

        # RGB preview
        Image.fromarray(rgb).save(f"{OUT}/images_rgb/{sample_id}.png")

        # multiband npy (R,G,B,NIR) float32, 0-1 scale, for model training
        multiband = np.stack([bands["B04"], bands["B03"], bands["B02"], bands["B08"]], axis=-1).astype(np.float32)
        np.save(f"{OUT}/images_multiband_npy/{sample_id}.npy", multiband)

        # mask (single channel, class ids 0-5)
        Image.fromarray(label, mode="L").save(f"{OUT}/masks/{sample_id}_mask.png")

        # colored mask for QA
        Image.fromarray(colorize(label)).save(f"{OUT}/masks_colored/{sample_id}_mask_color.png")

        total = label.size
        row = {"sample_id": sample_id, "aoi": aoi, "timepoint": t, "height": label.shape[0], "width": label.shape[1]}
        for cid, name in CLASS_NAMES.items():
            row[f"pct_{name}"] = round(100.0 * np.sum(label == cid) / total, 2)
        summary_rows.append(row)
        print(f"done {sample_id}  shape={label.shape}")

# write summary CSV
csv_path = f"{OUT}/labels_summary.csv"
fieldnames = list(summary_rows[0].keys())
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

print("ALL DONE. Samples:", len(summary_rows))
