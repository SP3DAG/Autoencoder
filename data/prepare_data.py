import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
from datetime import datetime

def decode_lsb_from_image(path):
    img = Image.open(path).convert("RGBA")  # Important: force RGBA like the canvas
    data = np.array(img).flatten()

    bits = []
    for i in range(0, len(data), 4):  # RGBA stride
        bits.append(data[i] & 1)     # R
        bits.append(data[i + 1] & 1) # G
        bits.append(data[i + 2] & 1) # B
        # Skip Alpha (i + 3)

    bytes_list = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte = int(''.join(map(str, byte_bits)), 2)
        if byte == 0:
            break
        bytes_list.append(byte)

    try:
        message = bytes(bytearray(bytes_list)).decode("utf-8")
        return message
    except UnicodeDecodeError:
        return None

def parse_metadata(message):
    if not message:
        return None

    # Match message format from your app
    match = re.search(r"Captured at: (.*?) \| Location: ([\d\.\-]+), ([\d\.\-]+)", message)
    if not match:
        return None

    # Parse datetime string to UNIX timestamp
    dt_str = match.group(1)
    dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S %z")
    timestamp = int(dt.timestamp()) / 1e10

    lat = float(match.group(2))
    lon = float(match.group(3))

    return [timestamp, lat, lon]


def prepare_dataset(png_dir, output_dir, image_size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)

    X = []
    y = []
    metadata_json = {}

    for filename in tqdm(os.listdir(png_dir)):
        if not filename.lower().endswith(".png"):
            continue

        png_path = os.path.join(png_dir, filename)
        message = decode_lsb_from_image(png_path)
        metadata = parse_metadata(message)
        if metadata is None:
            print(f"[WARN] No valid metadata found in {filename}")
            continue

        # Save metadata
        metadata_json[filename] = metadata
        y.append(metadata)

        # Load original PNG and compress to JPEG
        img = Image.open(png_path).resize(image_size)
        jpeg_path = os.path.join(output_dir, filename.replace(".png", ".jpg"))
        img.convert("RGB").save(jpeg_path, format="JPEG", quality=50)

        # Load back JPEG as input data
        jpeg_arr = np.array(Image.open(jpeg_path).resize(image_size)) / 255.0
        X.append(jpeg_arr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Save
    np.save(os.path.join(output_dir, "X_train.npy"), X)
    np.save(os.path.join(output_dir, "y_train.npy"), y)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata_json, f, indent=2)

    print(f"✅ Saved {len(X)} samples to {output_dir}")


if __name__ == "__main__":
    prepare_dataset("data/raw", "data/processed", image_size=(128, 128))
