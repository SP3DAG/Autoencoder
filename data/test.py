from PIL import Image
import numpy as np

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
    
print(decode_lsb_from_image("data/raw/IMG_1869.PNG"))
