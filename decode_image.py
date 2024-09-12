import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps, ImageFile
import torch
from torchvision import transforms

# Allow truncated images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set values for BCH error correction
BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, help='Path to the decoder model')
    parser.add_argument('--image', type=str, default=None, help='Path to a single image file')
    parser.add_argument('--images_dir', type=str, default=None, help='Directory containing images')
    parser.add_argument('--secret_size', type=int, default=100, help='Expected size of the secret')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    args = parser.parse_args()

    # Determine which images to process
    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    # Load the decoder model
    try:
        device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
        decoder = torch.load(args.model, map_location=device)
        decoder.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize BCH library
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # Define image processing parameters
    width, height = 400, 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    # Disable gradient calculation
    with torch.no_grad():
        for filename in files_list:
            if 'hidden' not in filename:
                continue

            try:
                # Load and preprocess the image
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, size)
                image = to_tensor(image).unsqueeze(0).to(device)

                # Decode the secret
                secret = decoder(image).cpu().numpy().flatten()  # Ensure tensor is on CPU for processing
                secret = np.round(secret).astype(int)  # Round and convert to integers (0 or 1)

                # Extract the binary packet
                packet_binary = "".join(map(str, secret[:96]))  # Take the first 96 bits
                packet = bytearray(int(packet_binary[i:i + 8], 2) for i in range(0, len(packet_binary), 8))

                # Separate data and ECC
                data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
                bitflips = bch.decode_inplace(data, ecc)

                # Check if decoding was successful
                if bitflips != -1:
                    try:
                        code = data.decode("utf-8").strip()  # Strip trailing spaces
                        print(f"{filename}: {code}")
                    except UnicodeDecodeError:
                        print(f"{filename}: Failed to decode UTF-8")
                else:
                    print(f"{filename}: Failed to correct errors")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
