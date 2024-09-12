import os
import glob
import bchlib
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

# set values for BCH
BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./images')
    parser.add_argument('--secret', type=str, default='Stega!!')
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--cuda', action='store_true')  # Changed to action flag
    args = parser.parse_args()

    # Determine image list
    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(os.path.join(args.images_dir, '*'))
    else:
        print('Missing input image')
        return

    # Load the encoder model
    try:
        encoder = torch.load(args.model, map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        encoder.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize BCH library
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # Check secret length
    if len(args.secret) > 7:
        print('Error: Can only encode 56 bits (7 characters) with ECC')
        return

    # Encode the secret
    data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])  # Padding
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
    if args.cuda and torch.cuda.is_available():
        secret = secret.cuda()

    # Define image size and transformation
    width, height = 400, 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    # Create save directory if it does not exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Process each image
    with torch.no_grad():
        for filename in files_list:
            try:
                # Open and preprocess the image
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, size)
                image = to_tensor(image).unsqueeze(0)
                if args.cuda and torch.cuda.is_available():
                    image = image.cuda()

                # Generate encoded image and residuals
                residual = encoder((secret, image))
                encoded = image + residual
                if args.cuda and torch.cuda.is_available():
                    residual = residual.cpu()
                    encoded = encoded.cpu()

                # Clamp pixel values and convert to uint8
                encoded = np.array(torch.clamp(encoded, 0, 1).squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))
                residual = np.array((residual[0] + 0.5).clamp(0, 1).squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

                # Save encoded and residual images
                save_name = os.path.splitext(os.path.basename(filename))[0]
                Image.fromarray(encoded).save(os.path.join(args.save_dir, f'{save_name}_hidden.png'))
                Image.fromarray(residual).save(os.path.join(args.save_dir, f'{save_name}_residual.png'))

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()

