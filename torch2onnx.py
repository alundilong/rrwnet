from pathlib import Path
import argparse

import torch
from skimage import io
import numpy as np

from model import RRWNet
from utils import pad_images_unet, to_torch_tensors

def export_to_onnx(model, example_image_tensor, output_path="tmp"):
    """
    Export the PyTorch model to ONNX format using an example processed image tensor.
    
    Args:
        model: PyTorch model (in eval mode)
        example_image_tensor: A processed image tensor that matches the model's expected input
        output_path: Path to save the ONNX model
    """
    output_path = f'{output_path}/models/'
    # Ensure the output directory exists
    Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    output_path = f'{output_path}/vessel_seg.onnx'
    
    # Export the model
    print(example_image_tensor.shape)
    torch.onnx.export(
        model,                     # PyTorch model
        example_image_tensor,      # Example processed input
        output_path,               # Output ONNX file path
        export_params=True,        # Store trained weights
        opset_version=12,          # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=["input"],     # Input tensor name
        output_names=["output"],   # Output tensor name
        dynamic_axes={
            "input": {0: "batch_size"},  # Allow dynamic batch size
            "output": {0: "batch_size"},
        },
    )
    print(f"Model successfully exported to {output_path}")

# example usage: python torch2onnx.py --weights shared_data\weights\rrwnet_RITE_1.pth --images-path data\images --masks-path data\masks --save-path tmp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get predictions from a model')
    parser.add_argument('--weights', type=str, required=True,
        help='Path to the model weights')
    parser.add_argument('--images-path', type=str, required=True,
        help='Path to the images')
    parser.add_argument('--masks-path', type=str, required=True,
        help='Path to the masks')
    parser.add_argument('--save-path', type=str, required=True,
        help='Path to ONNX model')
    args = parser.parse_args()

    model = RRWNet()

    print(f'Loading model from {args.weights}')
    model.load_state_dict(torch.load(args.weights), strict=True)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'

    print(f'Creating save path {args.save_path}')
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    print(f'Getting images and masks from {args.images_path} and {args.masks_path}')
    image_fns = sorted(Path(args.images_path).glob('*.png'))
    mask_fns = sorted(Path(args.masks_path).glob('*.png'))

    # We'll use the first image to create our example tensor for ONNX export
    if image_fns:
        image_fn = image_fns[0]
        mask_fn = mask_fns[0]
        
        print(f'Processing example image {image_fn.name} for ONNX export')
        
        img = (io.imread(image_fn) / 255.0)[..., :3]
        mask = io.imread(mask_fn) * 1.0
            
        # Process through the same pipeline as normal inference
        imgs, paddings = pad_images_unet([img, mask])
        img = imgs[0]
        padding = paddings[0]
        mask = imgs[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        tensors = to_torch_tensors([img, mask])
        image_tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        else:
            tensor = image_tensor.cpu()
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        
        if device == 'cuda':
            image_tensor = image_tensor.cuda()
        else:
            image_tensor = image_tensor.cpu()
        
        # Export to ONNX using this processed tensor as example
        export_to_onnx(model, image_tensor, output_path=save_path)
    else:
        print("No images found for creating example input!")
        exit(1)