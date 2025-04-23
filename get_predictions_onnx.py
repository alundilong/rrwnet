from pathlib import Path
import argparse

import onnxruntime as ort
import numpy as np
from skimage import io
import torch
from torchvision import utils as vutils

from preprocessing import enhance_image
from utils import pad_images_unet, to_torch_tensors

def load_onnx_session(onnx_path):
    """Load ONNX model and create inference session"""
    # Configure session options
    sess_options = ort.SessionOptions()
    
    # Set providers based on available hardware
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)

def predict_with_onnx(session, image_tensor):
    """Run inference using ONNX model"""
    # Convert to numpy if needed
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.cpu().numpy()
    else:
        image_np = image_tensor
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: image_np})
    prediction = outputs[-1]
    
    # Convert back to torch tensor for consistent post-processing
    prediction_tensor = torch.from_numpy(prediction)
    
    return prediction_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get predictions from an ONNX model')
    parser.add_argument('--onnx-model', type=str, required=True,
        help='Path to the ONNX model')
    parser.add_argument('--images-path', type=str, required=True,
        help='Path to the images')
    parser.add_argument('--masks-path', type=str, required=True,
        help='Path to the masks')
    parser.add_argument('--save-path', type=str, required=True,
        help='Path to save the predictions')
    parser.add_argument('--preprocess', action='store_true', 
        help='Preprocess the images')
    parser.add_argument('--refine', action='store_true', 
        help='Refine the predictions (must be supported by ONNX model)')
    args = parser.parse_args()

    # Load ONNX model
    print(f'Loading ONNX model from {args.onnx_model}')
    ort_session = load_onnx_session(args.onnx_model)

    print(f'Creating save path {args.save_path}')
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)

    print(f'Getting images and masks from {args.images_path} and {args.masks_path}')
    image_fns = sorted(Path(args.images_path).glob('*.png'))
    mask_fns = sorted(Path(args.masks_path).glob('*.png'))

    print('Processing images')
    for image_fn, mask_fn in zip(image_fns, mask_fns):
        print(f'  {image_fn.name}')
        assert Path(mask_fn).stem == Path(image_fn).stem
        
        # Load and preprocess image
        if args.preprocess:
            print('    Preprocessing first')
            img, mask = enhance_image(image_fn, mask_fn)
        else:
            img = (io.imread(image_fn) / 255.0)[..., :3]
            mask = io.imread(mask_fn) * 1.0
        
        # Pad images
        imgs, paddings = pad_images_unet([img, mask])
        img = imgs[0]
        padding = paddings[0]
        mask = imgs[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        
        # Convert to tensors
        tensors = to_torch_tensors([img, mask])
        image_tensor = tensors[0]
        mask_tensor = tensors[1]
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        
        # Run inference
        prediction = predict_with_onnx(ort_session, image_tensor)
        
        # Apply sigmoid if not refining
        if not args.refine:
            prediction = torch.sigmoid(prediction)
        
        # # Apply mask if provided
        if mask_tensor is not None:
            prediction[mask_tensor < 0.5] = 0

        # Remove padding
        prediction = prediction[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
        
        # Save result
        target_fn = save_path / Path(image_fn).name
        vutils.save_image(prediction, target_fn)

    print('Inference completed')