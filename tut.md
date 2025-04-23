### run prediction
```bash
python get_predictions.py --weights shared_data\weights\rrwnet_HRF_0.pth --images-path d
ata\images --masks-path data\masks --save-path predictions_HRF_0 --preprocess

python get_predictions.py --weights shared_data\weights\rrwnet_RITE_1.pth --images-path data\images --masks-path data\masks --save-path predictions_RITE_1 --preprocess

python get_predictions.py --weights shared_data\weights\rrwnet_RITE_refinement.pth --ima
ges-path data\images --masks-path data\masks --save-path predictions_RITE_refinement --preprocess
```

### save model to onnx
```bash
python torch2onnx.py --weights shared_data\weights\rrwnet_HRF_0.pth --images-path data\images --masks-path data\masks --save-path tmp
```

### pred with onnx
```bash
python get_predictions_onnx.py --onnx-model tmp\models\vessel_seg.onnx --images-path data\images --masks-path data\masks --save-path predictions_onnx --preprocess
```