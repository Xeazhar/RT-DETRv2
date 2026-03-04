import os
import sys
import torch
import torch.nn as nn

# 1. SETUP ROOT PATHS
# Adjust this to your actual RT-DETRv2 folder path
ROOT = r'C:\Users\jazzb\rt-deterv2\RT-DETRv2'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.core import YAMLConfig

# ---------------------------
# SETTINGS
# ---------------------------
# Make sure these paths are correct for your machine
CONFIG_PATH = os.path.join(ROOT, 'configs', 'rtdetrv2', 'rtdetrv2_r101vd_6x_coco.yml')
CHECKPOINT_PATH = os.path.join(ROOT, 'output', 'rtdetrv2_r101vd_6x_coco', 'best.pth')
OUTPUT_FILE = "rtdetrv2_raw_640.onnx"

OPSET_VERSION = 16

# ---------------------------
# 2. LOAD CONFIG AND MODEL
# ---------------------------
print(f"Loading config from: {CONFIG_PATH}")
cfg = YAMLConfig(CONFIG_PATH, resume=None)

print(f"Loading weights from: {CHECKPOINT_PATH}")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# Handle EMA or standard weights
if 'ema' in checkpoint:
    state = checkpoint['ema']['module']
elif 'model' in checkpoint:
    state = checkpoint['model']
else:
    state = checkpoint

cfg.model.load_state_dict(state)

# ---------------------------
# 3. RAW MODEL WRAPPER
# ---------------------------
# We bypass the post-processor here because it often causes 
# coordinate scaling issues during ONNX exports.
class RawModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.deploy()

    def forward(self, images):
        # RT-DETR returns a dict in deploy mode
        outputs = self.model(images)
        # We only need the raw logits (scores) and boxes (coordinates)
        return outputs['pred_logits'], outputs['pred_boxes']

model_to_export = RawModelWrapper(cfg.model)
model_to_export.eval()

# ---------------------------
# 4. EXPORT ONNX
# ---------------------------
print("Starting export...")
dummy_input = torch.randn(1, 3, 640, 640)

# We use dynamic axes so you can use different batch sizes later
dynamic_axes = {
    'images': {0: 'batch'},
    'logits': {0: 'batch'},
    'boxes': {0: 'batch'}
}

torch.onnx.export(
    model_to_export,
    dummy_input,
    OUTPUT_FILE,
    input_names=['images'],
    output_names=['logits', 'boxes'],
    dynamic_axes=dynamic_axes,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    verbose=False
)

print(f"Successfully exported: {OUTPUT_FILE}")

# ---------------------------
# 5. VERIFY FILE
# ---------------------------
import onnx
onnx_model = onnx.load(OUTPUT_FILE)
onnx.checker.check_model(onnx_model)
print("ONNX integrity check passed!")