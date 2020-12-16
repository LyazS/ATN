import torch
import numpy as np
import segmentation_models_pytorch as smp
device=torch.device("cuda")
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b7",
    encoder_weights=None,
    encoder_depth=5,
    in_channels=3,
    classes=2,
).to(device)
model.eval()
print(model)
a = np.random.randn(1, 3, 256, 256).astype(np.float32)
a = torch.from_numpy(a).to(device)
b = model(a)
print(b.shape)