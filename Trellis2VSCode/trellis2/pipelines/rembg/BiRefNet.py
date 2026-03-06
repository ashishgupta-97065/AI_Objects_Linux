from typing import *
from transformers import AutoModelForImageSegmentation, PreTrainedModel
import torch
from torchvision import transforms
from PIL import Image


def _patch_birefnet_post_init():
    """
    BiRefNet (briaai/RMBG-2.0) doesn't call post_init() in its __init__,
    so transformers 5.x never sets all_tied_weights_keys. Patch
    mark_tied_weights_as_initialized to handle this gracefully.
    """
    _orig = PreTrainedModel.mark_tied_weights_as_initialized
    def _patched(self, loading_info):
        if not hasattr(self, 'all_tied_weights_keys'):
            self.all_tied_weights_keys = {}
        return _orig(self, loading_info)
    PreTrainedModel.mark_tied_weights_as_initialized = _patched

_patch_birefnet_post_init()


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    