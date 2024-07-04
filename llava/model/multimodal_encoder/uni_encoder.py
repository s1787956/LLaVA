import torch
import torch.nn as nn
from PIL import Image
import os

from types import SimpleNamespace
from torchvision.transforms import ToTensor, Compose, Resize

import torch
import torch.nn as nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, ViTModel


class UniVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None, **kwargs):
        # .forward_features circumvents classification heads and pooling, 
        # could also do with timm.create_model(..., global_pool="", num_classes=0)
        self.vision_tower = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained("owkin/phikon") 
        self.vision_tower = self.vision_tower.eval()
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images, return_cls=False): # TRUE returns cls token

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))#, output_hidden_states=True)
                image_feature = image_forward_outs.last_hidden_state[:, 0, :]
                image_feature.unsqueeze_(1) # add dim1: dim bsz, dim1, hidden
                image_features.append(image_feature)
        else:
            if return_cls:
                image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
                image_features = image_forward_out
                # unsqueeze dim 0 to get a bsz,1,1024 (because the LLava uses a hidden layer out that is bsz, dim1, hidden)
                image_features.unsqueeze_(0)
            else:
                image_forward_out = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))#, output_hidden_states=True)
                image_features = image_forward_out

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.first_layer.dtype)

    @property
    def dtype(self):
        return self.first_layer.dtype

    @property
    def device(self):
        return self.first_layer.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
    @property
    def first_layer(self):
        for layer in self.vision_tower.children():
            if list(layer.parameters(recurse=False)):
                return next(layer.parameters())
        return None
            