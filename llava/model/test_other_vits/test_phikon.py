import torch
import torch.nn as nn
from PIL import Image
import os

from transformers import AutoImageProcessor, ViTModel
from types import SimpleNamespace
from torchvision.transforms import ToTensor, Compose, Resize

import torch
import torch.nn as nn

from transformers import AutoImageProcessor, ViTModel


class PhikonVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        #self.select_layer = args.mm_vision_select_layer
        #self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        #else:
        #    self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = ViTModel.from_pretrained(self.vision_tower_name, add_pooling_layer=False)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # completely ignored
    # def feature_select(self, image_forward_outs):
    #     image_features = image_forward_outs.hidden_states[self.select_layer]
    #     if self.select_feature == 'patch':
    #         image_features = image_features[:, 1:]
    #     elif self.select_feature == 'cls_patch':
    #         image_features = image_features
    #     else:
    #         raise ValueError(f'Unexpected select feature: {self.select_feature}')
    #     return image_features

    @torch.no_grad()
    def forward(self, images, return_cls=False):

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))#, output_hidden_states=True)
                image_feature = image_forward_outs.last_hidden_state[:, 0, :]
                image_feature.unsqueeze_(1) # add dim1: dim bsz, dim1, hidden
                image_features.append(image_feature)
        else:
            image_forward_out = self.vision_tower(images.to(device=self.device, dtype=self.dtype))#, output_hidden_states=True)

            if return_cls:
                image_features = image_forward_out.last_hidden_state[:, 0, :]
                # unsqueeze dim 0 to get a bsz,1,768 (because the LLava uses a hidden layer out that is bsz, dim1, hidden)
                image_features.unsqueeze_(1)
            else:
                image_features = image_forward_out.last_hidden_state

        print(f"{image_features.shape=}")
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
    
    


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("owkin"):
        return PhikonVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


# Define the configuration
vision_tower_cfg = SimpleNamespace(
    mm_vision_select_layer=-2, # This will be ignored
    mm_vision_select_feature="patch", # This will be ignored
    mm_vision_tower="owkin/phikon"
)



vision_tower = build_vision_tower(vision_tower_cfg)
print(vision_tower.image_processor)
# print(sum(p.numel() for p in vision_tower.parameters()))

# x = torch.randn(16, 3, 224, 224)


# transform = Compose([
#     Resize((224, 224)),
#     ToTensor()
# ])


with torch.no_grad():
    outputs = vision_tower(x)


print(outputs.shape)

print("DONE!")
