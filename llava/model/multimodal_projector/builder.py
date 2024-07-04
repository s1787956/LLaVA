"""
Based on: https://github.com/lucidrains/flamingo-pytorch
"""

import torch
import re

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum, nn


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        num_latents=128, # adjust
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()

        heads = dim // dim_head
        assert dim % dim_head == 0, "Number of heads must match ..."

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, v, d = x.shape[:3]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs, "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        # x = rearrange(
        #     x, "b T F v d -> b T (F v) d"
        # )  # flatten the frame and spatial dimensions
        #if exists(self.media_time_embs):
        #    x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b n d", b=b)

        for attn, ff in self.layers:         
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        out = self.norm(latents)
        return out


def build_vision_projector(config, mm_hidden_size=512, initial_pooling=False, delay_load=False, **kwargs):

    # config.mm_hidden_size = 2560 ## TODO: ProteinChat REMOVE THIS TERRIBLE LINE OF CODE 
    # config.mm_hidden_size = 1536 ## TODO: PathoChat REMOVE THIS TERRIBLE LINE OF CODE 
    # config.mm_hidden_size = 1024 ## TODO: PathoChat REMOVE THIS TERRIBLE LINE OF CODE 
    #config.mm_hidden_size = 512 ## TODO: PathoChat REMOVE THIS TERRIBLE LINE OF CODE 
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'xattention':
        print("Projector = Xattention.")
        print("config.hidden_size: ", config.hidden_size)
        print("mm_hidden_size: ", config.mm_hidden_size)
        return nn.Sequential(
            #nn.AdaptiveAvgPool2d((1024, config.hidden_size)) if initial_pooling else nn.Identity(),
            PerceiverResampler(dim=config.mm_hidden_size),
            nn.Linear(config.mm_hidden_size,config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    if projector_type == 'xattentionpool':
        print("Projector = Xattention with Adaptive Pooling.")
        return nn.Sequential(
            PerceiverResampler(dim=config.mm_hidden_size),
            nn.AdaptiveAvgPool2d((1, config.hidden_size)),
        )

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
