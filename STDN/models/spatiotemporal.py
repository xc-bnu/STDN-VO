import os
import torch
import torchvision
import torch.nn as nn
import timm
from torchvision.utils import save_image
from torchvision import transforms
from functools import partial
from tqdm import tqdm
import math
import warnings
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from STDN.models.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from STDN.models.vit_utils import DropPath, to_2tuple, trunc_normal_
from STDN.models.corr import CorrBlock

from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

def _cfg(url='', **kwargs): 
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module): 
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=420, input_dim=96): 
        super(ConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class Block(nn.Module):

    def __init__(self, dim, num_heads, frames=2, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        self.gru = ConvGRU(hidden_dim=frames, input_dim=frames)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            # Spatial
            xs = x + self.drop_path(self.attn(self.norm1(x)))
            atten_output_print = xs

            # Mlp
            x = xs
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            gru_atten_print = x
            return x, atten_output_print, gru_atten_print 

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=96, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class a_head_module1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LeakyReLU(negative_slope=5e-2),
            nn.Linear(512, 6)
        )
    
    def forward(self, input):
        outs = self.fc(input)
        return outs[:, :3], outs[:, 3:]

class VisionTransformer_gru(nn.Module):

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=96, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=(256, 256), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, frames=num_frames, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        '''
        =========== pretrained model feature extract ==========
        '''
        self.stmodel = timm.create_model('swinv2_tiny_window8_256', pretrained=True, features_only=True, pretrained_cfg_overlay=dict(file="./pretrained_models/swinv2_tiny_window8_256.bin"))
        self.stmodel = self.stmodel.cuda()
        self.stmodel = self.stmodel.eval()

        # multi-task module
        self.time_feature = nn.ModuleList([
            Time_Feature(dim1=48, dim2=420, dim3=192, scale=4, depth=self.depth),
            Time_Feature(dim1=120, dim2=420, dim3=220, scale=8, depth=self.depth),
            Time_Feature(dim1=240, dim2=420, dim3=280, scale=16, depth=self.depth),
            Time_Feature(dim1=480, dim2=420, dim3=400, scale=32, depth=self.depth)
        ])

        self.cnn_ = nn.ModuleList([
            CNN(dim1=192, dim2=48),
            CNN(dim1=220, dim2=96),
            CNN(dim1=280, dim2=192),
            CNN(dim1=400, dim2=192)
        ])
        self.cnn_out = nn.Sequential(
            nn.Conv2d(192, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.cnn_out_1 = nn.Sequential(
            nn.Conv2d(48, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(1024, 768),
            # nn.Dropout(p=0.5)
            nn.LeakyReLU(negative_slope=5e-2)
        )

        self.a_head1 = a_head_module1()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head
    
    def cosine_similarity(self, x, y, norm=False):
        xy = x.dot(y)
        x2y2 = np.linalg.norm(x, ord=2) * np.linalg.norm(x, ord=2)
        sim = xy / x2y2
        return sim

    def forward_features(self, x, i_ana):
        B = x.shape[0] 
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed 

        x = self.pos_drop(x)

        ## Attention blocks
        if i_ana % 500 == 0:
            j_ana = 0
        for blk in self.blocks:
            x, atten_output_print, gru_atten_print = blk(x, B, T, W)
            if i_ana % 500 == 0:
                j_ana += 1

        x = self.norm(x)
        x = x[:, 0]
        x = rearrange(x, '(b t) m -> b t m',b=B,t=T)
        return x
    

    def forward(self, x):
        i_ana = 0
        B = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        image = rearrange(x, '(b t) c h w -> b t c h w', b=B)
        image1 = image[:, 0, :, :, :]
        image2 = image[:, 1, :, :, :]

        features = self.stmodel(x)
        
        X_f = features[0]
        xt = self.time_feature[0](X_f, image1, i_ana)
        F = xt
        F = self.cnn_[0](F) 
        x = self.cnn_out_1(F)
        x = x.view(B, -1)
        x_gru = self.fc_1(x)

        X_f = rearrange(X_f, '(b t) h w c -> b c t h w', b=B)
        x_transformer = self.forward_features(X_f, i_ana)
        x_transformer = x_transformer.contiguous().view(B, -1)

        out = torch.cat((x_gru, x_transformer), dim=1)

        rot_results, tra_results = self.a_head1(out)

        
        return tra_results, rot_results

class CNN(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.cnn = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out
        
class Time_Feature(nn.Module):
    def __init__(self, dim1, dim2, dim3, scale, depth=12):
        super().__init__()

        self.dim1 = dim1 
        self.depth = depth 
        self.scale = scale 
        self.gru = ConvGRU(hidden_dim=324+self.dim1, input_dim=dim1)
        self.cnn = nn.Sequential(
            nn.Conv2d(324+self.dim1, dim3, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, stride=2)
        )

    def coords_grid(self, batch, ht, wd, device):
        coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
        coords = torch.stack(coords[::-1], dim=0).float()
        return coords[None].repeat(batch, 1, 1, 1)

    def initialize_coords(self, img):
        N, C, H, W = img.shape
        coords = self.coords_grid(N, H//self.scale, W//self.scale, device=img.device)
        return coords

    def forward(self, x, image1, i_ana):
            B = image1.shape[0]
            feature = rearrange(x, '(b t) h w c-> b t h w c', b=B) 
            fmap1 = rearrange(feature[:, 0, :, :], 'b h w c-> b c h w')
            fmap2 = rearrange(feature[:, 1, :, :], 'b h w c-> b c h w')
            net, inp = torch.split(fmap1, [self.dim1, self.dim1], dim=1)
            net = torch.tanh(net) 
            inp = torch.relu(inp)
            corr_fn = CorrBlock(fmap1, fmap2, radius=4) 
            coords = self.initialize_coords(image1) 
            coords = coords.detach()
            corr = corr_fn(coords)
            net = torch.cat([corr, net], dim=1)
            for i in range(self.depth):
                net = self.gru(net, inp) 
            output = self.cnn(net)

            return output


