# IMPORTANT: YOU ARE NOT SUPPOSED TO TOUCH THIS FILE. ONLY EXTEND IT, DO NOT EDIT FUNCTIONALITY.


""" Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-------------------------------------------------------------------------------

Cloned and trimmed version of timm.models.vision_transformer.py
Here for STABLE reference.

Check out https://github.com/pprp/timm/blob/master/timm/models/vision_transformer.py for the original file.

The following is the original docstring of the file.

-------------------------------------------------------------------------------

Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.layers import PatchEmbed, Mlp as TimmMlp, DropPath, trunc_normal_, lecun_normal_, resample_patch_embed, \
    resample_abs_pos_embed
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply
from timm.models.vision_transformer import _load_weights

from backbone.utils.layers import IncrementalClassifier
from backbone import MammothBackbone, register_backbone
from backbone.utils.lora_utils import LoRAAttention, LoRAMlp
from utils.conf import warn_once

__all__ = ['VisionTransformer']  # model_registry will add each entrypoint fn to this


class Mlp(TimmMlp):
    def forward(self, x, **kwargs):
        return super().forward(x)


class Attention(nn.Module):
    """
    Attention layer as used in Vision Transformer.

    Args:
        dim: Number of input channels
        num_heads: Number of attention heads
        qkv_bias: If True, add a learnable bias to q, k, v
        attn_drop: Dropout rate for attention weights
        proj_drop: Dropout rate after the final projection
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention_scores=False, **kwargs):
        """
        Forward pass of the attention layer.

        Args:
            x: Input tensor
            return_attention_scores: If True, return attention probabilities along with output
        Returns:
            If return_attention_scores is False: output tensor
            If return_attention_scores is True: tuple (output tensor, attention probabilities)
        """
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn_probs = None

        # If attention scores are needed, use manual calculation for reliable score extraction
        if return_attention_scores:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn_probs = attn.softmax(dim=-1)
            # Keep raw attention probabilities for visualization
            # Apply dropout only for the computation path
            dropped_attn = self.attn_drop(attn_probs)
            x = (dropped_attn @ v)
        elif torch.__version__ >= '2.1.0':
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=self.attn_drop.p)
        else:
            warn_once("Torch version < 2.1.0 detected. Using the original attention code.")
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention_scores:
            return x, attn_probs
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            attn_layer=Attention,
            mlp_layer=Mlp
    ):
        super().__init__()
        self.embed_dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, return_attention_scores=False, **kwargs):
        # Process normalized input through attention
        attn_output = self.attn(self.norm1(x), return_attention_scores=return_attention_scores, **kwargs)

        if return_attention_scores:
            current_x, attn_probs = attn_output
        else:
            current_x = attn_output
            attn_probs = None

        # Apply residual connection, layer scale, and drop path
        x = x + self.drop_path1(self.ls1(current_x))

        # MLP path (no attention scores involved)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), **kwargs)))

        if return_attention_scores:
            return x, attn_probs
        return x


class VisionTransformer(MammothBackbone):
    """ Vision Transformer.
    This implementation supports LoRA (Layer-wise Relevance Adaptation) parameters if `use_lora=True`.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=Block,
            attn_layer=None,
            mlp_layer=None,
            use_lora=False,
            args=None
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
            block_fn: (nn.Module): transformer block
            attn_layer: (nn.Module): attention layer
            args: (Namespace): optional command-line arguments
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'

        self.attn_pool = None
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.act_layer = act_layer or nn.GELU

        attn_layer = attn_layer if attn_layer is not None else (Attention if not use_lora else LoRAAttention)
        mlp_layer = mlp_layer if mlp_layer is not None else (Mlp if not use_lora else LoRAMlp)
        self.attn_layer = attn_layer
        self.norm_layer = norm_layer
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.class_token = class_token
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_dim = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.mlp_ratio = mlp_ratio
        self.args = args
        self.init_values = init_values
        self.qkv_bias = qkv_bias
        self.attn_drop_rate = attn_drop_rate
        self.depth = depth
        self.drop_rate = drop_rate
        self.mlp_layer = mlp_layer
        self.drop_path_rate = drop_path_rate

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=norm_layer,
                act_layer=self.act_layer,
                attn_layer=attn_layer,
                mlp_layer=mlp_layer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = IncrementalClassifier(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

        self.embed_dim = embed_dim

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor, AB={}, return_all=False, return_attention_scores=False):
        """
        Compute the forward pass of ViT (features only).
        Can take in an additional argument `AB`, which is a dictionary containing LoRA-style parameters for each block.

        Args:
            x: input tensor
            AB: dictionary containing LoRA-style parameters for each block
            return_all: whether to return all intermediate features
            return_attention_scores: whether to return attention probabilities from each block

        Returns:
            If return_attention_scores is False:
                if return_all: list of intermediate features
                else: final features
            If return_attention_scores is True:
                if return_all: tuple (list of intermediate features, list of attention maps)
                else: tuple (final features, list of attention maps)
        """
        # DEBUG: Track timing for ViT forward pass
        import time
        import logging
        logger = logging.getLogger(__name__)

        start_time = time.time()
        logger.debug(f"🚀 ViT forward_features started - batch_size: {x.shape[0]}, return_attention: {return_attention_scores}")

        int_features = []
        attn_maps = []

        # Patch embedding
        patch_start = time.time()
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        patch_time = time.time() - patch_start
        logger.debug(f"   ✓ Patch embedding completed in {patch_time:.3f}s")

        # Process through transformer blocks
        blocks_start = time.time()
        for idx, blk in enumerate(self.blocks):
            block_start = time.time()

            AB_blk = AB.get(idx)
            block_kwargs = {'AB': AB_blk} if AB_blk is not None else {}

            if return_attention_scores:
                # PERFORMANCE WARNING: Attention extraction is expensive
                if idx == 0:  # Only log once
                    logger.debug(f"   ⚠️  ATTENTION EXTRACTION ENABLED - This significantly slows training!")

                out, attn_probs = blk(x, return_attention_scores=True, **block_kwargs)
                attn_maps.append(attn_probs)
                x = out
            else:
                x = blk(x, **block_kwargs)

            if return_all:
                int_features.append(x.clone())

            block_time = time.time() - block_start
            if block_time > 0.1:  # Log slow blocks
                logger.debug(f"   Block {idx}: {block_time:.3f}s (slow)")
            elif idx % 4 == 0:  # Log every 4th block
                logger.debug(f"   Block {idx}: {block_time:.3f}s")

        blocks_time = time.time() - blocks_start
        logger.debug(f"   ✓ All {len(self.blocks)} blocks completed in {blocks_time:.3f}s")

        # Final normalization
        norm_start = time.time()
        x = self.norm(x)
        norm_time = time.time() - norm_start

        if return_all:
            int_features.append(x.clone())

        total_time = time.time() - start_time

        # PERFORMANCE MONITORING: Log timing warnings
        if total_time > 1.0:
            logger.warning(f"🐌 SLOW ViT forward pass: {total_time:.3f}s (batch_size: {x.shape[0]})")
            if return_attention_scores:
                logger.warning(f"   Attention extraction likely causing slowdown")
        elif total_time > 0.5:
            logger.debug(f"⚠️  Moderate ViT timing: {total_time:.3f}s")
        else:
            logger.debug(f"✓ ViT forward_features completed in {total_time:.3f}s")

        if return_attention_scores:
            if return_all:
                return int_features, attn_maps
            return x, attn_maps

        if return_all:
            return int_features
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        """
        Compute the forward pass of ViT (head only).
        Expects input of shape [batch_size, num_patches, embed_dim].

        Args:
            x: input tensor
            pre_logits: whether to return the pre-logits (pooled features) or the final class scores

        Returns:
            output tensor with shape [batch_size, num_classes] if `pre_logits` is False, else [batch_size, embed_dim]
        """
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor, AB: dict = {}, returnt='out', return_attention_scores=False):
        """
        Compute the forward pass of ViT.
        Can take in an additional argument `AB`, which is a dictionary containing LoRA-style parameters for each block.

        `AB` can contain
        - a single value for each block (e.g. `AB = {0: {"qkv": torch.Tensor(...)}, 1: {"qkv": torch.Tensor(...)}, ...}`)
        - a dictionary for each block with a single key `B` (e.g. `AB = {0: {"qkv": {"B": torch.Tensor(...)}}}`)
        - a dictionary for each block with both `A` and `B` keys of LoRA parameters (e.g. `AB = {0: {"qkv": {"A": torch.Tensor(...), "B": torch.Tensor(...)}}}`)

        Supported keys for each block are `qkv`, `proj`, `fc1`, `fc2`.

        NOTE: The values of `AB` are **summed** with the weights of the corresponding block.

        Args:
            x: input tensor
            AB: dictionary containing LoRA-style parameters for each block
            returnt: return type (a string among `out`, `features`, `both`, or `full`)
            return_attention_scores: whether to return attention probabilities from each block

        Returns:
            Depending on returnt and return_attention_scores:
            - returnt='out':
                if return_attention_scores=False: output tensor
                if return_attention_scores=True: tuple (output tensor, list of attention maps)
            - returnt='features':
                if return_attention_scores=False: features tensor
                if return_attention_scores=True: tuple (features tensor, list of attention maps)
            - returnt='both':
                if return_attention_scores=False: tuple (output tensor, features tensor)
                if return_attention_scores=True: tuple (output tensor, features tensor, list of attention maps)
            - returnt='full':
                if return_attention_scores=False: tuple (output tensor, list of all intermediate features)
                if return_attention_scores=True: tuple (output tensor, list of all intermediate features, list of attention maps)
        """
        assert returnt in ('out', 'features', 'both', 'full')

        features_output = self.forward_features(x, AB,
                                             return_all=returnt == 'full',
                                             return_attention_scores=return_attention_scores)

        # Unpack features and attention maps based on what was returned
        if return_attention_scores:
            if returnt == 'full':
                all_features, attn_maps = features_output
                x = all_features[-1]
            else:
                x, attn_maps = features_output
        else:
            if returnt == 'full':
                all_features = features_output
                x = all_features[-1]
            else:
                x = features_output

        feats = self.forward_head(x, pre_logits=True)

        if returnt == 'features':
            return (feats, attn_maps) if return_attention_scores else feats

        out = self.head(feats)

        if returnt == 'both':
            return (out, feats, attn_maps) if return_attention_scores else (out, feats)
        elif returnt == 'full':
            return (out, all_features, attn_maps) if return_attention_scores else (out, all_features)

        return (out, attn_maps) if return_attention_scores else out

    def get_params(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.

        Returns:
            parameters tensor
        """
        params = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'head' in kk:
                params.append(pp.view(-1))
        return torch.cat(params)

    def get_grads(self, discard_classifier=False) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.

        Returns:
            gradients tensor
        """
        grads = []
        for kk, pp in list(self.named_parameters()):
            if not discard_classifier or not 'head' in kk:
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb,
        posemb_new,
        num_prefix_tokens=1,
        gs_new=(),
        interpolation='bicubic',
        antialias=False,
):
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    logging.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def _convert_openai_clip(state_dict, model):
    out_dict = {}
    swaps = [
        ('visual.', ''), ('conv1', 'patch_embed.proj'), ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'), ('ln_pre', 'norm_pre'), ('ln_post', 'norm'), ('ln_', 'norm'),
        ('in_proj_', 'qkv.'), ('out_proj', 'proj'), ('mlp.c_fc', 'mlp.fc1'), ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith('visual.'):
            continue
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict,
        model,
        adapt_layer_scale=False,
        interpolation='bicubic',
        antialias=True,
):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def create_vision_transformer(variant, base_class=VisionTransformer, pretrained=False, filter_fn=checkpoint_filter_fn, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = filter_fn

    if variant == 'vit_base_patch16_224_in21k_fn_in1k_old':
        from timm.models import resolve_pretrained_cfg
        from backbone.utils.vit_default_cfg import default_cfgs

        pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=default_cfgs[variant].default)
        pretrained_cfg.custom_load = True

        return build_model_with_cfg(
            base_class,
            variant,
            pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_filter_fn=_filter_fn,
            pretrained_strict=True,
            **kwargs,
        )
    else:
        return build_model_with_cfg(
            base_class, variant, pretrained,
            pretrained_filter_fn=_filter_fn,
            **kwargs,
        )


def vit_base_patch16_224_prompt_prototype(pretrained=False, pretrain_type='in21k-ft-in1k', **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).

    By default, returns a model pre-trained on ImageNet-21k.
    Supports:
    - Pre-train on ImageNet-21k (pretrain_type='in21k')
    - Pre-train on ImageNet-21k and finetuned on ImageNet-1k (pretrain_type='in21k_old')
    - Pre-train with MoCoV3 on ImageNet-21k (pretrain_type='in21k-ft-in1k')

    Args:
        pretrained (bool): Load pre-trained weights.
        pretrain_type (str): Type of pre-training. Default is 'in21k'. Other options are 'in21k_old' and 'in1k'.
        **kwargs: Additional arguments to pass to the model.
    """
    assert pretrain_type in ['in21k', 'in21k_old', 'in21k-ft-in1k'], f"Invalid pretrain_type: {pretrain_type}"
    if not pretrained:
        logging.warning("creating a ViT without pre-trained weights. This is not recommended.")

    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    if kwargs is None:
        kwargs = {}

    if pretrain_type == 'in21k_old':
        model = create_vision_transformer('vit_base_patch16_224_in21k_fn_in1k_old', pretrained=pretrained, **dict(model_kwargs, **kwargs))
    elif pretrain_type == 'in21k':
        model = create_vision_transformer('vit_base_patch16_224.augreg_in21k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
    else:
        model = create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_kwargs, **kwargs))
    return model


@register_backbone("vit")
def vit_backbone(num_classes, pretrained=True, pretrain_type='in21k-ft-in1k'):
    return vit_base_patch16_224_prompt_prototype(pretrained=pretrained, pretrain_type=pretrain_type, num_classes=num_classes)

# your previous implementation, WRONG
# """Vision Transformer (ViT) implementation with attention visualization support."""

# import logging
# from typing import Dict, List, Optional, Tuple, Union, cast

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from backbone import MammothBackbone, register_backbone
# from backbone.utils.layers import IncrementalClassifier

# def trunc_normal_(tensor: torch.Tensor, mean: float = 0., std: float = 1.) -> None:
#     """Truncated normal initialization."""
#     torch.nn.init.trunc_normal_(tensor, mean=mean, std=std)

# class PatchEmbed(nn.Module):
#     """Image to Patch Embedding."""
#     def __init__(self, img_size: int = 224, patch_size: int = 16,
#                  in_chans: int = 3, embed_dim: int = 768):
#         super().__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size // patch_size, img_size // patch_size)
#         self.num_patches = self.grid_size[0] * self.grid_size[1]

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         return x

# class Attention(nn.Module):
#     """Multi-head self attention with attention score extraction."""
#     def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
#                  attn_drop: float = 0., proj_drop: float = 0.):
#         super().__init__()
#         self.num_heads = num_heads
#         self.scale = (dim // num_heads) ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
#         qkv = qkv.permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn_scores = attn.detach()  # Save before dropout
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         if return_attention:
#             return x, attn_scores
#         return x

# class Block(nn.Module):
#     """Transformer block with attention visualization support."""
#     def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.,
#                  qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0.):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                             attn_drop=attn_drop, proj_drop=drop)
#         self.norm2 = nn.LayerNorm(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, mlp_hidden_dim),
#             nn.GELU(),
#             nn.Dropout(drop),
#             nn.Linear(mlp_hidden_dim, dim),
#             nn.Dropout(drop)
#         )

#     def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         if return_attention:
#             attended, attn_scores = self.attn(self.norm1(x), return_attention=True)
#             x = x + attended
#             x = x + self.mlp(self.norm2(x))
#             return x, attn_scores

#         x = x + self.attn(self.norm1(x))
#         x = x + self.mlp(self.norm2(x))
#         return x

# class VisionTransformer(MammothBackbone):
#     """Vision Transformer with attention visualization support."""

#     def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
#                  num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
#                  num_heads: int = 12, mlp_ratio: float = 4., qkv_bias: bool = True,
#                  drop_rate: float = 0., attn_drop_rate: float = 0., **kwargs):
#         super().__init__()
#         self.num_classes = num_classes
#         self.embed_dim = embed_dim
#         self.num_features = embed_dim

#         # Initialize attention cache
#         self._cached_attention_maps: Optional[List[torch.Tensor]] = None

#         self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
#                                     in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         self.blocks = nn.ModuleList([
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
#                   qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(embed_dim)
#         self.head = IncrementalClassifier(embed_dim, num_classes)

#         # Initialize weights
#         trunc_normal_(self.pos_embed, std=.02)
#         trunc_normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)

#     def _init_weights(self, m: nn.Module) -> None:
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.zeros_(m.bias)
#             nn.init.ones_(m.weight)

#     def forward_features(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
#         x = self.patch_embed(x)
#         batch_size = x.shape[0]
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)

#         if not return_attention:
#             for blk in self.blocks:
#                 x = blk(x)
#             x = self.norm(x)
#             return x

#         attention_maps: List[torch.Tensor] = []
#         for blk in self.blocks:
#             x, attn = cast(Tuple[torch.Tensor, torch.Tensor], blk(x, return_attention=True))
#             attention_maps.append(attn)
#         x = self.norm(x)
#         return x, attention_maps

#     def forward(self, x: torch.Tensor, returnt: str = 'out') -> torch.Tensor:
#         """Forward pass with support for different return types.

#         Args:
#             x: Input tensor
#             returnt: Return type ('out', 'features', or 'attention')

#         Returns:
#             torch.Tensor: Depending on returnt:
#                 - 'out': class logits
#                 - 'features': CLS token features
#                 - 'attention': class logits (attention maps accessible via get_attention_maps)
#         """
#         output = self.forward_features(x, return_attention=(returnt == 'attention'))

#         if returnt == 'attention':
#             features, attention_maps = cast(Tuple[torch.Tensor, List[torch.Tensor]], output)
#             self._cached_attention_maps = attention_maps
#         else:
#             features = cast(torch.Tensor, output)
#             self._cached_attention_maps = None

#         # Get CLS token features
#         cls_features = features[:, 0]

#         if returnt == 'features':
#             return cls_features

#         return self.head(cls_features)

#     def get_attention_maps(self) -> Optional[List[torch.Tensor]]:
#         """Get the last computed attention maps.

#         Returns:
#             List of attention tensors, one per transformer block,
#             or None if no attention maps were computed.
#         """
#         return self._cached_attention_maps

#     def get_params(self) -> torch.Tensor:
#         """Returns all parameters concatenated in a single tensor."""
#         params = []
#         for pp in list(self.parameters()):
#             params.append(pp.view(-1))
#         return torch.cat(params)

# @register_backbone("vit")
# def vit_backbone(num_classes: int, **kwargs) -> VisionTransformer:
#     """Creates a Vision Transformer model.

#     Args:
#         num_classes: Number of output classes
#         **kwargs: Additional arguments for model configuration
#     """
#     # Filter out None values and 'kwargs' key if present
#     filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None and k != 'kwargs'}

#     # Create model with filtered arguments
#     model = VisionTransformer(num_classes=num_classes, **filtered_kwargs)

#     logging.warning("creating a ViT without pre-trained weights. This is not recommended.")
#     return model
