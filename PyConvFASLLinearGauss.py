# path: models/pyconv_lwanet_decoder_mhsa.py

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyconvresnet import (
    pyconvresnet18,
    pyconvresnet34,
    pyconvresnet50,
    pyconvresnet101,
    pyconvresnet152,
)


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


# -------------------------
# Option A: Linear Attention
# -------------------------
# Kernelized attention using phi(x) = elu(x) + 1 (stable, positive)
# Complexity: O(B * HW * D) vs O(B * (HW)^2 * D) for full MHSA.
class LinearSelfAttention(nn.Module):
    """
    Linearized multi-head attention over spatial tokens.
    Args:
      in_ch: input channels
      embed_dim: internal embedding dim per token (defaults to in_ch)
      num_heads: number of heads (embed_dim must be divisible by num_heads)
      proj_ratio: channel bottleneck ratio before attention (>=1). Embed_dim = max(16, in_ch//proj_ratio)
    """
    def __init__(self, in_ch: int, num_heads: int = 4, proj_ratio: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.num_heads = num_heads

        # embed dim chosen from proj_ratio and made divisible by num_heads
        e = max(16, in_ch // max(1, proj_ratio))
        e = (e // self.num_heads) * self.num_heads
        if e < self.num_heads:
            e = self.num_heads
        self.embed_dim = e
        self.head_dim = self.embed_dim // self.num_heads

        # projections (pointwise convs keep spatial structure)
        self.to_q = nn.Conv2d(in_ch, self.embed_dim, 1, bias=False)
        self.to_k = nn.Conv2d(in_ch, self.embed_dim, 1, bias=False)
        self.to_v = nn.Conv2d(in_ch, self.embed_dim, 1, bias=False)
        self.to_out = nn.Conv2d(self.embed_dim, in_ch, 1, bias=False)

        self.norm = nn.BatchNorm2d(in_ch)
        self.post_ln = nn.LayerNorm(self.embed_dim)  # applied over last dim of flattened tokens
        self.dropout = nn.Dropout(0.0)

    @staticmethod
    def _phi(x):
        # elu + 1 -> positive feature map, differentiable
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        # [B, E, H, W]
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # [B, E, N]
        q = q.flatten(2)
        k = k.flatten(2)
        v = v.flatten(2)

        # apply phi and reshape for heads -> [B, heads, head_dim, N]
        q = self._phi(q).view(B, self.num_heads, self.head_dim, N)
        k = self._phi(k).view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)

        # compute K * V^T in linearized manner:
        # context per head = (K @ V^T) -> shape [B, heads, head_dim, head_dim]
        # But more efficient: compute KV^T along tokens axis:
        # We compute S = K @ (V^T) across tokens: (head_dim x N) @ (N x head_dim) -> head_dim x head_dim
        # Then result_tokens = S^T @ q_tokens  (but we can compute Q^T (S) style)
        # A numerically stable & efficient implementation used below:
        # compute KV = k @ v.transpose(-1,-2)  -> [B, heads, head_dim, head_dim]
        KV = torch.einsum("bhdn,bhdm->bhnm", k, v)  # [B, heads, head_dim, head_dim]

        # now multiply: out_per_head_tokens = (KV^T @ q) along feature dim -> [B, heads, head_dim, N]
        # we want output per token: out[..., :, n] = KV @ q[..., :, n]
        out = torch.einsum("bhnm,bhdn->bhdn", KV, q)  # [B, heads, head_dim, N]

        # reshape back -> [B, E, N]
        out = out.reshape(B, self.embed_dim, N)

        # optional normalization (per token)
        out = self.post_ln(out.permute(0, 2, 1)).permute(0, 2, 1)  # LN over embed dim
        out = out.view(B, self.embed_dim, H, W)

        out = self.to_out(out)
        out = self.dropout(out)
        return self.norm(x + out)  # residual + BN


def gaussian_blur(x, kernel_size=3, sigma=1.0):
    channels = x.shape[1]
    grid = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
    gauss = torch.exp(-grid**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss[:, None] * gauss[None, :]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=channels)


def backbone_factory(name: str, pretrained: bool):
    name = name.lower()
    if name == "pyconvresnet18":
        return pyconvresnet18(pretrained=pretrained), "basic"
    if name == "pyconvresnet34":
        return pyconvresnet34(pretrained=pretrained), "basic"
    if name == "pyconvresnet50":
        return pyconvresnet50(pretrained=pretrained), "bottleneck"
    if name == "pyconvresnet101":
        return pyconvresnet101(pretrained=pretrained), "bottleneck"
    if name == "pyconvresnet152":
        return pyconvresnet152(pretrained=pretrained), "bottleneck"
    raise ValueError(f"Unsupported backbone_name: {name}")


def stage_channels(family: str) -> Tuple[int, int, int, int]:
    if family == "basic":
        return 64, 128, 256, 512
    elif family == "bottleneck":
        return 256, 512, 1024, 2048
    else:
        raise ValueError(f"Unknown family: {family}")


def build_final_head(in_ch: int, num_classes: int, target_head_ch: int = 64, max_compression_steps: int = 3) -> nn.Sequential:
    layers: List[nn.Module] = []
    current = in_ch
    steps = 0
    while current != target_head_ch and steps < max_compression_steps:
        next_ch = max(target_head_ch, current // 2)
        if next_ch == current:
            break
        layers.append(conv_dw(current, next_ch, 1))
        current = next_ch
        steps += 1
    if current != num_classes:
        layers.append(conv_dw(current, num_classes, 1))
    return nn.Sequential(*layers)


class PyConvFASLLinearGauss(nn.Module):
    """
    Decoder:
      - x1: MHSA (2 heads) + depthwise 1x1 conv
      - x2: MHSA (4 heads) + depthwise 1x1 conv
      - x3,x4: depthwise 3x3 conv only
      - All → proj → upsample → concat → final head
    """
    def __init__(
        self,
        num_classes: int = 11,
        backbone_name: str = "pyconvresnet50",
        pretrained: bool = True,
        target_head_ch: int = 64,
        max_compression_steps: int = 3,
        print_shapes: bool = False,
    ):
        super().__init__()
        backbone, family = backbone_factory(backbone_name, pretrained)
        C2, C3, C4, C5 = stage_channels(family)

        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer2 = backbone.layer1
        self.layer3 = backbone.layer2
        self.layer4 = backbone.layer3
        self.layer5 = backbone.layer4

        # MHSA
        self.linearselfattn1 = LinearSelfAttention(in_ch=C2, num_heads=2)
        self.linearselfattn2 = LinearSelfAttention(in_ch=C3, num_heads=4)

        # Depthwise convs
        self.proc1 = nn.Conv2d(C2, C2, 1, 1, 0, groups=C2, bias=False)
        self.proc2 = nn.Conv2d(C3, C3, 1, 1, 0, groups=C3, bias=False)
        self.proc3 = nn.Conv2d(C4, C4, 3, 1, 1, groups=C4, bias=False)
        self.proc4 = nn.Conv2d(C5, C5, 3, 1, 1, groups=C5, bias=False)

        # Project to same channel width
        stream_ch = C2
        self.proj1 = nn.Conv2d(C2, stream_ch, 1)
        self.proj2 = nn.Conv2d(C3, stream_ch, 1)
        self.proj3 = nn.Conv2d(C4, stream_ch, 1)
        self.proj4 = nn.Conv2d(C5, stream_ch, 1)

        # Final head
        self.final = build_final_head(
            in_ch=stream_ch * 4,
            num_classes=num_classes,
            target_head_ch=target_head_ch,
            max_compression_steps=max_compression_steps,
        )

        self.print_shapes = print_shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x4 = self.layer5(x3)

        if self.print_shapes:
            print(f"x1:{x1.shape}, x2:{x2.shape}, x3:{x3.shape}, x4:{x4.shape}")

        # Apply MHSA to x1 and x2
        x1 = self.linearselfattn1(x1)
        x2 = self.linearselfattn2(x2)

        # Depthwise convs
        s1 = self.proj1(self.proc1(x1))
        s2 = self.proj2(self.proc2(x2))
        s3 = self.proj3(self.proc3(x3))
        s4 = self.proj4(self.proc4(x4))

        # Upsample all to x1 spatial size
        target_size = x1.shape[2:]
        s2 = F.interpolate(s2, size=target_size, mode="bilinear", align_corners=False)
        s3 = F.interpolate(s3, size=target_size, mode="bilinear", align_corners=False)
        s4 = F.interpolate(s4, size=target_size, mode="bilinear", align_corners=False)

        x_cat = torch.cat([s1, s2, s3, s4], dim=1)

        out = self.final(x_cat)
        
        out = gaussian_blur(out, sigma=1.0)
        
        return F.log_softmax(out, dim=1)



    
if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = PyConvFASLLinearGauss(backbone_name="pyconvresnet50", pretrained=True)
    flops, params = get_model_complexity_info(model=model, input_res=( 3, 400,500), print_per_layer_stat=True)
    print(f"PyConv50_LWANet_FASL_MHSA: Params: {params}, Flops: {flops}")