# spynet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic 5-layer flow estimator
class BasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3)
        )

    def forward(self, x):
        return self.conv1(x)

class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic = nn.ModuleList([BasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def preprocess(self, img):
        return (img - self.mean) / self.std

    def forward(self, img1, img2):
        """
        img1, img2: [B, 3, H, W], float32 in [0,1], CUDA
        returns optical flow: [B, 2, H, W]
        """
        img1 = self.preprocess(img1)
        img2 = self.preprocess(img2)

        H, W = img1.shape[-2:]
        flow = torch.zeros(img1.size(0), 2, H, W, device=img1.device)

        # Multi-scale pyramid
        for level in range(5, -1, -1):
            scale = 2 ** level
            H_l, W_l = H // scale, W // scale
            
            img1_l = F.interpolate(img1, (H_l, W_l), mode="bilinear", align_corners=False)
            img2_l = F.interpolate(img2, (H_l, W_l), mode="bilinear", align_corners=False)

            flow = F.interpolate(flow, (H_l, W_l), mode="bilinear", align_corners=False) * 2.0

            warped = F.grid_sample(
                img2_l,
                torch.stack(torch.meshgrid(
                    torch.linspace(-1,1,W_l,device=flow.device),
                    torch.linspace(-1,1,H_l,device=flow.device),
                    indexing='xy'
                ), dim=-1).unsqueeze(0).expand(flow.shape[0],-1,-1,-1)
                + flow.permute(0,2,3,1),
                align_corners=False
            )

            flow = flow + self.basic[level](torch.cat([img1_l, warped, flow], dim=1))

        return flow
