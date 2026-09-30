from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from SPyNet import SPyNet

class GridCache:
    def __init__(self):
        self.grid = {}

    def get(self, H, W, device, dtype):
        key = (H, W, device, dtype)
        if key not in self.grid:
            gy, gx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij"
            )
            base_grid = torch.stack((gx, gy), dim=-1)  # [H, W, 2]
            self.grid[key] = base_grid
        return self.grid[key]

    
def warp_with_flow(x, flow, grid_cache):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # Reuse grid
    base_grid = grid_cache.get(H, W, device, dtype)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)
    
    if flow.shape[-2:] != (H, W):
        flow = torch.nn.functional.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
        flow = flow * (flow.shape[-1] / W)  # scale flow properly
        
    # Normalize flow to [-1,1]
    flow_x = flow[:, 0] * (2.0 / (W - 1))
    flow_y = flow[:, 1] * (2.0 / (H - 1))
    flow_norm = torch.stack([flow_x, flow_y], dim=-1)


    sampling_grid = base_grid + flow_norm

    return F.grid_sample(
        x,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )


class TemporalRefinery(nn.Module):
    def __init__(self, num_classes, hidden_dim=64):
        super().__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.grid_cache = GridCache()   # <-- HERE
        self.spynet = SPyNet().to(device)
        self.spynet.eval()

        for p in self.spynet.parameters():
            p.requires_grad = False
        
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 3, padding=1)
        )

    def forward(self, prev_img, curr_img, prev_logits, curr_logits):
    # def forward(self, prev_logits):
        # curr_logits=prev_logits
        flow = self.spynet(prev_img, curr_img)
        warped_logits = warp_with_flow(prev_logits, flow, self.grid_cache)
        
        x = torch.cat([warped_logits, curr_logits], dim=1)
        refined = self.refine(x)
        # return curr_logits + 0.05 * refined   # residual correction
        return curr_logits + 0.15 * refined   # residual correction

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = TemporalRefinery(num_classes=12)
    flops, params = get_model_complexity_info(model=model, input_res=( 12, 400,500), print_per_layer_stat=True)
    print(f"Temporal Refinery Params: {params}, Flops: {flops}")