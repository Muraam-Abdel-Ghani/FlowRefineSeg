import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from dataset import EndoVisDataset
from metrics.dice_score import dice_score
from metrics.miou_score import miou_score
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
import cv2

from PyConvFASLLinearGauss import PyConvFASLLinearGauss
from RefineryModuleWithFlow import TemporalRefinery
from temporal_dataset import TemporalRefinementDataset

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


import shutil
import os

def remove_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f'Removed: {file_path}')
        except Exception as e:
            print(f'Error removing {file_path}: {e}')


def warp_with_flow(x, flow, grid_cache):
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # Reuse grid
    base_grid = grid_cache.get(H, W, device, dtype)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Normalize flow to [-1,1]
    flow_x = flow[:, 0] * (2.0 / (W - 1))
    flow_y = flow[:, 1] * (2.0 / (H - 1))
    flow_norm = torch.stack([flow_x, flow_y], dim=-1)

    sampling_grid = base_grid + flow_norm

    return torch.nn.functional.grid_sample(
        x,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )


def temporal_consistency_loss_fast(curr_logits: torch.Tensor, prev_logits_warped: torch.Tensor, use_logit_l1: bool = True) -> torch.Tensor:
    """
    Fast temporal loss:
    - Option A (default): L1 on logits (cheap, no softmax)
    - Option B: small-channel-wise sigmoid/softmax approximation if needed (slower)
    """
    # curr_logits, prev_logits_warped: [B, C, H, W]
    if use_logit_l1:
        # Option: normalize by channel-wise max to reduce scale sensitivity
        diff = curr_logits - prev_logits_warped
        return diff.abs().mean()
    else:
        # fallback (slower): L1 on softmax probabilities
        curr_prob = F.softmax(curr_logits, dim=1)
        prev_prob = F.softmax(prev_logits_warped, dim=1)
        return torch.mean(torch.abs(curr_prob - prev_prob))

class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, smooth=1, tversky_alpha=0.7, tversky_beta=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.tversky_loss = TverskyLoss(smooth=smooth, alpha=tversky_alpha, beta=tversky_beta)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # print(f'in combined: {outputs.shape}, {targets.shape}')
        tversky_loss = self.tversky_loss(outputs, targets)
        # print(f'combined: tversky loss: {tversky_loss.shape}')
        ce_loss = self.cross_entropy_loss(outputs, targets)
        # return ce_loss
        return self.alpha * tversky_loss + (1 - self.alpha) * ce_loss

# Define Tversky Loss
class TverskyLoss(torch.nn.Module):
    def __init__(self, smooth=1, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):

        #In LWANet, output is onehot 128 by 128 size. need to interpolate to match 
        # outputs_up = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)

        num_classes = outputs.size(1)
        targets_one_hot = torch.eye(num_classes, device=targets.device)[targets].permute(0, 3, 1, 2)

        outputs_flat = outputs.reshape(-1)
        targets_flat = targets_one_hot.reshape(-1)
        
        true_pos = torch.sum(outputs_flat * targets_flat)
        false_neg = torch.sum(targets_flat * (1 - outputs_flat))
        false_pos = torch.sum((1 - targets_flat) * outputs_flat)
        
        # print(f'true_pos: {true_pos.shape}, false_neg: {false_neg.shape}, false_pos: {false_pos.shape}')
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)

        
        return 1 - tversky_index
    
def compute_iou_per_class(pred, labels, num_classes=12):
    """
    Computes IoU for each class and returns:
      - iou_list: list of IoU for each class [class0, class1, ...]
      - mean_iou: average IoU across all classes
    Args:
        outputs: (B, C, H, W) raw logits
        labels:  (B, H, W) integer class indices
    """
    # Convert to predicted classes
    # pred = outputs.argmax(dim=1)  # shape (B, H, W)
    
    iou_list = []
    for cls_id in range(num_classes):
        intersection = ((pred == cls_id) & (labels == cls_id)).sum().item()
        union = ((pred == cls_id) | (labels == cls_id)).sum().item()
        if union == 0:
            iou = 1.0  # If no pixels in union, consider IoU = 1.0 to avoid penalizing empty classes
        else:
            iou = intersection / union
        iou_list.append(iou)

    mean_iou = sum(iou_list) / len(iou_list)
    return iou_list, mean_iou



# Augment both image and labels consistently
def apply_augmentation(inputs, labels):
    # Augment the images and labels simultaneously using torchvision.transforms.functional
    if torch.rand(1).item() > 0.5:
        inputs = TF.hflip(inputs)
        labels = TF.hflip(labels)
    if torch.rand(1).item() > 0.5:
        inputs = TF.vflip(inputs)
        labels = TF.vflip(labels)
        
    if torch.rand(1).item() > 0.5:
        angle = random.uniform(-15, 15)
        inputs = TF.rotate(inputs, angle, interpolation=TF.InterpolationMode.BILINEAR)
        labels = TF.rotate(labels, angle, interpolation=TF.InterpolationMode.NEAREST)
    # Apply other augmentations (like cropping) if necessary, similar approach
    return inputs, labels


def resize_predictions(predictions, target_size=(1024, 1280), mode='nearest'):
    """
    Resizes the predictions tensor to the desired spatial size.
    
    Args:
        predictions (torch.Tensor): Input tensor of shape [batch, height, width].
        target_size (tuple): Target size as (height, width).
        mode (str): Interpolation mode ('nearest' for class labels, 'bilinear' for continuous values).
    
    Returns:
        torch.Tensor: Resized tensor of shape [batch, target_height, target_width].
    """
    # Ensure input is a floating-point tensor for interpolation (except for 'nearest')
    if mode != 'nearest' and predictions.dtype != torch.float32:
        predictions = predictions.float()
    elif mode=='nearest' and predictions.dtype==torch.long:
        predictions = predictions.float()

    # Add channel dimension if not already present
    if len(predictions.shape) == 3:  # Shape is [batch, height, width]
        predictions = predictions.unsqueeze(1)  # Add channel dimension, becomes [batch, 1, height, width]

    # Resize using the specified interpolation mode (nearest mode since interpolating class labels)
    resized_predictions = F.interpolate(predictions, size=target_size, mode='nearest')

    # Remove channel dimension (if it was added)
    if resized_predictions.shape[1] == 1:  # Shape is [batch, 1, height, width]
        resized_predictions = resized_predictions.squeeze(1)  # Remove channel dimension, becomes [batch, height, width]

    return resized_predictions

# Training loop
def train_one_epoch(model, train_loader, optimizer, criterion, device,scaler, epoch_idx):
    model.train()
    running_loss = 0.0
    total_dice_score = 0.0
    total_iou_score = 0.0
          
    pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx} [Train]", unit="batch")
    for (inputs, labels, original_labels) in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        original_labels = original_labels.to(device).long()

        optimizer.zero_grad()

        # Apply augmentations to both inputs and labels
        augmented_inputs, augmented_labels = apply_augmentation(inputs, labels)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(augmented_inputs)
            
            #In LWANet, output is onehot 128 by 128 size. need to interpolate to match 
            outputs_up = F.interpolate(outputs, size=image_size, mode='bilinear', align_corners=False)
            loss = criterion(outputs_up, augmented_labels)
            
            if torch.isnan(loss):
                print("NaN detected in loss!")
                print(f"outputs range: {outputs_up.min().item()} to {outputs_up.max().item()}")
                print(f"targets unique: {torch.unique(augmented_labels)}")
                continue
        # print(f'{loss.shape}')

        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        running_loss += loss.item()


        preds = torch.argmax(outputs_up, dim=1)
        
        shape = (original_labels.shape[1], original_labels.shape[2])
        
        resized_preds =  resize_predictions(preds, target_size=shape)
        
        total_dice_score += dice_score(resized_preds, original_labels, num_classes)
        total_iou_score += miou_score(resized_preds, original_labels, num_classes)
        
        # if batch_idx % 50 == 0:
        #     print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}')

    avg_running_loss = running_loss / len(train_loader)
    avg_dice = total_dice_score / len(train_loader)
    avg_iou = total_iou_score / len(train_loader)
    
    return avg_running_loss, avg_dice, avg_iou

# Validation loop (unchanged)
def validate(model, val_loader, criterion, device, scaler, epoch_idx):
    model.eval()
    running_loss = 0.0
    total_dice_score = 0.0
    total_iou_score = 0.0
    class_iou_sums = np.zeros(num_classes, dtype=np.float32)
    class_dice_sums = np.zeros(num_classes, dtype=np.float32)
    
    steps = 0

    # For storing IoU per class sums
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch_idx} [Train]", unit="batch")
        for (inputs, labels, original_labels) in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            original_labels = original_labels.to(device).long()


            outputs = model(inputs)
            # outputs = torch.softmax(outputs, dim=1)
            #In LWANet, output is onehot 128 by 128 size. need to interpolate to match 
            outputs_up = F.interpolate(outputs, size=image_size, mode='bilinear', align_corners=False)
            loss = criterion(outputs_up, labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs_up, dim=1)
            
            shape = (original_labels.shape[1], original_labels.shape[2])

            resized_preds =  resize_predictions(preds, target_size=shape)


            per_class_dice, batch_dice_score = dice_score(resized_preds, original_labels, num_classes, return_per_class=True)
            per_class_miou, batch_iou_score = miou_score(resized_preds, original_labels, num_classes, return_per_class=True)
            
            class_iou_sums += per_class_miou
            class_dice_sums += per_class_dice
            total_dice_score += batch_dice_score
            total_iou_score += batch_iou_score
            
            steps += 1
        
            # if batch_idx % 10 == 0:
            #     print(f"Validation Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item()}")

    avg_val_loss = running_loss / len(val_loader)
    avg_val_dice = total_dice_score / len(val_loader)
    avg_val_miou = total_iou_score / len(val_loader)
    
    avg_iou_per_class = class_iou_sums / steps  # array of shape [num_classes]
    avg_dice_per_class = class_dice_sums / steps  # array of shape [num_classes]
    

    return avg_val_loss, avg_val_dice, avg_val_miou,avg_iou_per_class, avg_dice_per_class


def train_temporal_refinery(refinery, base_model, dataloader, optimizer, criterion, device, num_classes, epoch_idx):
    refinery.train()
    total_loss = 0
    total_dice =0
    total_iou=0
    
    for prev_img, curr_img, _, curr_mask in tqdm(dataloader, f"Train Epoch {epoch_idx}:"):
        prev_img, curr_img, curr_mask = prev_img.to(device), curr_img.to(device), curr_mask.to(device)

        with torch.no_grad():
            prev_logits = base_model(prev_img)   # [B, num_classes, H, W]
            curr_logits = base_model(curr_img)

        # Handle first frame (identity warm start)
        if torch.allclose(prev_img, curr_img):
            prev_logits = curr_logits.clone()

        #Compute optical flow between raw RGB frames and produce frame refinement
        
        refined_logits, warped_prev = refinery(prev_img, curr_img, prev_logits, curr_logits, is_training=True)
        

        refined_logits_up = F.interpolate(refined_logits, size=image_size, mode='bilinear', align_corners=False)
        
        
        loss_seg = criterion(refined_logits_up, curr_mask)
        loss_temporal = temporal_consistency_loss_fast(refined_logits, warped_prev, use_logit_l1=True)

        loss = loss_seg + 0.2 * loss_temporal
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = refined_logits_up.argmax(dim=1)
        
        total_loss += loss.item()
        
        batch_dice_score = dice_score(preds, curr_mask, num_classes, return_per_class=False)
        batch_iou_score = miou_score(preds, curr_mask, num_classes, return_per_class=False)
            
        total_dice+=batch_dice_score
        total_iou+=batch_iou_score
        
    return total_loss / len(dataloader), total_iou/len(dataloader), total_dice/len(dataloader)

def validate_temporal_refinery(refinery, base_model, dataloader, criterion, device, num_classes, epoch_idx):
    refinery.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
        
    class_iou_sums = np.zeros(num_classes, dtype=np.float32)
    class_dice_sums = np.zeros(num_classes, dtype=np.float32)
    

    with torch.no_grad():
        for prev_img, curr_img, _, curr_mask in tqdm(dataloader, f"Validate Epoch {epoch_idx}:"):
            prev_img, curr_img, curr_mask = prev_img.to(device), curr_img.to(device), curr_mask.to(device)

            prev_logits = base_model(prev_img)
            curr_logits = base_model(curr_img)
            if torch.allclose(prev_img, curr_img):
                prev_logits = curr_logits.clone()

            refined_logits, warped_prev = refinery(prev_img, curr_img, prev_logits, curr_logits, is_training=True)
            
        
            
            refined_logits_up = F.interpolate(refined_logits, size=image_size, mode='bilinear', align_corners=False)
  
            loss_seg = criterion(refined_logits_up, curr_mask)
            
            loss_temporal = temporal_consistency_loss_fast(refined_logits, warped_prev, use_logit_l1=True)

            loss = loss_seg + 0.2 * loss_temporal
            preds = refined_logits_up.argmax(dim=1)

            total_loss += loss.item()
            
            per_class_dice, batch_dice_score = dice_score(preds, curr_mask, num_classes, return_per_class=True)
            per_class_iou, batch_iou_score = miou_score(preds, curr_mask, num_classes, return_per_class=True)

            total_dice+=batch_dice_score
            total_iou+=batch_iou_score
            class_iou_sums += per_class_iou
            class_dice_sums += per_class_dice

    return total_loss / len(dataloader), total_iou/len(dataloader), total_dice/len(dataloader), class_iou_sums/len(dataloader), class_dice_sums/len(dataloader)


# Define instrument mappings (background included)
# EndoVis2018 tools
# id2color = {
#     0: 0,
#     1: 1,
#     2: 2,
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 7,
# }

# id2label={
#     0: "background",
#     1: "Bipolar_Forceps",
#     2: "Prograsp_Forceps",
#     3: "Large_Needle_Driver",
#     4: "Monopolar_Curved_Scissors",
#     5: "Ultrasound_Probe",
#     6: "Suction_Instrument",
#     7: "Clip_Applier"
# }

# classes = [1, 2, 3, 4, 5, 6, 7]  # Only instrument classes

#EndoVis2018 parts

id2color = {
    0: [0,0,0],
    1: [0,255,0], #"instrument-shaft"
    2: [0,255,255], #"instrument-clasper"
    3: [125,255,12], #"instrument-wrist"
    4: [255,55,0], #"kidney-parenchyma"
    5: [24,55,125], #"covered-kidney"
    6: [187,155,25], #"thread"
    7: [ 0,255,125],#"clamps"
    8: [255,255,125],#"suturing-needle"
    9: [123,15,175], #"suction-instrument"
    10: [124,155,5], #"small-intestine"
    11: [12,255,141] , #"ultrasound-probe"
}

id2label={
    0: "background-tissue",
    1: "instrument-shaft",
    2: "instrument-clasper",
    3: "instrument-wrist",
    4: "kidney-parenchyma",
    5: "covered-kidney",
    6: "thread",
    7: "clamps",
    8: "suturing-needle",
    9:"suction-instrument",
    10: "small-intestine",
    11: "ultrasound-probe"
}

classes = [1, 2, 3, 4,5,6, 7,8,9,10,11] 


# CholecSeg8K
# id2color={
#      0 : [127, 127, 127],
#      1 : [210, 140, 140],
#      2 : [255, 114, 114],
#      3 : [231, 70, 156],
#      4 : [186, 183, 75],
#      5 : [170, 255, 0],
#      6 : [255, 85, 0],
#      7 : [255, 0, 0],
#      8 : [255, 255, 0],
#      9 : [169, 255, 184],
#     10 : [255, 160, 165],
#     11 : [0, 50, 128],
#     12 : [111, 74, 0] 
# }

# id2label={
#     0: 'Black Background',
#     1: 'Abdominal Wall',
#     2: 'Liver',
#     3: 'Gastrointestinal Tract',
#     4: 'Fat',
#     5: 'Grasper',
#     6: 'Connective Tissue',
#     7: 'Blood',
#     8: 'Cystic Duct',
#     9: 'L-hook Electrocautery',
#     10: 'Gallbladder',
#     11: 'Hepatic Vein',
#     12: 'Liver Ligament'
# }

# classes = [1, 2, 3, 4,5,6, 7,8,9,10,11, 12] 



# Define parameters
image_size = (512, 512)

batch_size = 8

num_classes = len(classes) + 1  # Including background class
validation_split = 0.2 # 20% of training data for validation


# Create dataset (Set  has_greyscale_labels to specify if labels are RGB or not)
# Specify dataset directory path for training data
# Specify sequences to use for training
# If data is all in the train folder without sequences:
# Specify root as the main dataset folder, then pass "train" as the sequence ["train"]

# train_parts
dataset_name="endovis18_holistic"
train_dataset = EndoVisDataset(
    root_dir="/notebooks/endovis18/train",
    sequences=["seq_1", "seq_3", "seq_4",  "seq_6", "seq_7",  "seq_10", "seq_11", "seq_12", "seq_13", "seq_14", "seq_16"],
    id2color=id2color,
    has_greyscale_labels = False, 
    instrument_ids=classes,
    image_size=image_size
)

val_dataset = EndoVisDataset(
    root_dir="/notebooks/endovis18/train",
    sequences=["seq_2", "seq_5",  "seq_9", "seq_15"],
    id2color=id2color,
    has_greyscale_labels = False, 
    instrument_ids=classes,
    image_size=image_size
)


# train_tools
# dataset_name="endovis18_tools"
# train_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2018_tools/",
#     sequences=["train"],
#     id2color=id2color,
#     has_greyscale_labels = True, 
#     instrument_ids=classes,
#     image_size=image_size
# )

# val_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2018_tools/",
#     sequences=["val"],
#     id2color=id2color,
#     has_greyscale_labels = True, 
#     instrument_ids=classes,
#     image_size=image_size
# )

#train cholecseg8k
# dataset_name="cholecseg8k"
# train_dataset = EndoVisDataset(
#     root_dir="/notebooks/cholecseg8k/train",
#     sequences=[ "video01", "video09", "video18", "video20", "video24", "video25", "video26", "video28", "video35", "video37", "video43", "video48", "video55"],
#     id2color=id2color,
#     instrument_ids=classes,
#     image_size=image_size,
#     has_greyscale_labels = False
# )


# val_dataset = EndoVisDataset(
#     root_dir="/notebooks/cholecseg8k/train",
#     sequences=[ "video17", "video52"],
#     id2color=id2color,
#     instrument_ids=classes,
#     image_size=image_size,
#     has_greyscale_labels = False
# )



# Split dataset into training and validation
# dataset_size = len(dataset)
# test_size=  int(test_split * dataset_size)
# train_val_size = dataset_size - test_size
# train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size])

# val_size = int(validation_split * len(dataset))
# train_size =  len(dataset) - val_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create dataloaders


temporal_train_dataset = TemporalRefinementDataset(train_dataset)
temporal_val_dataset = TemporalRefinementDataset(val_dataset)


train_loader = DataLoader(temporal_train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
val_loader = DataLoader(temporal_val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_miou = 0.0
start_epoch= 0

num_epochs = 50
# 

experiment_name="RefineryModuleWithFlow_BaseModel"


# Initialize mixed precision scaler
scaler = GradScaler()

# loss function (Combined Loss)
criterion = CombinedLoss(alpha=0.7)

checkpoint = torch.load(f"/notebooks/real_time_model/best_PyConvFASL_512.pth")

base_model = PyConvFASLLinearGauss(num_classes=num_classes)
base_model.load_state_dict(checkpoint['model_state_dict'])
base_model.to(device)

base_model.eval()  # important: inference only
for param in base_model.parameters():
    param.requires_grad = False
    

refinery = TemporalRefinery(num_classes=num_classes)

refinery.to(device)

# Set up optimizer

optimizer = optim.Adam(refinery.parameters(), lr=0.00001)

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

for epoch in range(start_epoch, num_epochs):
    
    # print(f'Endo18 Tools')
    print(f'{experiment_name}')
    print(f'Epoch {epoch+1}/{num_epochs}')

    # Train for one epoch
    train_loss, train_iou, train_dice = train_temporal_refinery(refinery, base_model, train_loader, optimizer, criterion, device, num_classes, epoch_idx=epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}] - Training Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, mIoU: {train_iou:.4f}')

    val_loss, val_iou, val_dice, avg_iou_per_class, avg_dice_per_class = validate_temporal_refinery(refinery, base_model, val_loader, criterion, device,num_classes, epoch_idx=epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, mIoU: {val_iou:.4f}')

        # Print UNet Metric results
    print(f"U-Net metrics")
    print(f"  Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
    print(f"  Val   Loss: {val_loss:.4f},   Val mIoU: {val_iou:.4f}")
    print(f"  Val   IoU/class: {[round(v, 3) for v in avg_iou_per_class]}")
    print(f"  Val   Dice/class: {[round(v, 3) for v in avg_dice_per_class]}")


    remove_files_in_folder(f"models/Ablation/ckpts")
    torch.save({ 'model_state_dict': refinery.state_dict()} , 
               f"models/Ablation/ckpts/checkpoint_epoch_{epoch+1}.pth")
    print(f"New checkpoint saved epoch:{epoch+1}")

            # Step the scheduler
    scheduler.step()
    # Optional: log current LR
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1}] - Current LR: {current_lr:.8f}")
    
    # Save the best model
    if val_iou > best_miou:
        best_miou = val_iou
        torch.save({ 'model_state_dict': refinery.state_dict()} ,
                   f"models/Ablation/{experiment_name}/{dataset_name}/best_{experiment_name}.pth")
        # _epoch_{epoch+1}_miou_{val_iou:.4f}.pth")
        print(f"New best model saved with mIoU: {best_miou:.4f}")

# Save the last model
torch.save({ 'model_state_dict': refinery.state_dict()} ,
           f"models/Ablation/{experiment_name}/{dataset_name}/last_{experiment_name}_miou_{val_iou:.4f}.pth")

