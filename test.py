import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch.nn.functional as F
from dataset import EndoVisDataset

from PyConvFASLLinearGauss import PyConvFASLLinearGauss

from RefineryModuleWithFlow import TemporalRefinery
from temporal_dataset import TemporalRefinementDataset

from metrics.dice_score import dice_score
from metrics.miou_score import miou_score
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

best_miou = 0.0

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
        return self.alpha * tversky_loss + (1 - self.alpha) * ce_loss

# Define Tversky Loss
class TverskyLoss(torch.nn.Module):
    def __init__(self, smooth=1, alpha=0.7, beta=0.3):
        super(TverskyLoss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
       
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



#EndoVis2018 parts

# id2color = {
#     0: [0,0,0],
#     1: [0,255,0], #"instrument-shaft"
#     2: [0,255,255], #"instrument-clasper"
#     3: [125,255,12], #"instrument-wrist"
#     4: [255,55,0], #"kidney-parenchyma"
#     5: [24,55,125], #"covered-kidney"
#     6: [187,155,25], #"thread"
#     7: [ 0,255,125],#"clamps"
#     8: [255,255,125],#"suturing-needle"
#     9: [123,15,175], #"suction-instrument"
#     10: [124,155,5], #"small-intestine"
#     11: [12,255,141] , #"ultrasound-probe"
# }

# id2label={
#     0: "background-tissue",
#     1: "instrument-shaft",
#     2: "instrument-clasper",
#     3: "instrument-wrist",
#     4: "kidney-parenchyma",
#     5: "covered-kidney",
#     6: "thread",
#     7: "clamps",
#     8: "suturing-needle",
#     9:"suction-instrument",
#     10: "small-intestine",
#     11: "ultrasound-probe"
# }

# classes = [1, 2, 3, 4,5,6, 7,8,9,10,11] 

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




# # # # CholecSeg8K
id2color={
     0 : [127, 127, 127],
     1 : [210, 140, 140],
     2 : [255, 114, 114],
     3 : [231, 70, 156],
     4 : [186, 183, 75],
     5 : [170, 255, 0],
     6 : [255, 85, 0],
     7 : [255, 0, 0],
     8 : [255, 255, 0],
     9 : [169, 255, 184],
    10 : [255, 160, 165],
    11 : [0, 50, 128],
    12 : [111, 74, 0] 
}

id2label={
    0: 'Black Background',
    1: 'Abdominal Wall',
    2: 'Liver',
    3: 'Gastrointestinal Tract',
    4: 'Fat',
    5: 'Grasper',
    6: 'Connective Tissue',
    7: 'Blood',
    8: 'Cystic Duct',
    9: 'L-hook Electrocautery',
    10: 'Gallbladder',
    11: 'Hepatic Vein',
    12: 'Liver Ligament'
}

classes = [1, 2, 3, 4,5,6,7,8,9,10,11, 12] 


# Define parameters

image_size=(512,512)

batch_size = 1
num_epochs = 100
num_classes = len(classes) + 1  # Including background class
# validation_split = 0.2 # 20% of training data for validation

# Initialize the feature extractor
# extractor = SegformerImageProcessor(size=image_size)

# dataset_name = "endovis18_holistic"
# test_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis18/test",
#     sequences=["seq_1", "seq_2", "seq_3", "seq_4"],
#     # sequences=["seq_1"],
#     # sequences=[ "seq_2"],
#     # sequences=["seq_3"],
#     # sequences=["seq_4"],
    
#     id2color=id2color,
#     has_greyscale_labels = False, 
#     instrument_ids=classes,
#     image_size=image_size
# )

# dataset_name = "endovis18_tools"
# test_dataset = EndoVisDataset(
#     root_dir="/notebooks/endovis2018_tools",
#     sequences=["val"],
#     id2color=id2color,
#     has_greyscale_labels = True, 
#     instrument_ids=classes,
#     image_size=image_size
# )


    
dataset_name="cholecseg8k"
test_dataset = EndoVisDataset(
    root_dir="/notebooks/cholecseg8k/test",
    sequences=["video12", "video27"],
    id2color=id2color,
    has_greyscale_labels = False, 
    instrument_ids=classes,
    image_size=image_size
)



test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PyConvFASLLinearGauss(num_classes=num_classes)

experiment_name = "PyConvFASLLinearGauss"

checkpoint = torch.load(f"/notebooks/models/{experiment_name}/{dataset_name}/best_{experiment_name}_512.pth")


model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

refinery_checkpoint = torch.load(f"/notebooks//models/RefineryModuleWithFlow/{dataset_name}/alpha.1/best_RefineryModuleWithFlow.pth")
refinery = TemporalRefinery(num_classes=num_classes)

refinery.load_state_dict(refinery_checkpoint['model_state_dict'])
refinery.to(device)

# Set up optimizer and loss function (Combined Loss)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = CombinedLoss(alpha=0.7)


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
# Test function to evaluate mIoU and Dice score
def test_model(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    total_dice_score = 0.0
    total_iou_score = 0.0
    class_iou_sums = np.zeros(num_classes, dtype=np.float32)
    class_dice_sums = np.zeros(num_classes, dtype=np.float32)
 
#     total_fpr_score=0.0
#     class_fpr_sums = np.zeros(num_classes, dtype=np.float32)
    
    steps = 0
    criterion = CombinedLoss(alpha=0.7)  # You may use the same loss as in training
    
    prev_frame = None
    prev_pred = None
    
    with torch.no_grad():
        # for batch_idx, batch in enumerate(test_loader):
        # with open(f"./{dataset_name}_log.txt", "a") as log_file:
        pbar = tqdm(test_loader, desc=f"Testing:", unit="batch")
        for (inputs, labels, original_labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                original_labels = original_labels.to(device).long()
                shape = (original_labels.shape[1], original_labels.shape[2])
                # Forward pass
                outputs = model(inputs)

                #No Refinery:
                # outputs_up = F.interpolate(outputs, size=image_size, mode='bilinear', align_corners=False)

                ##With Refinery:
                if prev_frame is None:
                    prev_pred = outputs
                    outputs_up = F.interpolate(outputs, size=image_size, mode='bilinear', align_corners=False)
                    prev_frame=inputs
                else:
                    refined_outputs = refinery(prev_frame, inputs, prev_pred, outputs)

                    
                    # refined_outputs=refinery(outputs, prev_pred)
                    prev_pred = refined_outputs
                    prev_frame=inputs
                    outputs_up = F.interpolate(refined_outputs, size=image_size, mode='bilinear', align_corners=False)

                loss = criterion(outputs_up, labels)
                running_loss += loss.item()

                preds = torch.argmax(outputs_up, dim=1)

                resized_preds =  resize_predictions(preds, target_size=shape)

                per_class_miou, batch_iou = miou_score(resized_preds, original_labels, num_classes, return_per_class=True)
                # log_file.write(f"{batch_iou}:{per_class_miou}\n")
                per_class_dice, batch_dice = dice_score(resized_preds, original_labels, num_classes, return_per_class=True)
                # log_file.write(f"{batch_dice}:{per_class_dice}\n")

                # per_class_fpr, batch_fpr = fpr_score(resized_preds, original_labels, num_classes, return_per_class=True)

                total_iou_score += batch_iou
                total_dice_score += batch_dice

                class_iou_sums += per_class_miou
                class_dice_sums += per_class_dice

                steps += 1

    
    avg_test_loss = running_loss / len(test_loader)
    avg_test_dice = total_dice_score / len(test_loader)
    avg_test_miou = total_iou_score / len(test_loader)
    avg_iou_per_class = class_iou_sums / steps  # array of shape [num_classes]
    avg_dice_per_class = class_dice_sums / steps  # array of shape [num_classes]

    return avg_test_loss, avg_test_dice, avg_test_miou,avg_iou_per_class, avg_dice_per_class 

# # Test the model
test_loss, test_dice, test_iou,avg_iou_per_class, avg_dice_per_class = test_model(model, test_loader, device)
print(f"---------------------------Results {dataset_name} --------------------------------")
print(f'  Test Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, mIoU: {test_iou:.4f}')
print(f"  Test IoU/class: {[round(v, 3) for v in avg_iou_per_class]}")
print(f"  Test Dice/class: {[round(v, 3) for v in avg_dice_per_class]}")
# # print(f"  Test fpr/class: {[round(v, 3) for v in avg_fpr_per_class]}")



# # EXTRA STATS::::
# import time

# def evaluate_with_fps(model, dataloader, device):
#     model.eval()
#     total_frames = 0
#     elapsed_time=0
#     # start_time = time.time()

#     prev_frame = None
#     prev_pred = None
    
#     with torch.no_grad():
#         for (inputs, _, _) in dataloader:
#             inputs = inputs.to(device)  # shape: [B, C, H, W]
#             batch_size = inputs.size(0)
            
#             start_time = time.time()
#             outputs = model(inputs)  # Run inference
#             elapsed_time+= time.time() - start_time
            
#             if prev_frame is None:
#                 prev_pred = outputs
#                 # outputs_up = F.interpolate(outputs, size=image_size, mode='bilinear', align_corners=False)
#                 prev_frame=inputs
#             else:
# #                 prev_bgr = tensor_to_bgr(prev_frame[0])   # take batch element
# #                 curr_bgr = tensor_to_bgr(inputs[0])

# #                 start_time = time.time()
# #                 flow = compute_optical_flow(prev_bgr, curr_bgr)  # [B, 2, H, W]
# #                 elapsed_time+= time.time() - start_time
                
# #                 flow.to(device)
                
# #                 start_time = time.time()
# #                 warped_prev_pred = warp_with_flow(prev_pred, flow)
# #                 elapsed_time+= time.time() - start_time
                
# #                 # Predict refinement
                
#                 start_time = time.time()
# #                 refined_outputs = refinery(warped_prev_pred, outputs)
# #                 elapsed_time+= time.time() - start_time
                
#                 # start_time = time.time()
#                 # flow = compute_optical_flow_batch(prev_frame, inputs, flow_scale=0.5, cuda_device=device)
#                 # warped_prev = warp_logits_with_flow(prev_pred, flow, use_fp16=True)
#                 # refined_outputs = refinery(warped_prev, outputs)
                
#                 refined_outputs = refinery(prev_frame, inputs, prev_pred, outputs)
                
#                 elapsed_time +=time.time()-start_time

#                 # refined_outputs=refinery(outputs, prev_pred)
#                 prev_pred = refined_outputs
#                 prev_frame=inputs
#                 # outputs_up = F.interpolate(refined_outputs, size=image_size, mode='bilinear', align_corners=False)

#             # elapsed_time+=time.time()-start_time
#             total_frames += batch_size  # Count total frames processed
#             if total_frames==100:
#                 break

#     # elapsed_time = time.time() - start_time
#     fps = total_frames / elapsed_time
#     print(f"Processed {total_frames} frames in {elapsed_time:.2f}s → {fps:.2f} FPS")
#     return fps

# total_fps=0.0
# for i in range(10):
#     fps= evaluate_with_fps(model, test_loader, device)
#     total_fps+=fps
#     print(f"FPS: {fps:.4f}")
    
# print(f"Mean FPS: {total_fps/10.0}")

# torch.cuda.reset_peak_memory_stats()


# for (inputs, labels, original_labels) in test_loader:
#     inputs = inputs.to(device)  # shape: [B, C, H, W]
#     _ = model(inputs)  # input with shape [B, C, H, W]
#     break
# torch.cuda.synchronize()
# peak_memory = torch.cuda.max_memory_allocated()
# print(f"Peak memory: {peak_memory / (1024 ** 2):.2f} MB")

# def measure_peak_memory(model, dataloader, device):
#     model.eval()
#     peak_memory = 0

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = batch['pixel_values'].to(device)

#             torch.cuda.reset_peak_memory_stats()
#             torch.cuda.synchronize()
#             _ = model(inputs)
#             torch.cuda.synchronize()

#             current_peak = torch.cuda.max_memory_allocated()
#             peak_memory = max(peak_memory, current_peak)

#     print(f"Peak GPU memory across all batches: {peak_memory / (1024 ** 2):.2f} MB")
#     return peak_memory

# _ = measure_peak_memory(model, test_loader, device)