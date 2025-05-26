import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic

def calculate_f1_score(prediction, target, num_classes, average=None):
    """
    Calculate F1 score for segmentation results with fixed output shape.
    
    Args:
        prediction: Predicted segmentation mask (numpy array)
        target: Ground truth segmentation mask (numpy array)
        num_classes: Total number of classes (fixed size for output)
        average: How to average F1 scores for multiple classes
                 None: returns per-class F1
                 'macro': unweighted mean of per-class F1
                 'weighted': weighted by class frequency
    
    Returns:
        F1 score (float or array of floats for each class)
    """
    # Initialize array for all possible classes
    class_f1 = np.zeros(num_classes)
    class_weights = np.zeros(num_classes)
    
    for cls in range(num_classes):  # Loop through ALL possible classes
        # Create binary masks
        pred_mask = (prediction == cls)
        gt_mask = (target == cls)
        
        # Calculate TP, FP, FN
        true_positives = np.sum(np.logical_and(pred_mask, gt_mask))
        false_positives = np.sum(np.logical_and(pred_mask, np.logical_not(gt_mask)))
        false_negatives = np.sum(np.logical_and(np.logical_not(pred_mask), gt_mask))
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_f1[cls] = f1
        class_weights[cls] = np.sum(gt_mask)
    
    # Return based on averaging method
    if average == 'macro':
        return np.mean(class_f1)
    elif average == 'weighted':
        return np.sum(class_f1 * class_weights) / np.sum(class_weights) if np.sum(class_weights) > 0 else 0
    else:
        return class_f1




class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=8, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        # Prediction has positives but ground truth is empty
        # This is a false positive (incorrectly detected something)
        return 0, 0
    elif pred.sum() == 0 and gt.sum() > 0:
        # Prediction is empty but ground truth has positives
        # This is a false negative (missed detection)
        return 0, 0
    else:
        # Both prediction and ground truth are empty
        # This is a true negative (correctly predicted nothing)
        return 1, 0
    '''   
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0
    '''



#code for iou
def IoU_bin_old(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = 1.0 * np.sum(intersection) / np.sum(union)
    return iou_score


def IoU_bin(prediction, target):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    
    # Handle empty union case
    if np.sum(union) == 0:
        # Both prediction and target are empty
        if np.sum(intersection) == 0:
            return 1.0  # Perfect match for empty masks
        else:
            return 0.0  # No overlap when one is empty
    
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def dice_coefficient(prediction, target):
    """
    Compute Sørensen-Dice coefficient between two binary masks.
    
    Args:
        prediction: Binary prediction mask
        target: Binary ground truth mask
        
    Returns:
        Dice coefficient (float between 0 and 1)
    """
    intersection = np.logical_and(target, prediction)
    
    # Calculate volumes (sum of positive pixels in each mask)
    pred_volume = np.sum(prediction)
    target_volume = np.sum(target)
    total_volume = pred_volume + target_volume
    
    # Handle empty masks case
    if total_volume == 0:
        # Both prediction and target are empty
        return 1.0  # Perfect match for empty masks
    
    # Calculate Dice coefficient: 2*|intersection|/(|prediction| + |target|)
    dice_score = 2 * np.sum(intersection) / total_volume
    return dice_score

def test_single_image1(image, label, net, classes, multimask_output,
                      patch_size=[256, 256], input_size=[224, 224],
                      test_save_path=None, case=None):

    image, label = image.squeeze(0).cpu().numpy(), label.squeeze(0).cpu().numpy()

    # image: (3, H, W), label: (H, W)
    _, x, y = image.shape

    # Resize image if needed
    if x != input_size[0] or y != input_size[1]:
        image = zoom(image, (1, input_size[0] / x, input_size[1] / y), order=3)

    _, new_x, new_y = image.shape
    if new_x != patch_size[0] or new_y != patch_size[1]:
        image = zoom(image, (1, patch_size[0] / new_x, patch_size[1] / new_y), order=3)

    inputs = torch.from_numpy(image).unsqueeze(0).float().cuda()  # shape: (1, 3, H, W)

    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().numpy()  # shape: (H, W)

    # Resize prediction back to original shape
    if prediction.shape != label.shape:
        prediction = zoom(prediction, (label.shape[0] / prediction.shape[0],
                                       label.shape[1] / prediction.shape[1]), order=0)

    # Compute metrics
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list




def test_single_volume(image, label, net, classes, multimask_output, patch_size):
    net.eval()
    image = image.cuda().unsqueeze(0)  # add batch dim
    #print(image.shape)
    with torch.no_grad():
        outputs = net(image, multimask_output, patch_size[0])
        output_masks = outputs['masks']
       # print(output_masks.shape)
        prediction = torch.argmax(F.softmax(output_masks, dim=1), dim=1).squeeze(0).cpu().numpy()
       # print(prediction.shape)
       # print(prediction[:5,:5])
        label = label.numpy()
        print(label.shape)
    # Resize prediction back to original shape
    if prediction.shape != label.shape:
        prediction = zoom(prediction, (label.shape[0] / prediction.shape[0],
                                       label.shape[1] / prediction.shape[1]), order=0)

    metric_list = [] 
    IoUs = []
    dice_score = []
    iou_scores = []
    iou_muskan = [-1]*8
    miou = 0.0
    f1_scores = calculate_f1_score(prediction, label, classes)
    macro_f1 = calculate_f1_score(prediction, label,classes,  average='macro')
    weighted_f1 = calculate_f1_score(prediction, label,classes,  average='weighted')
    for seg_class in np.unique(label):
       # if seg_class == 0:  # Skip background class
        #    continue
        seg_class_mask = prediction == seg_class
        gt_class_mask = label == seg_class
        iou = IoU_bin_old(seg_class_mask, gt_class_mask)
        iou_scores.append(iou)
        iou_muskan[int(seg_class)] = iou
        miou = np.mean(iou_scores)
 
    for i in range(0, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        IoUs.append(IoU_bin(prediction == i, label == i))
        dice_score.append(dice_coefficient(prediction == i, label == i))
    return metric_list, prediction, IoUs,dice_score,  miou, iou_muskan, f1_scores, macro_f1, weighted_f1
