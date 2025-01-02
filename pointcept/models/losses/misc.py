"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .builder import LOSSES
from pytorch3d.ops import knn_points


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight

@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        target = target[:, None].float()
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class ChamferDistance(nn.Module):
    def __init__(self, loss_weight=1.0):
        """
        Chamfer Distance损失，用于度量点云之间的距离。
        
        Args:
            loss_weight (float): 损失的权重。
        """
        super(ChamferDistance, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
            """
            pred: (N, 4) 去雨后点云，包含(x, y, z, p)的信息
            target: (M, 4) 去雨前的点云 GT，包含(x, y, z, p)的信息
            """
            # 计算两组点云之间的Chamfer距离
            pred = pred[:, :3]  # 只考虑空间坐标 (x, y, z)
            target = target[:, :3]  # 只考虑空间坐标 (x, y, z)

            # 使用 KNN 查找最近邻点
            adv_KNN = knn_points(pred.unsqueeze(0), target.unsqueeze(0), K=1)
            ori_KNN = knn_points(target.unsqueeze(0), pred.unsqueeze(0), K=1)

            # 计算几何距离的 Chamfer Distance
            min_dist_adv_to_ori = adv_KNN.dists.squeeze(0).min(dim=1)[0]
            min_dist_ori_to_adv = ori_KNN.dists.squeeze(0).min(dim=1)[0]
            chamfer_loss = torch.mean(min_dist_adv_to_ori) + torch.mean(min_dist_ori_to_adv)

            # # 计算每个点到最近点的距离
            # dist = torch.cdist(pred, target, p=2)  # 计算欧氏距离
            # min_dist_pred_to_gt = dist.min(dim=1)[0]  # 每个预测点到最近GT点的距离
            # min_dist_gt_to_pred = dist.min(dim=0)[0]  # 每个GT点到最近预测点的距离

            # # Chamfer损失: 对于每个点，计算最小距离的平方，并进行求和
            # chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)
            return chamfer_loss*self.loss_weight

@LOSSES.register_module()
class IntensityLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(IntensityLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        """
        pred: (N, 4) 预测的点云，包含(x, y, z, p)的信息
        target: (M, 4) GT 点云，包含(x, y, z, p)的信息
        """
        pred_intensity = pred[:, 3]  # 预测的强度
        target_intensity = target[:, 3]  # GT强度

        # 如果 pred 和 target 的大小不同，我们进行最近邻匹配
        if pred.size(0) != target.size(0):
            # # 计算空间坐标的欧氏距离 (N, M)
            dist = torch.cdist(pred[:, :3], target[:, :3], p=2)  # 计算预测点和目标点之间的欧氏距离
            min_dist_pred_to_gt, idx = dist.min(dim=1)  # 获取每个预测点到最近目标点的索引
            target_intensity = target_intensity[idx]  # 使用这些索引来选择目标点的强度值
            # 使用 KNN 查找最近邻点
            # adv_KNN = knn_points(pred_intensity.unsqueeze(0), target_intensity.unsqueeze(0), K=1)
            # ori_KNN = knn_points(target_intensity.unsqueeze(0), pred_intensity.unsqueeze(0), K=1)

            # 计算几何距离的 Chamfer Distance
            # min_dist_adv_to_ori = adv_KNN.dists.squeeze(0).min(dim=1)[0]
            # min_dist_ori_to_adv = ori_KNN.dists.squeeze(0).min(dim=1)[0]
            # intensity_loss = torch.mean(min_dist_adv_to_ori) + torch.mean(min_dist_ori_to_adv)

        # 计算强度的L2损失
        intensity_loss = F.mse_loss(pred_intensity, target_intensity)
        return intensity_loss*self.loss_weight


@LOSSES.register_module()
class ChamferDistanceWithPolarity(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0, beta=0.5):
        """
        Chamfer Distance损失，结合极性信息来度量点云之间的距离。

        Args:
            loss_weight (float): 总体损失的权重。
            alpha (float): 几何距离的权重。
            beta (float): 极性差异的权重。
        """
        super(ChamferDistanceWithPolarity, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # 几何距离的权重
        self.beta = beta    # 极性差异的权重

    def forward(self, pred, target):
        """
        pred: (N, 4) 预测的点云，包含 (x, y, z, p) 的信息
        target: (M, 4) GT点云，包含 (x, y, z, p) 的信息
        """
        # 分离出空间坐标和极性信息
        pred_xyz = pred[:, :3]  # (N, 3)，空间坐标
        target_xyz = target[:, :3]  # (M, 3)，GT空间坐标
        pred_p = pred[:, 3].unsqueeze(1)  # (N, 1)，预测点的极性
        target_p = target[:, 3].unsqueeze(0)  # (1, M)，GT点的极性

        # 计算几何距离 (x, y, z)
        dist_xyz = torch.cdist(pred_xyz, target_xyz, p=2) ** 2  # (N, M)，平方欧氏距离

        # 计算极性差异 (p)
        dist_p = (pred_p - target_p) ** 2  # (N, M)，极性差异平方

        # 综合距离: 几何距离 + 极性差异 (加权)
        combined_dist = self.alpha * dist_xyz + self.beta * dist_p  # (N, M)

        # 计算 Chamfer 距离
        min_dist_pred_to_gt = combined_dist.min(dim=1)[0]  # 每个预测点到最近GT点的距离
        min_dist_gt_to_pred = combined_dist.min(dim=0)[0]  # 每个GT点到最近预测点的距离

        # Chamfer损失: 对于每个点，计算最小距离的平方，并进行求和
        chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

        return chamfer_loss * self.loss_weight

@LOSSES.register_module()
class chamfer_loss_with_intensity(nn.Module):
    def __init__(self, loss_weight=1.0, intensity_weight=0.5):
        super(chamfer_loss_with_intensity, self).__init__()
        self.loss_weight = loss_weight
        self.intensity_weight = intensity_weight

    def forward(self, adv_pc, ori_pc):
        device = adv_pc.device

        # 只考虑空间坐标 (x, y, z)
        adv_xyz = adv_pc[:, :3]
        ori_xyz = ori_pc[:, :3]

        # 使用 KNN 查找最近邻点
        adv_KNN = knn_points(adv_xyz.unsqueeze(0), ori_xyz.unsqueeze(0), K=1)
        ori_KNN = knn_points(ori_xyz.unsqueeze(0), adv_xyz.unsqueeze(0), K=1)

        # 计算几何距离的 Chamfer Distance
        min_dist_adv_to_ori = adv_KNN.dists.squeeze(0).min(dim=1)[0]
        min_dist_ori_to_adv = ori_KNN.dists.squeeze(0).min(dim=1)[0]
        chamfer_loss = torch.mean(min_dist_adv_to_ori) + torch.mean(min_dist_ori_to_adv)

        # 计算强度差异的损失
        adv_intensity = adv_pc[:, 3]
        ori_intensity = ori_pc[:, 3]

        # 使用最近邻点对齐强度
        adv_indices = adv_KNN.idx.squeeze(0).squeeze(-1).to(device).long()
        ori_indices = ori_KNN.idx.squeeze(0).squeeze(-1).to(device).long()

        # 方向 1：从 ori_pc 中取出 adv_pc 对应的强度
        ori_intensity_aligned = ori_intensity[adv_indices]
        adv_intensity_diff_1 = torch.mean((adv_intensity - ori_intensity_aligned) ** 2)

        # 方向 2：从 adv_pc 中取出 ori_pc 对应的强度
        adv_intensity_aligned = adv_intensity[ori_indices]
        adv_intensity_diff_2 = torch.mean((ori_intensity - adv_intensity_aligned) ** 2)

        # 强度损失取平均值
        intensity_loss = (adv_intensity_diff_1 + adv_intensity_diff_2) / 2
        # 结合几何距离和强度差异的损失
        total_loss = chamfer_loss * self.loss_weight + intensity_loss * self.intensity_weight
        return total_loss


# @LOSSES.register_module()
# class chamfer_loss_with_intensity(nn.Module):
#     def __init__(self,loss_weight=1, intensity_weight=0.5):
#         """
#         Chamfer Distance损失，结合极性信息来度量点云之间的距离。

#         Args:
#             loss_weight (float): 总体损失的权重。
#             alpha (float): 几何距离的权重。
#             beta (float): 极性差异的权重。
#         """
#         super(chamfer_loss_with_intensity, self).__init__()
#         self.loss_weight = loss_weight
#         self.intensity_weight = intensity_weight    # 极性差异的权重
    
#     def forward(self, adv_pc, ori_pc):
#         """
#         计算包含强度的 Chamfer Loss。
#         :param adv_pc: 扰动点云, [N, 4] (x, y, z, intensity)
#         :param ori_pc: 原始点云, [N, 4] (x, y, z, intensity)
#         :param intensity_weight: 强度的权重 
#         :return: Chamfer Loss, 标量
#         """
#         # 调整强度的权重
#         adv_pc[..., 3] *= self.intensity_weight
#         ori_pc[..., 3] *= self.intensity_weight
        
#         # 计算最近邻
#         # 如果数据没有批量维度，直接使用 `knn_points`，并通过 unsqueeze 扩展维度
#         adv_pc = adv_pc.unsqueeze(0)  # 变为 [1, N, 4]
#         ori_pc = ori_pc.unsqueeze(0)  # 变为 [1, N, 4]
        
#               # 计算 Chamfer Loss
#         adv_KNN = knn_points(adv_pc, ori_pc, K=1)  # 保持 [batch_size, num_points, point_dim]
#         ori_KNN = knn_points(ori_pc, adv_pc, K=1)
        
#         # 计算距离并取均值
#         dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1) + ori_KNN.dists.contiguous().squeeze(-1).mean(-1)
#         return dis_loss*self.loss_weight
    

@LOSSES.register_module()
class hausdorff_loss_with_intensity(nn.Module):
    def __init__(self, loss_weight=1.0, intensity_weight=0):
        """
        Hausdorff 距离损失，结合强度信息（intensity）来度量点云之间的距离。

        Args:
            loss_weight (float): 总体损失的权重。
            intensity_weight (float): 强度（intensity）的权重。
        """
        super(hausdorff_loss_with_intensity, self).__init__()
        self.loss_weight = loss_weight
        self.intensity_weight = intensity_weight

    def forward(self, adv_pc, ori_pc):
        """
        计算包含强度的 Hausdorff Loss。
        :param adv_pc: 扰动点云, [N, 4] (x, y, z, intensity)
        :param ori_pc: 原始点云, [N, 4] (x, y, z, intensity)
        :return: Hausdorff Loss, 标量
        """
        # 调整强度的权重
        adv_pc = adv_pc.clone()  # 避免对输入数据本身进行修改
        ori_pc = ori_pc.clone()
        adv_pc[..., 3] *= self.intensity_weight
        ori_pc[..., 3] *= self.intensity_weight

        # 增加批量维度 [N, 4] -> [1, N, 4]
        adv_pc = adv_pc.unsqueeze(0)
        ori_pc = ori_pc.unsqueeze(0)

        # 计算预测点云到目标点云的最近邻
        adv_KNN = knn_points(adv_pc, ori_pc, K=1)  # [dists:[1,N,1], idx:[1,N,1]]
        
        # 计算 Hausdorff 距离: 每个点到目标点集的最大距离
        hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0]  # [1], 即一个标量

        return hd_loss * self.loss_weight
