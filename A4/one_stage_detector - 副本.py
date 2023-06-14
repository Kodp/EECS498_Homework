import math
from typing import Dict, List, Optional

import torch
from a4_helper import *
from common import DetectorBackboneWithFPN, class_spec_nms, get_fpn_location_coords
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss

# Short hand type notation:
TensorDict = Dict[str, torch.Tensor]


def hello_one_stage_detector():
    print("Hello from one_stage_detector.py!")


class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        ######################################################################
        # TODO: Create a stem of alternating 3x3 convolution layers and RELU
        # activation modules. Note there are two separate stems for class and
        # box stem. The prediction layers for box regression and centerness
        # operate on the output of `stem_box`.
        # See FCOS figure again; 
        #@ both stems are identical.
        #
        # Use `in_channels` and `stem_channels` for creating these layers, the
        # docstring above tells you what they mean. Initialize weights of each
        # conv layer from a normal distribution with mean = 0 and std dev = 0.01
        # and all biases with zero. Use conv stride = 1 and zero padding such
        # that size of input features remains same: remember we need predictions
        # at every location in feature map, we shouldn't "lose" any locations.
        ######################################################################
        # Fill these.
        stem_cls = []
        stem_box = []  # 框子和center
        # Replace "pass" statement with your code
        
        self.num_classes = num_classes
        #TODO 注释原理、笔记
        for out_channels in stem_channels:
            convb = nn.Conv2d(in_channels,out_channels,kernel_size=3)
            nn.init.normal_(convb.weight,mean=0,std=0.01)  # 默认N(0,1)分布，mean=0，std=1
            nn.init.constant_(convb.bias,0)
            stem_box.append(convb)
            stem_box.append(nn.ReLU())
            
            convc = nn.Conv2d(in_channels,out_channels,kernel_size=3)
            nn.init.normal_(convc.weight,mean=0,std=0.01) 
            nn.init.zeros_(convc.bias)
            stem_cls.append(convc)
            stem_cls.append(nn.ReLU())
            
            in_channels = out_channels
            
        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        ######################################################################
        # TODO: Create THREE 3x3 conv layers for individually predicting three
        # things at every location of feature map:
        #     1. object class logits (`num_classes` outputs)
        #     2. box regression deltas (4 outputs: LTRB deltas from locations)  Left, Top, Right, Bottom
        #     3. centerness logits (1 output)
        #
        # Class probability and actual centerness are obtained by applying
        # sigmoid activation to these logits. 
        #!However, DO NOT initialize those modules here.
        # This module should always output logits; PyTorch loss
        # functions have numerically stable implementations with logits. During
        # inference, logits are converted to probabilities by applying sigmoid,
        # BUT OUTSIDE this module.
        #
        ######################################################################

        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls = None  # Class prediction conv
        self.pred_box = None  # Box regression conv
        self.pred_ctr = None  # Centerness conv

        # 不初始化
        # out_channels
        self.pred_cls = nn.Conv2d(out_channels,num_classes,kernel_size=1)
        self.pred_box = nn.Conv2d(out_channels,4,kernel_size=1)
        self.pred_ctr = nn.Conv2d(out_channels,1,kernel_size=1)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # OVERRIDE: Use a negative bias in `pred_cls` to improve training
        # stability. Without this, the training will most likely diverge. (发散)
        # STUDENTS: You do not need to get into details of why this is needed.
        # 这个设定的背景是，一般情况下，目标检测任务中的背景（即非目标对象）像素通常远多于前景（目标对象）像素。如果我们使用0作为初始化偏置，模型在早期训练阶段可能会预测大量的假阳性（false positives），也就是将背景像素错误地识别为前景，这会导致损失函数的值非常大，从而使得训练过程变得不稳定。
        # 通过将分类头部的偏置设为负值，我们可以使模型在早期训练阶段更倾向于预测背景类别。具体来说，对于二元分类问题，如果我们设置偏置为-log(99)，那么sigmoid激活函数的输出将会接近1/100，也就是说模型在训练初始阶段会预测出大约99%的背景像素，从而避免产生过多的假阳性。这种方式可以有效地提高训练稳定性，并加速模型的收敛。
        # 这种策略是一种启发式的方法，不一定在所有情况下都有效，但在许多目标检测模型中都已经证明了其有效性。
        torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as performing inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """

        ######################################################################
        # TODO: Iterate over every FPN feature map and obtain predictions using
        # the layers defined above. Remember that prediction layers of box
        # regression and centerness will operate on output of `stem_box`,
        # and classification layer operates separately on `stem_cls`.
        #
        # CAUTION: The original FCOS model uses shared stem for centerness and
        # classification. Recent follow-up papers commonly place centerness and
        # box regression predictors with a shared stem, which we follow here.
        # 
        # DO NOT apply sigmoid to classification and centerness logits.
        ######################################################################
        # Fill these with keys: {"p3", "p4", "p5"}, same as input dictionary.
        class_logits = {}  # layer_key : batch_size, H * W, num_classes
        boxreg_deltas = {}
        centerness_logits = {}

        # HINT: stem_box用于box和center，stem_cls用于class
        
        for L in ['p3', 'p4', 'p5']:
            feats = feats_per_fpn_level[L]
            N = feats.shape[0]
            class_out = self.pred_cls(self.stem_cls(feats))
            # 按照上面要求、docstring里的形态，先permute在合并维度
            # reshape和contiguous().view() 都是安全方法，如果只用view，则当tensor在内存中不连续时会出错
            # reshape在不连续时会返回副本操作，contiguous().view() 保证在原地
            class_out = class_out.permute(0,2,3,1).reshape(N,-1,self.num_classes)
            class_logits[L] = class_out  # (batch_size, H * W, num_classes)`
            
            box_out = self.pred_box(self.stem_box(feats))
            box_out = box_out.permute(0,2,3,1).contiguous().view(N,-1,4)
            boxreg_deltas[L] = box_out
            
            center_out = self.pred_ctr(self.stem_box(feats))
            center_out = center_out.permute(0,2,3,1).reshape(N,-1,1)
            centerness_logits[L] = center_out

        # 模型大小和张量是否连续并没有直接关系。一个张量是否连续，取决于它在内存中的存储方式。
        # 例如，如果你从一个张量中取一个切片，或者对一个张量进行转置操作，得到的新张量可能就不再是连续的，即使原张量在内存中是连续的。
        # 所以，即使你的模型小到可以完全放在内存中，也不能保证所有的张量都是连续的。
        # 在实践中，最好的做法是，如果你不确定一个张量是否连续，就使用 reshape() 或者 contiguous().view()，以避免可能的问题。
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        return [class_logits, boxreg_deltas, centerness_logits]


@torch.no_grad()
def fcos_match_locations_to_gt(
    locations_per_fpn_level: TensorDict,
    strides_per_fpn_level: Dict[str, int],
    gt_boxes: torch.Tensor,
) -> TensorDict:
    """
    Match centers of the locations of FPN feature with a set of GT bounding
    boxes of the input image. Since our model makes predictions at every FPN
    feature map location, we must supervise it with an appropriate GT box.
    There are multiple GT boxes in image, so FCOS has a set of heuristics to
    assign centers with GT, which we implement here.

    NOTE: This function is NOT BATCHED. Call separately for GT box batches.

    Args:
        locations_per_fpn_level: Centers at different levels of FPN (p3, p4, p5),
            that are already projected to absolute co-ordinates in input image
            dimension. Dictionary of three keys: (p3, p4, p5) giving tensors of
            shape `(H * W, 2)` where H = W is the size of feature map.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `common.py` for more details.
        gt_boxes: GT boxes of a single image, a batch of `(M, 5)` boxes with
            absolute co-ordinates and class ID `(x1, y1, x2, y2, C)`. In this
            codebase, this tensor is directly served by the dataloader.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(N, 5)` GT boxes, one for each center. They are
            one of M input boxes, or a dummy box called "background" that is
            `(-1, -1, -1, -1, -1)`. Background indicates that the center does
            not belong to any object.
    这段代码的主要目标是对每个FPN层级的特征点进行匹配，找到与每个特征点最匹配的目标框（GT框）。
    在匹配过程中，每个特征点都必须在其匹配的GT框内，且每个特征点只对某个尺度范围的目标框负责。
    如果有多个GT框与同一个特征点匹配，则选择面积最小的GT框。
    如果一个特征点没有匹配到任何GT框，则分配一个虚拟的背景框。
    通过这种方式，每个特征点都可以找到一个最匹配的GT框，从而进行后续的目标检测任务。
    """

    # 初始化一个字典，用于存储每个FPN层级匹配到的目标框
    matched_gt_boxes = {
        level_name: None for level_name in locations_per_fpn_level.keys()
    }

    # 对每个FPN层级分别进行匹配
    for level_name, centers in locations_per_fpn_level.items():

        # 获取这个FPN层级的步长
        stride = strides_per_fpn_level[level_name]

        # 将中心点的坐标分解为x和y
        # 维度扩展是为了后续计算方便
        x, y = centers.unsqueeze(dim=2).unbind(dim=1)

        # 将目标框的坐标分解为x0,y0,x1,y1
        # 维度扩展是为了后续计算方便
        x0, y0, x1, y1 = gt_boxes[:, :4].unsqueeze(dim=0).unbind(dim=2)

        # 计算每个特征中心点与GT框边界的距离
        pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

        # 为了后续计算方便，对距离矩阵进行维度转换
        pairwise_dist = pairwise_dist.permute(1, 0, 2)

        # 计算匹配矩阵，规则是每个锚点必须在其匹配的GT框内
        match_matrix = pairwise_dist.min(dim=2).values > 0

        # 计算每个锚点到GT框的最大距离
        pairwise_dist = pairwise_dist.max(dim=2).values

        # 确定每个FPN层级对应的目标框尺度范围
        lower_bound = stride * 4 if level_name != "p3" else 0
        upper_bound = stride * 8 if level_name != "p5" else float("inf")
        
        # 更新匹配矩阵，规则是每个锚点只对某个尺度范围的目标框负责
        match_matrix &= (pairwise_dist > lower_bound) & (
            pairwise_dist < upper_bound
        )

        # 计算每个GT框的面积
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
            gt_boxes[:, 3] - gt_boxes[:, 1]
        )

        # 更新匹配矩阵，规则是如果有多个GT框匹配到同一个锚点，选择面积最小的GT框
        match_matrix = match_matrix.to(torch.float32)
        match_matrix *= 1e8 - gt_areas[:, None]

        # 根据匹配矩阵，找到每个锚点匹配的GT框
        # 如果没有匹配到GT框，索引值为-1
        match_quality, matched_idxs = match_matrix.max(dim=0)
        matched_idxs[match_quality < 1e-5] = -1

        # 根据匹配的索引，得到每个锚点匹配的GT框
        # 如果锚点没有匹配到GT框（索引为-1），则使用一个虚拟的背景框（坐标为-1，表示不属于任何对象）
        matched_boxes_this_level = gt_boxes[matched_idxs.clip(min=0)]
        matched_boxes_this_level[matched_idxs < 0, :] = -1

        # 将这个FPN层级的匹配结果存入字典
        matched_gt_boxes[level_name] = matched_boxes_this_level

    # 返回每个FPN层级的匹配结果
    return matched_gt_boxes


def fcos_get_deltas_from_locations(
    locations: torch.Tensor, gt_boxes: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Compute distances from feature locations to GT box edges. These distances
    are called "deltas" - `(left, top, right, bottom)` or simply `LTRB`. The
    feature locations and GT boxes are given in absolute image co-ordinates.

    These deltas are used as targets for training FCOS to perform box regression
    and centerness regression. They must be "normalized" by the stride of FPN
    feature map (from which feature locations were computed, see the function
    `get_fpn_location_coords`). If GT boxes are "background", then deltas must
    be `(-1, -1, -1, -1)`.

    NOTE: This transformation function should not require GT class label. Your
    implementation must work for GT boxes being `(N, 4)` or `(N, 5)` tensors -
    without or with class labels respectively. You may assume that all the
    background boxes will be `(-1, -1, -1, -1)` or `(-1, -1, -1, -1, -1)`.

    Args:
        locations: Tensor of shape `(N, 2)` giving `(xc, yc)` feature locations.
        gt_boxes: Tensor of shape `(N, 4 or 5)` giving GT boxes.
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            Tensor of shape `(N, 4)` giving deltas from feature locations, that
            are normalized by feature stride.  (N, LTRB)
            
    输入deltas（对于stride的比例），返回由deltas计算的真实图片上的location
    """
    ##########################################################################
    # TODO: Implement the logic to get deltas from feature locations.        #
    ##########################################################################
    # Set this to Tensor of shape (N, 4) giving deltas (left, top, right, bottom)
    # from the locations to GT box edges, normalized by FPN stride.
    
    deltas = torch.empty(gt_boxes.shape[0],4,device=locations.device)
    
    deltas[:, 0] = locations[:, 0] - gt_boxes[:, 0] # xc - x1
    deltas[:, 1] = locations[:, 1] - gt_boxes[:, 1] # yc - y1
    deltas[:, 2] = gt_boxes[:, 2] - locations[:, 0] # x2 - xc
    deltas[:, 3] = gt_boxes[:, 3] - locations[:, 1] # y2 - yc
    deltas /= stride
    deltas[gt_boxes[:,0]==-1] = -1
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return deltas


def fcos_apply_deltas_to_locations(
    deltas: torch.Tensor, locations: torch.Tensor, stride: int
) -> torch.Tensor:
    """
    Implement the inverse of `fcos_get_deltas_from_locations` here:

    Given edge deltas (left, top, right, bottom) and feature locations of FPN, get
    the resulting bounding box co-ordinates by applying deltas on locations. This
    method is used for inference in FCOS: deltas are outputs from model, and
    applying them to anchors will give us final box predictions.

    Recall in above method, we were required to normalize the deltas by feature
    stride. Similarly, we have to un-normalize the input deltas with feature
    stride before applying them to locations, because the given input locations are
    already absolute co-ordinates in image dimensions.

    Args:
        deltas: Tensor of shape `(N, 4)` giving edge deltas to apply to locations.
        locations: Locations to apply deltas on. shape: `(N, 2)`
        stride: Stride of the FPN feature map.

    Returns:
        torch.Tensor
            `(N, 4)`, Same shape as deltas and locations, giving co-ordinates of the
            resulting boxes `(x1, y1, x2, y2)`, absolute in image dimensions.
            
    fcos_get_deltas_from_locations的反函数
    """
    ##########################################################################
    # TODO: Implement the transformation logic to get boxes.                 #
    #                                                                        #
    # NOTE: The model predicted deltas MAY BE negative, which is not valid   #
    # for our use-case because the feature center must lie INSIDE the final  #
    # box. Make sure to clip them to zero.                                   #
    # 我们不允许预测的目标框“翻转”，即左边界在右边，上边界在下面这样的情况。            #
    ##########################################################################
    # Replace "pass" statement with your code
    N = locations.shape[0]
    output_boxes = torch.empty(N,4,device=locations.device)
    
    deltas = deltas.clamp(min=0)
    xc, yc = locations[:,0],locations[:,1]
    output_boxes[:,0] = xc - deltas[:,0] * stride  # xc - l * stride
    output_boxes[:,1] = yc - deltas[:,1] * stride  # yc - t * stride
    output_boxes[:,2] = xc + deltas[:,2] * stride  # xc + r * stride
    output_boxes[:,3] = yc + deltas[:,3] * stride  # yc + b * stride
    
    # for i in range(N):
    #     xc, yc = locations[i]
    #     l,t,r,b = deltas[i]
    #     x1 = xc - l * stride
    #     y1 = yc - t * stride
    #     x2 = xc + r * stride
    #     y2 = yc + b * stride
    #     output_boxes[i] = torch.tensor([x1, y1, x2, y2], device=locations.device)

    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return output_boxes


def fcos_make_centerness_targets(deltas: torch.Tensor):
    """
    Given LTRB deltas of GT boxes, compute GT targets for supervising the
    centerness regression predictor. See `fcos_get_deltas_from_locations` on
    how deltas are computed. If GT boxes are "background" => deltas are
    `(-1, -1, -1, -1)`, then centerness should be `-1`.

    For reference, centerness equation is available in FCOS paper
    https://arxiv.org/abs/1904.01355 (Equation 3).

    Args:
        deltas: Tensor of shape `(N, 4)` giving LTRB deltas for GT boxes.

    Returns:
        torch.Tensor
            Tensor of shape `(N, )` giving centerness regression targets.
    """
    ##########################################################################
    # TODO: Implement the centerness calculation logic.                      #
    ##########################################################################
    # HINT: deltas LTRB
    lr,tb = deltas[:,[0,2]],deltas[:,[1,3]]
    centerness = torch.sqrt(lr.min(dim=1)[0]*tb.min(dim=1)[0]/(lr.max(dim=1)[0]*tb.max(dim=1)[0]))
    centerness[deltas.sum(dim=1)==-4]=-1
    ##########################################################################
    #                             END OF YOUR CODE                           #
    ##########################################################################

    return centerness


class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes
 
        ######################################################################
        # TODO: Initialize backbone and prediction network using arguments.  #
        ######################################################################
        self.backbone = DetectorBackboneWithFPN(out_channels=fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes=num_classes,in_channels=fpn_channels,stem_channels=stem_channels)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Averaging factor for training loss; EMA of foreground locations.
        # STUDENTS: See its use in `forward` when you implement losses.
        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored 
                during training.

        Returns:
            Losses during training and predictions during inference.
        """

        ######################################################################
        # TODO: Process the image through backbone, FPN, and prediction head #
        # to obtain model predictions at every FPN location.                 #
        # Get dictionaries of keys {"p3", "p4", "p5"} giving predicted class #
        # logits, deltas, and centerness.                                    #
        ######################################################################
        fpn_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(fpn_feats)
        ######################################################################
        # TODO: Get absolute co-ordinates `(xc, yc)` for every location in
        # FPN levels.
        #
        # HINT: You have already implemented everything, just have to
        # call the functions properly.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        fpn_feats_shapes = {
            level_name: feat.shape for level_name, feat in fpn_feats.items()
        }
        locations_per_fpn_level = get_fpn_location_coords(fpn_feats_shapes,self.backbone.fpn_strides)
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        ######################################################################
        # TODO: Assign ground-truth boxes to feature locations. We have this
        # implemented in a `fcos_match_locations_to_gt`. This operation is NOT
        # batched so call it separately per GT boxes in batch.
        ######################################################################
        # List of dictionaries with keys {"p3", "p4", "p5"} giving matched
        # boxes for locations per FPN level, per image. Fill this list:
        matched_gt_boxes = []
        # Replace "pass" statement with your code
        
        matched_gt_boxes = [fcos_match_locations_to_gt(locations_per_fpn_level,self.backbone.fpn_strides,box) for box in gt_boxes]
        
        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        matched_gt_deltas = []
        
        for box in matched_gt_boxes:
            delta_p3 = fcos_get_deltas_from_locations(
                locations_per_fpn_level['p3'], box['p3'], self.backbone.fpn_strides['p3']
            )
            delta_p4 = fcos_get_deltas_from_locations(
                locations_per_fpn_level['p4'], box['p4'], self.backbone.fpn_strides['p4']
            )
            delta_p5 = fcos_get_deltas_from_locations(
                locations_per_fpn_level['p5'], box['p5'], self.backbone.fpn_strides['p5']
            )
            matched_gt_deltas.append({'p3':delta_p3,'p4':delta_p4,'p5':delta_p5})
            
        ######################################################################
        #                           END OF YOUR CODE                         #
        ######################################################################

        # Collate lists of dictionaries, to dictionaries of batched tensors.
        # These are dictionaries with keys {"p3", "p4", "p5"} and values as
        # tensors of shape (batch_size, locations_per_fpn_level, 5 or 4)
        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image

        #######################################################################
        # TODO: Calculate losses per location for classification, box reg and
        # centerness. Remember to set box/centerness losses for "background"
        # positions to zero.
        ######################################################################
        # Feel free to delete this line: (but keep variable names same)
        loss_cls, loss_box, loss_ctr = None, None, None
        # Replace "pass" statement with your code
        # 获取预测的分类logits的维度信息。其中 N 为批量大小，WH 为特征图上的位置数（宽度乘以高度），C 为类别数
        # 获取预测的分类logits的维度信息。其中 N 为批量大小，WH 为特征图上的位置数（宽度乘以高度），C 为类别数。
        N, WH, C = pred_cls_logits.shape

        # 获取匹配的真实偏移量的维度信息。其中 B 为批量大小，WH 为特征图上的位置数（宽度乘以高度）。
        B, WH, _ = matched_gt_deltas.shape

        # 对真实偏移量进行形状变换并计算其对应的中心性目标，之后再将其形状变换回原来的形状。这里的中心性目标表示每个位置的目标对象的中心性，即该位置与目标对象的中心的接近程度。
        center_gt = fcos_make_centerness_targets(matched_gt_deltas.reshape(-1,4)).reshape(B,WH,1)

        # 创建一个全零的one-hot编码张量，用于记录每个位置的类别信息。张量的形状是 (N*WH, C+1)，表示有 N*WH 个位置，每个位置有 C+1 个可能的类别（包含背景类）。
        gt_cls_onehot = torch.zeros((N*WH, C+1),device=pred_cls_logits.device)

        # 将真实类别对应的位置在one-hot编码张量中设置为1，表示这个位置的真实类别。
        gt_cls_onehot[torch.arange(N*WH),matched_gt_boxes[:,:,-1].reshape(-1).long()] = 1

        # 将one-hot编码张量的形状变换回原来的形状，并去掉背景类别，得到真实的类别标签。
        label = gt_cls_onehot[:,:-1].reshape(N,WH,-1)

        # 计算预测的类别logits和真实的类别标签之间的sigmoid focal loss。这个损失表示预测的类别和真实的类别之间的不一致程度。
        loss_cls = sigmoid_focal_loss(pred_cls_logits, label.float())

        # 计算预测的偏移量和真实的偏移量之间的smooth L1 loss，并乘以一个系数0.25。这个损失表示预测的边框位置和真实的边框位置之间的不一致程度。这里，任何负数的偏移量（表示这个位置没有匹配到任何真实目标）的损失都会被设置为零，意味着对于背景的预测不会对损失产生贡献。
        loss_box = 0.25 * F.smooth_l1_loss(pred_boxreg_deltas, matched_gt_deltas, reduction="none")
        loss_box[matched_gt_deltas < 0] *= 0.0
        
        
        # 对于预测的中心性值和真实的中心性值，我们使用二元交叉熵损失（binary cross entropy loss）来计算它们之间的差异。
        # 预测的中心性值来自模型的输出（pred_ctr_logits），而真实的中心性值（center_gt）在之前的步骤中已经计算出来。
        # 'reduction="none"'表示我们会对每一个预测位置分别计算损失值，而不是得到一个总体的损失值。
        loss_ctr = F.binary_cross_entropy_with_logits(pred_ctr_logits, center_gt, reduction="none")

        # 如果一个预测位置对应的真实中心性值是负数（这意味着这个位置不包含任何目标对象，是背景），那么我们就将这个位置的中心性损失值设为0。
        # 这样做的目的是只对那些包含目标对象的预测位置计算损失，忽略掉背景的预测位置。
        loss_ctr[center_gt < 0] *= 0.0

        # 如果一个预测位置对应的真实中心性值是负数（这意味着这个位置不包含任何目标对象，是背景），那么我们就将这个位置的中心性损失值设为0。
        # 这样做的目的是只对那些包含目标对象的预测位置计算损失，忽略掉背景的预测位置。
        loss_ctr[center_gt < 0] *= 0.0

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################
        # Sum all locations and average by the EMA of foreground locations.
        # In training code, we simply add these three and call `.backward()`
        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes. 

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.  NMS过滤高度重叠的预测
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.  下面维度说明了，本来第一维是batch的，推理时不是batch，而是单张，所以用0
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]  # (layer_key : batch_size, H * W, num_classes)-> (H * W, num_classes)
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]

            ##################################################################
            # TODO: FCOS uses the geometric mean of class probability and
            # centerness as the final confidence score. This helps in getting
            # rid of excessive amount of boxes far away from object centers.
            # Compute this value here (recall sigmoid(logits) = probabilities)
            #
            # Then perform the following steps in order:
            #   1. Get the most confidently predicted class and its score for
            #      every box. Use level_pred_scores: (N, num_classes) => (N, )
            #   2. Only retain prediction that have a confidence score higher
            #      than provided threshold in arguments.
            #   3. Obtain predicted boxes using predicted deltas and locations
            #   4. Clip XYXY box-cordinates that go beyond thr height and
            #      and width of input image.
            ##################################################################
            # Feel free to delete this line: (but keep variable names same)        
            
            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )  # (H * W, num_classes)
            # Step 1:
            # 对于每个框，获取最有信心预测的类别及其分数。 Use level_pred_scores: (N, num_classes) => (N, )
            level_pred_classes,classes = level_pred_scores.max(dim=1) # (N,) (N,)
            # Step 2:
            # 只保留具有高于参数中提供的阈值的置信度分数的预测。
            keep = level_pred_scores > test_score_thresh
            
            level_pred_classes = classes[keep]  # (N,)
            level_pred_scores = level_pred_scores[keep]  # (N,)

            # Step 3:
            # 使用预测的差值和位置获取预测框
            level_pred_boxes = fcos_apply_deltas_to_locations(level_deltas,level_locations,self.backbone.fpn_strides[level_name])  # (N, 4)
            level_pred_boxes = level_pred_boxes[keep]  # 对应去除
            
            # Step 4: Use `images` to get (height, width) for clipping.
            # 裁剪那些超出输入图像高度和宽度的 XYXY 盒坐标
            _, _, H, W = images.shape
            for i, v in enumerate([0,0,H,W]):  # 左上裁到(0,0)，右下裁到(H,W)
                level_pred_boxes[:,i].clamp_(min=v)
                
            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################

            pred_boxes_all_levels.append(level_pred_boxes)
            pred_classes_all_levels.append(level_pred_classes)
            pred_scores_all_levels.append(level_pred_scores)

        ######################################################################
        # Combine predictions from all levels and perform NMS.
        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        # STUDENTS: This function depends on your implementation of NMS.
        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
