from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction

# 通用函数示例
def hello_common():
    print("来自common.py的问候!")

# 定义一个检测器的主干网络模型，包含一个微型的RegNet模型和一个特征金字塔网络（FPN）
class DetectorBackboneWithFPN(nn.Module):
    """
    检测主干网络：配备特征金字塔网络（FPN）的微型RegNet模型。
    此模型接收形状为`(B, 3, H, W)`的输入图像批次，并提供三个不同FPN级别的特征，
    形状和总步长为：
        - p3级别： (out_channels, H /  8, W /  8)      步长 =  8
        - p4级别： (out_channels, H / 16, W / 16)      步长 = 16
        - p5级别： (out_channels, H / 32, W / 32)      步长 = 32
    注意：我们可以使用任何逐步对输入图像进行下采样的卷积网络架构，并将其与FPN相结合。
    我们使用的是足够小的主干，可以在Colab GPU上工作，性能也相当不错。
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # 使用ImageNet预训练权重进行初始化
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision模型仅返回最后一级的特征。但检测器的主干（带有FPN）需要不同规模的中间特征。
        # 所以我们用torchvision的特征提取器包装ConvNet。这里我们会得到带有名称（c3，c4，c5）的输出特征，
        # 同上述的(p3, p4, p5)具有相同的步长。
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # 通过一批虚拟的输入图像来推断(c3, c4, c5)的形状。
        # 特征是一个字典，键是上面定义的。值是NCHW格式的张量批次，它们提供了来自主干网络的中间特征。
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("对于形状为(2, 3, 224, 224)的虚拟输入图像")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"{level_name}特征的形状： {feature_shape}")

        # TODO：初始化FPN的附加卷积层
        # 创建三个“横向”1x1卷积层来转换(c3, c4, c5)，使它们都最终具有相同的`out_channels`。
        # 然后创建三个“输出”3x3卷积层，以将合并的FPN特征转换为输出(p3, p4, p5)特征。
        # 所有的卷积层必须有stride=1且padding保持特征在3x3卷积后不会被下采样。
        # 提示：你必须使用上面定义的`dummy_out_shapes`来决定这些层的输入/输出通道。
        
        # 这就像一个Python字典，但让PyTorch理解其中包含有可训练的权重。
        # 添加三个1x1卷积和三个3x3卷积层。
        self.fpn_params = nn.ModuleDict()
        for i, kv in enumerate(dummy_out_shapes):  
            C = kv[1][1]  # "c3":[2, 64, 28, 28], 取 64
            # 1x1卷积将结果卷到out_channel
            self.fpn_params[f'1x1conv{i+3}'] = nn.Conv2d(C,out_channels,kernel_size=1)  
            self.fpn_params[f'3x3conv{i+3}'] = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        
    # 定义一个属性，表示到FPN级别的总步长。
    # 对于固定的ConvNet，这些值对于输入图像大小是不变的。你可以随时访问这些值以在FCOS / Faster R-CNN中实现你的逻辑。
    @property
    def fpn_strides(self):
        return {"p3": 8, "p4": 16, "p5": 32}

    # 定义前向传播函数
    def forward(self, images: torch.Tensor):

        # 多尺度特征，字典键：{"c3", "c4", "c5"}。
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        # TODO：用RegNet特征（c3，c4，c5）和上面创建的FPN卷积层填充输出FPN特征（p3，p4，p5）。
        # 提示：使用`F.interpolate`进行FPN特征的上采样。
        
        m5 = self.fpn_params['1x1conv5'](backbone_feats['c5'])
        m4 = self.fpn_params['1x1conv4'](backbone_feats['c4'])
        m3 = self.fpn_params['1x1conv3'](backbone_feats['c3'])
        
        # HINT: 1x1卷积的结果是lateral，加上上层upsample之后再3x3卷积得到p，顶层p5不用加lateral（FPN论文 Figure3，我放入ipynb里了）
        p5 = self.fpn_params['3x3conv5'](m5)
        p4 = self.fpn_params['3x3conv4'](m4 + F.interpolate(m5,scale_factor=2,mode='nearest'))  # 结果加上层的upsample
        p3 = self.fpn_params['3x3conv3'](m3 + F.interpolate(m4,scale_factor=2))  # 默认nearest
        
        fpn_feats["p5"], fpn_feats["p4"], fpn_feats["p3"] = p5, p4, p3
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    
    输入fpn形态，返回fpn输出的每一个位置在原来图片的中心点位置
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        H, W = feat_shape[2:]
        # meshgrid生成一个HxW的横坐标网格和一个HxW的纵坐标网格
        # 在横坐标网格中，每一行的数字都是相同的，表示同一行的像素有相同的横坐标。
        # 同样，在纵坐标网格中，每一列的数字都是相同的，表示同一列的像素有相同的纵坐标。
        x, y = torch.meshgrid(torch.arange(H,dtype=dtype,device=device),torch.arange(W,dtype=dtype,device=device))
        # 用torch.stack函数把这两个网格叠加在一起，dim = -1 表示沿着最后一个维度合并，形成一个HxWx2的张量
        # 由于横纵坐标网格叠在一起，所以在这个张量中,每次沿着第三维度取两个值就是当前点的坐标
        locations = torch.stack([(x+0.5)*level_stride,(y+0.5)*level_stride], dim=-1) # shape(H,W,2)
        locations = locations.view(-1,2)  # 转为(H*W,2)
        location_coords[level_name] = locations
        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
