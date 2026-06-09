
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. 基础模块
# =============================================================================
class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class to_map(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.to_map = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.to_map(x)

#Dual-Pooling Spatial Enhancement Module
class DPSE(nn.Module):
    def __init__(self, spatial_dim=None):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        B, C, H, W = x.shape
        if self.spatial_dim is not None and (H != self.spatial_dim or W != self.spatial_dim):
            x_in = F.interpolate(x, size=(self.spatial_dim, self.spatial_dim), mode='bilinear', align_corners=True)
        else:
            x_in = x
      
        avg_feat = torch.mean(x_in, dim=1, keepdim=True)
        max_feat = torch.max(x_in, dim=1, keepdim=True)[0]
        att_feat = torch.cat([avg_feat, max_feat], dim=1)
        att_map = self.spatial_att(att_feat)

        enhanced = x_in * att_map

        if self.spatial_dim is not None and (H != self.spatial_dim or W != self.spatial_dim):
            enhanced = F.interpolate(enhanced, size=(H, W), mode='bilinear', align_corners=True)

        out = x + self.alpha * enhanced
        return out

#Multi-Scale Feature Fusion Module
class MSFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_d128 = nn.Conv2d(128, 128, 1)
        self.proj_g128 = nn.Conv2d(128, 128, 1)
        self.proj_d512 = nn.Conv2d(512, 128, 1)
        self.proj_g512 = nn.Conv2d(512, 128, 1)

        self.fusion_map = conv_bn_relu(256, 128)
        self.fusion_score = conv_bn_relu(512, 256)

    def forward(self, d128, d512, g128, g512):
        d128 = self.proj_d128(d128)
        g128 = self.proj_g128(g128)
        d512 = self.proj_d512(d512)
        g512 = self.proj_g512(g512)

        g128 = F.interpolate(g128, size=d128.shape[2:], mode='bilinear', align_corners=True)
        g512 = F.interpolate(g512, size=d512.shape[2:], mode='bilinear', align_corners=True)
        d512_up = F.interpolate(d512, size=d128.shape[2:], mode='bilinear', align_corners=True)
        g512_up = F.interpolate(g512, size=d128.shape[2:], mode='bilinear', align_corners=True)

        feat_for_map = self.fusion_map(torch.cat([d128, g128], dim=1))
        feat_for_score = self.fusion_score(torch.cat([d128, g128, d512_up, g512_up], dim=1))

        return feat_for_map, feat_for_score

class Decoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.decoder = conv_bn_relu(128, 64)

    def forward(self, x):
        x = self.decoder(x)
        return F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=True)

# Dual-branch network
class BICNet(nn.Module):
    def __init__(self, is_pretrain=True, size1=512, size2=256):
        super().__init__()
        self.size1 = size1
        self.size2 = size2
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if is_pretrain else None
        r1 = torchvision.models.resnet18(weights=weights)
        r2 = torchvision.models.resnet18(weights=weights)
        layers1, layers2 = list(r1.children()), list(r2.children())

        self.d_conv1 = nn.Sequential(*layers1[:3])
        self.d_max = layers1[3]
        self.d_layer1 = layers1[4]
        self.d_layer2 = layers1[5]
        self.d_layer3 = layers1[6]
        self.d_layer4 = layers1[7]

        self.d_slam0 = DPSE(128)
        self.d_slam1 = DPSE(128)
        self.d_slam2 = DPSE(64)
        self.d_slam3 = DPSE(32)
        self.d_slam4 = DPSE(16)

        self.c_conv1 = nn.Sequential(*layers2[:3])
        self.c_max = layers2[3]
        self.c_layer1 = layers2[4]
        self.c_layer2 = layers2[5]
        self.c_layer3 = layers2[6]
        self.c_layer4 = layers2[7]

        self.c_slam0 =  DPSE(64)
        self.c_slam1 =  DPSE(64)
        self.c_slam2 =  DPSE(32)
        self.c_slam3 =  DPSE(16)
        self.c_slam4 =  DPSE(8)

        self.msff = MSFF() 

#Dual-Task Dual-Prediction Head
        self.decoder = Decoder(output_size=(size1 // 4, size1 // 4))
        self.heatmap_head = to_map(64)
        self.score_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x1):
        B, C, H, W = x1.shape
        x2 = F.interpolate(x1, size=(self.size2, self.size2), mode='bilinear', align_corners=True)

        d = self.d_conv1(x1)
        d = self.d_slam0(d)
        d = self.d_max(d)
        d = self.d_slam1(self.d_layer1(d))
        d128 = self.d_slam2(self.d_layer2(d))
        d = self.d_slam3(self.d_layer3(d128))
        d512 = self.d_slam4(self.d_layer4(d))


        c = self.c_conv1(x2)
        c = self.c_slam0(c)
        c = self.c_max(c)
        c = self.c_slam1(self.c_layer1(c))
        c128 = self.c_slam2(self.c_layer2(c))
        c = self.c_slam3(self.c_layer3(c128))
        c512 = self.c_slam4(self.c_layer4(c))

        feat_map, feat_score = self.msff(d128, d512, c128, c512)

        heatmap = self.heatmap_head(self.decoder(feat_map))
        heatmap = F.interpolate(heatmap, size=(self.size1, self.size1), mode='bilinear', align_corners=True)

        score = self.score_head(feat_score).squeeze()

        return score, heatmap
