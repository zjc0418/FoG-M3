import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
    
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_type):
        super().__init__()
        upsample = nn.Sequential(
                conv1x1(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
            ) if in_channels != out_channels else None
        block_type.expansion = 1
        self.enhance = nn.Identity()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = block_type(in_channels, out_channels, downsample=upsample)
        self.conv_bak = conv1x1(32, 128, 1)
        
    def forward(self, x1, x2):  
        x1 = self.enhance(x1)
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        if x1.shape[1] != x2.shape[1]:
            x2 = self.conv_bak(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    