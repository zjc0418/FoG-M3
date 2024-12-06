import torch.nn as nn
import torch.nn.functional as F
from modules import *
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


class ResEncoder(nn.Module):
    def __init__(self, block, layers, in_ch):
        super(ResEncoder, self).__init__()
        self.inplanes = 32*2
        self.conv1 = nn.Conv1d(in_ch, 32*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32*2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if return_features:
            return [x4, x3, x2, x1, x0]
        else:
            return x4

class Decoder(nn.Module):
    def __init__(self, block, in_chs, out_ch, out_len):
        super(Decoder, self).__init__()
        self.up_layer1 = UpsampleBlock(in_chs[0], in_chs[1], block)
        self.up_layer2 = UpsampleBlock(in_chs[1], in_chs[2], block)
        self.up_layer3 = UpsampleBlock(in_chs[2], in_chs[3], block)
        self.up_layer4 = UpsampleBlock(in_chs[3], 32, block)
        self.up_layer5 = nn.ConvTranspose1d(32, 32, kernel_size=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(out_len)
        self.lin = nn.Linear(32,out_ch)
    def forward(self, x):
        out = self.up_layer1(x[0], x[1])
        out = self.up_layer2(out, x[2])
        out = self.up_layer3(out, x[3])
        out = self.up_layer4(out, x[4])
        out1 = self.up_layer5(out)
        out = self.pool(out1)
        out = out.permute(0,2,1)
        out = self.lin(out)
        return out    


    
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, out_len,mconfig,i_mamba, block, layers, is_basic=True):
        super(UNet, self).__init__()
        self.encoder = ResEncoder(block, layers, in_ch)
        self.mamba = MambaLMHeadModel(mconfig)  
        self.i_mamba = i_mamba
        if is_basic:
            self.decoder = Decoder(block, [1024, 512, 256,  128], out_ch, out_len)
        else:
            self.decoder = Decoder(block, [2048, 1024, 512, 256], out_ch, out_len)

    def forward(self, x):
        features = self.encoder(x, return_features=True) 
        if self.i_mamba:
            mamba_output = self.mamba(features[0])
            features[0] = mamba_output
        features_to_pass = features[0] 
        output = self.decoder(features)  
        return output, features_to_pass


def unet_18(in_ch, out_ch, out_len,mconfig, i_mamba=True):
    return UNet(in_ch, out_ch, out_len, mconfig, i_mamba, BasicBlock, [2,2,2,2])


