import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet18
from torchvision.models import resnet34
from torchvision.models import resnet50
from torchvision.models import resnet101
import torch.utils.model_zoo as model_zoo
from fightingcv_attention.attention.ECAAttention import ECAAttention
from fightingcv_attention.attention.SEAttention import SEAttention
from .fca_layer import FcaLayer

class Resnet18(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet18, self).__init__()

        self.model = resnet18(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet34(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet34, self).__init__()

        self.model = resnet34(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = avg_x + max_x
        
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)

        if self.is_norm:
            x = self.l2_norm(x)

        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet50(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet50, self).__init__()
        print("Init resnet50")
        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.uncertainty = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x_semantic = self.model.embedding(x)
        x_uncertainty = self.model.uncertainty(x)
        
        if self.is_norm:
            x_semantic = self.l2_norm(x_semantic)
            x_uncertainty = self.l2_norm(x_uncertainty)
        
        return x_semantic, x_uncertainty

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet50_attention(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet50_attention, self).__init__()
        print("Init resnet50_attention")
        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.uncertainty = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

        # attention module
        # Part I: channel attention
        # ECANet
        self.eca = False
        self.eca_semantic = ECAAttention(kernel_size=3)
        self.eca_uncertainty = ECAAttention(kernel_size=3)

        # SENet
        self.se = False
        self.se_semantic = SEAttention(channel=512,reduction=8)
        self.se_uncertainty = SEAttention(channel=512,reduction=8)
        
        # FCANet
        self.fca = True
        self.fca_semantic = FcaLayer(channel=512, reduction=16, width=1, height=1)
        self.fca_uncertainty = FcaLayer(channel=512, reduction=16, width=1, height=1)

        if [self.se, self.eca, self.fca].count(True) != 1:
            raise ValueError("Only a varible is True.")

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x_semantic = self.model.embedding(x)
        x_uncertainty = self.model.uncertainty(x)
        
        if self.is_norm:
            x_semantic = self.l2_norm(x_semantic)
            x_uncertainty = self.l2_norm(x_uncertainty)

        if self.eca or self.se or self.fca:
            x_semantic = x_semantic.unsqueeze(-1).unsqueeze(-1)
            x_uncertainty = x_uncertainty.unsqueeze(-1).unsqueeze(-1)

        if self.eca:
            x_semantic = self.eca_semantic(x_semantic)
            x_uncertainty = self.eca_uncertainty(x_uncertainty)
        elif self.se:
            x_semantic = self.se_semantic(x_semantic)
            x_uncertainty = self.se_uncertainty(x_uncertainty)
        elif self.fca:
            x_semantic = self.fca_semantic(x_semantic)
            x_uncertainty = self.fca_uncertainty(x_uncertainty)
        
        if self.eca or self.se or self.fca:
            x_semantic = torch.squeeze(x_semantic)
            x_uncertainty = torch.squeeze(x_uncertainty)

        return x_semantic, x_uncertainty

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet50_attention_multiscale(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet50_attention_multiscale, self).__init__()
        print("Init resnet50_attention_multiscale")
        self.model = resnet50(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.uncertainty = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

        # attention module
        # Part I: channel attention
        # ECANet
        self.eca = False
        self.eca_semantic = ECAAttention(kernel_size=3)
        self.eca_uncertainty = ECAAttention(kernel_size=3)

        # SENet
        self.se = False
        self.se_semantic = SEAttention(channel=512,reduction=8)
        self.se_uncertainty = SEAttention(channel=512,reduction=8)

        # FCANet
        self.fca = False
        self.fca_semantic = FcaLayer(channel=512, reduction=16, width=1, height=1)
        self.fca_uncertainty = FcaLayer(channel=512, reduction=16, width=1, height=1)

        # Dual Harmonized Focus Attention Module
        self.DHFAM = True
        self.DHFAM_semantic_fca = FcaLayer(channel=512, reduction=16, width=1, height=1)
        self.DHFAM_uncertainty_fca = FcaLayer(channel=512, reduction=16, width=1, height=1)
        self.DHFAM_semantic_eca = ECAAttention(kernel_size=3)
        self.DHFAM_uncertainty_eca = ECAAttention(kernel_size=3)

        self.DHFAM_semantic_adjusted = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU()
        )
        self.DHFAM_uncertainty_adjusted = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU()
        )
        
        # attention info
        if self.eca:
            print("Embedded ECANet")
        elif self.se:
            print("Embedded SENet")

        if [self.se, self.eca, self.fca, self.DHFAM].count(True) != 1:
            raise ValueError("Only a varible is True.")

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        l1 = self.model.layer1(x) # shape: 30, 256, 56, 56
        l2 = self.model.layer2(l1) # shape: 30, 512, 28, 28
        l3 = self.model.layer3(l2) # shape: 30, 1024, 14, 14
        l4 = self.model.layer4(l3) # shape: 30, 2048, 7, 7

        avg_x = self.model.gap(l4)
        max_x = self.model.gmp(l4)

        avg_l2 = self.model.gap(l2)
        max_l2 = self.model.gap(l2)

        avg_l3 = self.model.gap(l3)
        max_l3 = self.model.gap(l3)

        x = max_x + avg_x
        x_l2 = avg_l2 + max_l2
        x_l3 = avg_l3 + max_l3
        x_cat = torch.cat((x, x_l2, x_l3), dim=1)
        x = x.view(x_cat.size(0), -1)

        x_semantic = self.model.embedding(x)
        x_uncertainty = self.model.uncertainty(x)
        
        if self.is_norm:
            x_semantic = self.l2_norm(x_semantic)
            x_uncertainty = self.l2_norm(x_uncertainty)
        
        if self.eca or self.se or self.fca or self.DHFAM:
            x_semantic = x_semantic.unsqueeze(-1).unsqueeze(-1)
            x_uncertainty = x_uncertainty.unsqueeze(-1).unsqueeze(-1)

        if self.eca:
            x_semantic = self.eca_semantic(x_semantic)
            x_uncertainty = self.eca_uncertainty(x_uncertainty)
        elif self.se:
            x_semantic = self.se_semantic(x_semantic)
            x_uncertainty = self.se_uncertainty(x_uncertainty)
        elif self.fca:
            x_semantic = self.fca_semantic(x_semantic)
            x_uncertainty = self.fca_uncertainty(x_uncertainty)
        elif self.DHFAM:
            x_semantic_fca = self.DHFAM_semantic_fca(x_semantic)
            x_uncertainty_fca = self.DHFAM_uncertainty_fca(x_uncertainty)
            x_semantic_eca = self.DHFAM_semantic_eca(x_semantic)
            x_uncertainty_eca = self.DHFAM_uncertainty_eca(x_uncertainty)

            x_concat_semantic = torch.concat((x_semantic_fca, x_semantic_eca),dim=1)
            x_concat_uncertainty = torch.concat((x_uncertainty_fca, x_uncertainty_eca),dim=1)

            x_concat_semantic_adjusted = self.DHFAM_semantic_adjusted(x_concat_semantic)
            x_concat_uncertainty_adjusted = self.DHFAM_uncertainty_adjusted(x_concat_uncertainty)

            x_semantic = x_concat_semantic_adjusted
            x_uncertainty = x_concat_uncertainty_adjusted
            
        if self.eca or self.se or self.fca or self.DHFAM:
            x_semantic = torch.squeeze(x_semantic)
            x_uncertainty = torch.squeeze(x_uncertainty)

        return x_semantic, x_uncertainty

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)

class Resnet101(nn.Module):
    def __init__(self,embedding_size, pretrained=True, is_norm=True, bn_freeze = True):
        super(Resnet101, self).__init__()

        self.model = resnet101(pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)

        x = max_x + avg_x
        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        
        if self.is_norm:
            x = self.l2_norm(x)
            
        return x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)
