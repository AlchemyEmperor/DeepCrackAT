from torch import nn, tensor
import torch
import torch.nn.functional as F
import math
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn import ConvModule
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import types
from abc import ABCMeta, abstractmethod
import pdb
import cv2


## 单层MLP在最后一层
def Conv3X3(in_, out):
    return torch.nn.Conv2d(in_, out, 3, padding=1)
def Conv3X3_fd(in_, out, d):
    return torch.nn.Conv2d(in_, out, 3, padding=1,dilation=d)
def Conv3X3_hdc(in_, out, d, p):
    return torch.nn.Conv2d(in_, out, 3, padding=p,dilation=d)

#定义一个层
class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm2d(num_features=out)
        self.activation = torch.nn.ReLU(inplace=True)
        #self.cbam = CBAM(in_, ratio=16, kernel_size=7)

    def forward(self, x):
        #x = self.cbam(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#定义一个空洞卷积层
class ConvRelu_fd(nn.Module):
    def __init__(self, in_, out, d):
        super().__init__()
        self.conv = Conv3X3_fd(in_, out, d)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm2d(num_features=out)
        self.activation = torch.nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ConvRelu_hdc(nn.Module):
    def __init__(self, in_, out, d, p):
        super().__init__()
        self.conv = Conv3X3_hdc(in_, out, d, p)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm2d(num_features=out)
        self.activation = torch.nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#加入注意力机制
class ConvRelu_cbam(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = Conv3X3(in_, out)
        self.dropout = torch.nn.Dropout(0.3)
        self.bn = torch.nn.BatchNorm2d(num_features=out)
        self.activation = torch.nn.ReLU(inplace=True)
        self.cbam = CBAM(out, ratio=16, kernel_size=7)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.cbam(x)
        return x

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

#Shift_MLP
class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  ##(输入,输出)张量的大小
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x

class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


#DWCONV
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.dropout(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    #embed_dim=768
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


#池化
class Down(nn.Module):

    def __init__(self, nn):
        super(Down,self).__init__()
        self.nn = nn
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)


    def forward(self,inputs):
        down = self.nn(inputs)
        unpooled_shape = down.size()
        outputs, indices = self.maxpool_with_argmax(down)
        return outputs, down, indices, unpooled_shape


#Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):#kernel_size=7
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):#kernel_size=7
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

def batch_forme_v2(x, encoder, is_training, is_first_layer):
    # x: input features with the shape(B, N, C)
    # encoder:TransformerEncoderLayer(C, nhead, C, drop_rate, batch_first=False)
    if not is_training:
        return x
    orig_x = x
    if not is_first_layer:
        orig_x , x = torch.split(x, len(x)//2)
    x = encoder(x)
    x = torch.cat([orig_x, x] , dim=0)
    return x


#上采样
class Up(nn.Module):

    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.unpool=torch.nn.MaxUnpool2d(2,2)

    def forward(self,inputs,indices,output_shape):
        outputs = self.unpool(inputs, indices=indices, output_size=output_shape)
        outputs = self.nn(outputs)
        return outputs


#融合预测
class Fuse(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)
        return self.conv(outputs)

class Fuse_3D(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):

        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)
        return self.conv(outputs)

class Fuse_resize(nn.Module):

    def __init__(self, nn, scale):
        super().__init__()
        self.nn = nn
        self.scale = scale
        self.conv = Conv3X3(64,1)

    def forward(self,down_inp,up_inp):
        outputs = torch.cat([down_inp, up_inp], 1)
        outputs = F.interpolate(outputs, scale_factor=self.scale, mode='bilinear')
        outputs = self.nn(outputs)
        return self.conv(outputs)



class DeepCrack(nn.Module):

    def __init__(self, num_classes=1000, input_channels=3, deep_supervision=False,img_size=448, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.,depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], norm_layer=nn.LayerNorm):
        super(DeepCrack, self).__init__()

        self.down1 = Down(torch.nn.Sequential(
            ConvRelu_hdc(3, 64, 1, 1),
            ConvRelu_hdc(64, 64, 1, 1),
        ))

        self.down2 = Down(torch.nn.Sequential(
            ConvRelu_hdc(64, 128, 1, 1),
            ConvRelu_hdc(128, 128, 2, 2),

        ))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.down3 = Down(torch.nn.Sequential(
            ConvRelu_hdc(128, 256, 1, 1),
            ConvRelu_hdc(256, 256, 2, 2),
            ConvRelu_hdc(256, 256, 3, 3),
        ))

        self.down4 = Down(torch.nn.Sequential(
            ConvRelu_hdc(256, 512, 1, 1),
            ConvRelu_hdc(512, 512, 2, 2),
            ConvRelu_hdc(512, 512, 3, 3),

        ))


        # self.block1 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[4], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        # self.dblock2 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[3], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])

        self.up4 = Up(torch.nn.Sequential(
            ConvRelu_hdc(512, 512, 3, 3),
            ConvRelu_hdc(512, 512, 2, 2),
            ConvRelu_hdc(512, 256, 1, 1),
        ))


        self.up3 = Up(torch.nn.Sequential(
            ConvRelu_hdc(256, 256, 3, 3),
            ConvRelu_hdc(256, 256, 2, 2),
            ConvRelu_hdc(256, 128, 1, 1),
        ))

        self.up2 = Up(torch.nn.Sequential(
            ConvRelu_hdc(128, 128, 2, 2),
            ConvRelu_hdc(128, 64, 1, 1),
        ))

        self.up1 = Up(torch.nn.Sequential(
            ConvRelu_hdc(64, 64, 2, 2),
            ConvRelu_hdc(64, 64, 1, 1),
        ))





        self.fuse5 = Fuse(ConvRelu_cbam(512 + 128, 64), scale=16)
        self.fuse4 = Fuse(ConvRelu_cbam(512 + 256, 64), scale=8)
        self.fuse3 = Fuse(ConvRelu_cbam(256 + 128, 64), scale=4)
        self.fuse2 = Fuse(ConvRelu_cbam(128 + 64, 64), scale=2)
        self.fuse1 = Fuse(ConvRelu_cbam(64 + 64, 64), scale=1)

        # self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[3],
        #                                       embed_dim=embed_dims[4])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[4],
                                              embed_dim=embed_dims[4])

        self.decoder1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(512, 256, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(512)
        self.dbn2 = nn.BatchNorm2d(256)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)

        self.norm3 = norm_layer(embed_dims[4])
        self.norm4 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(512)
        self.dnorm4 = norm_layer(256)


        self.final = Conv3X3(5,1)

    def forward(self,inputs):

        B = inputs.shape[0]
        # encoder part
        # down 1-4
        out, down1, indices_1, unpool_shape1 = self.down1(inputs)
        out, down2, indices_2, unpool_shape2 = self.down2(out)
        out, down3, indices_3, unpool_shape3 = self.down3(out)
        out, down4, indices_4, unpool_shape4 = self.down4(out)

        ### Tokenized MLP Stage

        # # down 4
        # out, H, W = self.patch_embed3(out)
        # for i, blk in enumerate(self.block1):
        #     out = blk(out, H, W)
        # out = self.norm3(out)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # down4 = out.reshape(B, H*2, W*2, -1).permute(0, 3, 1, 2).contiguous()
        # # down4_shape = down4.shape

        # down 5
        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        down5 = out.reshape(B, H*2, W*2, -1).permute(0, 3, 1, 2).contiguous()


        # up 5
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2), mode='bilinear'))

        # out = torch.add(out, down4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        up5 = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # # up 4
        # out = self.dnorm3(out)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        # # out = torch.add(out, down3)
        # _, _, H, W = out.shape
        # out = out.flatten(2).transpose(1, 2)
        #
        # for i, blk in enumerate(self.dblock2):
        #     out = blk(out, H, W)
        #
        # out = self.dnorm4(out)
        # out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # up4 = out

        # up 4-1
        up4 = self.up4(up5, indices=indices_4, output_shape=unpool_shape4)
        up3 = self.up3(up4, indices=indices_3, output_shape=unpool_shape3)
        up2 = self.up2(up3, indices=indices_2, output_shape=unpool_shape2)
        up1 = self.up1(up2, indices=indices_1, output_shape=unpool_shape1)




        fuse5 = self.fuse5(down_inp=down5, up_inp=up5)
        fuse4 = self.fuse4(down_inp=down4, up_inp=up4)
        fuse3 = self.fuse3(down_inp=down3, up_inp=up3)
        fuse2 = self.fuse2(down_inp=down2, up_inp=up2)
        fuse1 = self.fuse1(down_inp=down1, up_inp=up1)


        output = self.final(torch.cat([fuse5,fuse4,fuse3,fuse2,fuse1],1))

        return output, fuse5, fuse4, fuse3, fuse2, fuse1


if __name__ == '__main__':
    inp = torch.randn((1,3,512,512))

    model = DeepCrack()

    out = model(inp)

