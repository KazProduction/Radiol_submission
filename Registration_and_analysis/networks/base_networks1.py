from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal

def custom_padding_3d(x):
# padding same as Tensorflow when asymmetry
    x = F.pad(x, [1, 1, 1, 1, 1, 1], mode='constant', value=0)
    x = x[:, :, 1:, 1:, 1:]
    return x

def ReLU(x):
    return F.relu(x)

def LeakyReLU(x, alpha=0.1):
    return F.leaky_relu(x, negative_slope=alpha)

class conv3_leakyrelu_block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, alpha=0.1, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride, padding, bias=bias)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out

class conv3_block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, ksize, stride, padding, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out

class upconv3_leakyrelu_block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, alpha=0.1, stride=1, padding=1):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, ksize, stride, padding, bias=False)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.upconv(x)
        out = self.activation(out)
        return out

class upconv3_block(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=1):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, ksize, stride, padding, bias=False)

    def forward(self, x):
        out = self.upconv(x)
        return out

class VTN(nn.Module):
    def __init__(self, flow_value, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_value = flow_value
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = 3

        c = self.channels

        # encoder
        self.enc = nn.ModuleList()
        self.enc.append(conv3_leakyrelu_block(2, c, 3, stride=2)) #64 * 64 * 64
        self.enc.append(conv3_leakyrelu_block(c, c*2, 3, stride=2)) #32 * 32 * 32
        self.enc.append(conv3_leakyrelu_block(c*2, c*4, 3, stride=2)) # 16 * 16 * 16
        self.enc.append(conv3_leakyrelu_block(c*4, c*4, 3, stride=1))#16 * 16 * 16
        self.enc.append(conv3_leakyrelu_block(c*4, c*8, 3, stride=2)) #8 * 8 * 8
        self.enc.append(conv3_leakyrelu_block(c*8, c*8, 3, stride=1)) #8 * 8 * 8
        self.enc.append(conv3_leakyrelu_block(c*8, c*16, 3, stride=2)) #4 * 4 * 4
        self.enc.append(conv3_leakyrelu_block(c*16, c*16, 3, stride=1)) #4 * 4 * 4
        self.enc.append(conv3_leakyrelu_block(c*16, c*32, 3, stride=2)) #2 * 2 * 2
        self.enc.append(conv3_leakyrelu_block(c*32, c*32, 3, stride=1)) #2 * 2 * 2

        self.dec_ext = nn.ModuleList()
        self.dec_ext.append(conv3_block(c * 32, self.dim, 3, stride=1))  # 2 * 2 * 2 * 3
        self.dec_ext.append(conv3_block(c * 16 * 2 + self.dim, self.dim, 3, stride=1))  # 4 * 4 * 4 * 3
        self.dec_ext.append(conv3_block(c * 8 * 2 + self.dim, self.dim, 3, stride=1))  # 8 * 8 * 8 * 3
        self.dec_ext.append(conv3_block(c * 4 * 2 + self.dim, self.dim, 3, stride=1))  # 16 * 16 * 16 * 3
        self.dec_ext.append(conv3_block(c * 2 * 2 + self.dim, self.dim, 3, stride=1))  # 32 * 32 * 32 * 3
        self.dec_ext.append(upconv3_block(c * 2 + self.dim, self.dim, 4, stride=2))  # 128 * 128 * 128 * 3

        self.dec = nn.ModuleList()
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 4* 4 * 4 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 32, c * 16, 4, stride=2))  # 4 * 4 * 4 * (c * 16)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 8 * 8 * 8 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 16 * 2 + self.dim, c * 8, 4, stride=2))  # 8 * 8 * 8 * (c * 8)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2))  # 16 * 16 * 16 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 8 * 2 + self.dim, c * 4, 4, stride=2)) # 16 * 16 * 16 * (c * 4)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 32 * 32 * 32 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 4 * 2 + self.dim, c * 2, 4, stride=2))  # 32 * 32 * 32 * (c * 2)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 64 * 64 * 64 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 2 * 2 + self.dim, c, 4, stride=2))  # 64 * 64 * 64 * (c)


    def forward(self, x):
        # encode
        conv1_x = self.enc[0](x)
        conv2_x = self.enc[1](conv1_x)
        conv3_x = self.enc[2](conv2_x)
        conv31_x = self.enc[3](conv3_x)
        conv4_x = self.enc[4](conv31_x)
        conv41_x = self.enc[5](conv4_x)
        conv5_x = self.enc[6](conv41_x)
        conv51_x = self.enc[7](conv5_x)
        conv6_x = self.enc[8](conv51_x)
        conv61_x = self.enc[9](conv6_x)

        # decode
        pred6 = self.dec_ext[0](conv61_x)
        upsamp6to5 = self.dec[0](pred6)
        deconv5 = self.dec[1](conv61_x)
        concat5 = torch.cat([conv51_x, deconv5, upsamp6to5], dim=1) # dim=1 is channel in torch

        pred5 = self.dec_ext[1](concat5)
        upsamp5to4 = self.dec[2](pred5)
        deconv4 = self.dec[3](concat5)
        concat4 = torch.cat([conv41_x, deconv4, upsamp5to4], dim=1)

        pred4 = self.dec_ext[2](concat4)
        upsamp4to3 = self.dec[4](pred4)
        deconv3 = self.dec[5](concat4)
        concat3 = torch.cat([conv31_x, deconv3, upsamp4to3], dim=1)

        pred3 = self.dec_ext[3](concat3)
        upsamp3to2 = self.dec[6](pred3)
        deconv2 = self.dec[7](concat3)
        concat2 = torch.cat([conv2_x, deconv2, upsamp3to2], dim=1)

        pred2 = self.dec_ext[4](concat2)
        upsamp2to1 = self.dec[8](pred2)
        deconv1 = self.dec[9](concat2)
        concat1 = torch.cat([conv1_x, deconv1, upsamp2to1], dim=1)

        pred0 = self.dec_ext[5](concat1)

        return {'flow': pred0 * self.flow_value * self.flow_multiplier}

class VTN_CustomPadding(nn.Module):
    def __init__(self, flow_value, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_value = flow_value
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = 3

        c = self.channels

        # encoder
        self.enc = nn.ModuleList()
        self.enc.append(conv3_leakyrelu_block(2, c, 3, stride=2, padding=0)) #64 * 64 * 64
        self.enc.append(conv3_leakyrelu_block(c, c*2, 3, stride=2, padding=0)) #32 * 32 * 32
        self.enc.append(conv3_leakyrelu_block(c*2, c*4, 3, stride=2, padding=0)) # 16 * 16 * 16
        self.enc.append(conv3_leakyrelu_block(c*4, c*4, 3, stride=1))#16 * 16 * 16
        self.enc.append(conv3_leakyrelu_block(c*4, c*8, 3, stride=2, padding=0)) #8 * 8 * 8
        self.enc.append(conv3_leakyrelu_block(c*8, c*8, 3, stride=1)) #8 * 8 * 8
        self.enc.append(conv3_leakyrelu_block(c*8, c*16, 3, stride=2, padding=0)) #4 * 4 * 4
        self.enc.append(conv3_leakyrelu_block(c*16, c*16, 3, stride=1)) #4 * 4 * 4
        self.enc.append(conv3_leakyrelu_block(c*16, c*32, 3, stride=2, padding=0)) #2 * 2 * 2
        self.enc.append(conv3_leakyrelu_block(c*32, c*32, 3, stride=1)) #2 * 2 * 2

        self.dec_ext = nn.ModuleList()
        self.dec_ext.append(conv3_block(c * 32, self.dim, 3, stride=1))  # 2 * 2 * 2 * 3
        self.dec_ext.append(conv3_block(c * 16 * 2 + self.dim, self.dim, 3, stride=1))  # 4 * 4 * 4 * 3
        self.dec_ext.append(conv3_block(c * 8 * 2 + self.dim, self.dim, 3, stride=1))  # 8 * 8 * 8 * 3
        self.dec_ext.append(conv3_block(c * 4 * 2 + self.dim, self.dim, 3, stride=1))  # 16 * 16 * 16 * 3
        self.dec_ext.append(conv3_block(c * 2 * 2 + self.dim, self.dim, 3, stride=1))  # 32 * 32 * 32 * 3
        self.dec_ext.append(upconv3_block(c * 2 + self.dim, self.dim, 4, stride=2))  # 128 * 128 * 128 * 3

        self.dec = nn.ModuleList()
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 4* 4 * 4 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 32, c * 16, 4, stride=2))  # 4 * 4 * 4 * (c * 16)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 8 * 8 * 8 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 16 * 2 + self.dim, c * 8, 4, stride=2))  # 8 * 8 * 8 * (c * 8)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2))  # 16 * 16 * 16 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 8 * 2 + self.dim, c * 4, 4, stride=2)) # 16 * 16 * 16 * (c * 4)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 32 * 32 * 32 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 4 * 2 + self.dim, c * 2, 4, stride=2))  # 32 * 32 * 32 * (c * 2)
        self.dec.append(upconv3_block(self.dim, self.dim, 4, stride=2)) # 64 * 64 * 64 * 3
        self.dec.append(upconv3_leakyrelu_block(c * 2 * 2 + self.dim, c, 4, stride=2))  # 64 * 64 * 64 * (c)


    def forward(self, x):
        # encode
        x = custom_padding_3d(x)
        ori_conv1_x = self.enc[0](x)
        conv1_x = custom_padding_3d(ori_conv1_x)
        ori_conv2_x = self.enc[1](conv1_x)
        conv2_x = custom_padding_3d(ori_conv2_x)
        conv3_x = self.enc[2](conv2_x)
        ori_conv31_x = self.enc[3](conv3_x)
        conv31_x = custom_padding_3d(ori_conv31_x)
        conv4_x = self.enc[4](conv31_x)
        ori_conv41_x = self.enc[5](conv4_x)
        conv41_x = custom_padding_3d(ori_conv41_x)
        conv5_x = self.enc[6](conv41_x)
        ori_conv51_x = self.enc[7](conv5_x)
        conv51_x = custom_padding_3d(ori_conv51_x)
        conv6_x = self.enc[8](conv51_x)
        conv61_x = self.enc[9](conv6_x)

        # decode
        pred6 = self.dec_ext[0](conv61_x)
        upsamp6to5 = self.dec[0](pred6)
        deconv5 = self.dec[1](conv61_x)
        concat5 = torch.cat([ori_conv51_x, deconv5, upsamp6to5], dim=1) # dim=1 is channel in torch

        pred5 = self.dec_ext[1](concat5)
        upsamp5to4 = self.dec[2](pred5)
        deconv4 = self.dec[3](concat5)
        concat4 = torch.cat([ori_conv41_x, deconv4, upsamp5to4], dim=1)

        pred4 = self.dec_ext[2](concat4)
        upsamp4to3 = self.dec[4](pred4)
        deconv3 = self.dec[5](concat4)
        concat3 = torch.cat([ori_conv31_x, deconv3, upsamp4to3], dim=1)

        pred3 = self.dec_ext[3](concat3)
        upsamp3to2 = self.dec[6](pred3)
        deconv2 = self.dec[7](concat3)
        concat2 = torch.cat([ori_conv2_x, deconv2, upsamp3to2], dim=1)

        pred2 = self.dec_ext[4](concat2)
        upsamp2to1 = self.dec[8](pred2)
        deconv1 = self.dec[9](concat2)
        concat1 = torch.cat([ori_conv1_x, deconv1, upsamp2to1], dim=1)

        pred0 = self.dec_ext[5](concat1)

        return {'flow': pred0 * self.flow_value * self.flow_multiplier}

class VoxelMorph(nn.Module):
    def __init__(self, flow_value, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        self.dim = 3
        self.encoders = [m * channels for m in [1, 2, 2, 2]]
        self.decoders = [m * channels for m in [2, 2, 2, 2, 2, 1, 1]] + [3]

        # encoder
        self.enc = nn.ModuleList()
        self.enc.append(conv3_leakyrelu_block(2, self.encoders[0], 3, stride=2))  # 64 * 64 * 64
        self.enc.append(conv3_leakyrelu_block(self.encoders[0], self.encoders[1], 3, stride=2))  # 32 * 32 * 32
        self.enc.append(conv3_leakyrelu_block(self.encoders[1], self.encoders[2], 3, stride=2))  # 16 * 16 * 16
        self.enc.append(conv3_leakyrelu_block(self.encoders[2], self.encoders[3], 3, stride=2))  # 8 * 8 * 8

        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(conv3_leakyrelu_block(self.encoders[3], self.decoders[0], 3))
        self.dec.append(conv3_leakyrelu_block(self.decoders[0] * 2, self.decoders[1], 3))
        self.dec.append(conv3_leakyrelu_block(self.decoders[1] * 2, self.decoders[2], 3))
        self.dec.append(conv3_leakyrelu_block(self.decoders[2] + self.encoders[0], self.decoders[3], 3))
        self.dec.append(conv3_leakyrelu_block(self.decoders[3], self.decoders[4], 3))
        self.dec.append(conv3_leakyrelu_block(self.decoders[4] + 2, self.decoders[5], 3))
        self.dec.append(conv3_leakyrelu_block(self.decoders[5], self.decoders[6], 3))
        self.dec.append(nn.Conv3d(self.decoders[6], self.decoders[7], kernel_size=3, padding=1))

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        nd = Normal(0, 1e-5)
        self.dec[-1].weight = nn.Parameter(nd.sample(self.dec[-1].weight.shape))
        self.dec[-1].bias = nn.Parameter(torch.zeros(self.dec[-1].bias.shape))

    def forward(self, x):
        # encode
        conv1_x = self.enc[0](x)
        conv2_x = self.enc[1](conv1_x)
        conv3_x = self.enc[2](conv2_x)
        conv4_x = self.enc[3](conv3_x)

        # decode
        deconv4 = self.dec[0](conv4_x)
        upsamp4to3 = self.upsample(deconv4)
        concat3 = torch.cat([upsamp4to3, conv3_x], dim=1)  # dim=1 is channel in torch

        deconv3 = self.dec[1](concat3)
        upsamp3to2 = self.upsample(deconv3)
        concat2 = torch.cat([upsamp3to2, conv2_x], dim=1)

        deconv2 = self.dec[2](concat2)
        upsamp2to1 = self.upsample(deconv2)
        concat1 = torch.cat([upsamp2to1, conv1_x], dim=1)

        deconv1 = self.dec[3](concat1)
        deconv1_1 = self.dec[4](deconv1)
        upsamp1to0 = self.upsample(deconv1_1)
        concat0 = torch.cat([upsamp1to0, x], dim=1)

        deconv0 = self.dec[5](concat0)
        deconv0_1 = self.dec[6](deconv0)
        flow = self.dec[7](deconv0_1)

        return {'flow': flow  * self.flow_multiplier}


def affine_flow(W, b, len1, len2, len3):
    b = torch.reshape(b, [-1, 1, 1, 1, 3])
    xr = torch.arange(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, dtype=torch.float32)
    xr = torch.reshape(xr, [1, -1, 1, 1, 1])
    yr = torch.arange(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, dtype=torch.float32)
    yr = torch.reshape(yr, [1, 1, -1, 1, 1])
    zr = torch.arange(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, dtype=torch.float32)
    zr = torch.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = torch.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = torch.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = torch.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr.to(wx.device) * wx + yr.to(wy.device) * wy) + (zr.to(wz.device) * wz + b)

def det3x3(M):

    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return M[0][0] * M[1][1] * M[2][2] +  M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1] - \
           (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])


def elem_sym_polys_of_eigen_values(M):

    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    sigma1 = M[0][0] + M[1][1] + M[2][2]
    sigma2 = M[0][0] * M[1][1] + M[1][1] * M[2][2] + M[2][2] * M[0][0] - \
             (M[0][1] * M[1][0] + M[1][2] * M[2][1] + M[2][0] * M[0][2])
    sigma3 = M[0][0] * M[1][1] * M[2][2] + M[0][1] * M[1][2] * M[2][0] + M[0][2] * M[1][0] * M[2][1] - \
             (M[0][0] * M[1][2] * M[2][1] + M[0][1] * M[1][0] * M[2][2] + M[0][2] * M[1][1] * M[2][0])
    return sigma1, sigma2, sigma3

class VTNAffine(nn.Module):
    def __init__(self, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        self.dim = 3
        self.channels = channels

        c = self.channels
        # conv
        self.conv = nn.ModuleList()
        self.conv.append(conv3_leakyrelu_block(2, c, 3, stride=2)) #64 * 64 * 64
        self.conv.append(conv3_leakyrelu_block(c, c*2, 3, stride=2)) #32 * 32 * 32
        self.conv.append(conv3_leakyrelu_block(c*2, c*4, 3, stride=2)) # 16 * 16 * 16
        self.conv.append(conv3_leakyrelu_block(c*4, c*4, 3, stride=1))#16 * 16 * 16
        self.conv.append(conv3_leakyrelu_block(c*4, c*8, 3, stride=2)) #8 * 8 * 8
        self.conv.append(conv3_leakyrelu_block(c*8, c*8, 3, stride=1)) #8 * 8 * 8
        self.conv.append(conv3_leakyrelu_block(c*8, c*16, 3, stride=2)) #4 * 4 * 4
        self.conv.append(conv3_leakyrelu_block(c*16, c*16, 3, stride=1)) #4 * 4 * 4
        self.conv.append(conv3_leakyrelu_block(c*16, c*32, 3, stride=2)) #2 * 2 * 2
        self.conv.append(conv3_leakyrelu_block(c*32, c*32, 3, stride=1)) #2 * 2 * 2

        self.conv.append(conv3_block(c*32, 9, 2, padding=0, bias=False))
        self.conv.append(conv3_block(c*32, 3, 2, padding=0, bias=False))

    def forward(self, x):
        conv1 = self.conv[0](x)
        conv2 = self.conv[1](conv1)
        conv3 = self.conv[2](conv2)
        conv31 = self.conv[3](conv3)
        conv4 = self.conv[4](conv31)
        conv41 = self.conv[5](conv4)
        conv5 = self.conv[6](conv41)
        conv51 = self.conv[7](conv5)
        conv6 = self.conv[8](conv51)
        conv61 = self.conv[9](conv6)

        conv7W = self.conv[10](conv61)
        conv7b = self.conv[11](conv61)

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = torch.reshape(conv7W, [-1, 3, 3]) * self.flow_multiplier
        b = torch.reshape(conv7b, [-1, 3]) * self.flow_multiplier
        A = W + torch.repeat_interleave(torch.from_numpy(np.array(I)).float().to(W.device), W.shape[0], dim=0)
        sx, sy, sz = x.shape[2:5] # torch tensor = [B, C, H, W, D]
        flow = affine_flow(W, b, sx, sy, sz).permute(0, 4, 1, 2, 3)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = torch.norm(det.float() - 1.0)**2/2
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = torch.matmul(torch.transpose(A, 1, 2), A) + torch.repeat_interleave(torch.from_numpy(np.array(epsI)).float().to(A.device), A.shape[0], dim=0)
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = torch.sum(ortho_loss, dim=0)

        return {'flow': flow, 'W': W, 'b': b, 'det_loss': det_loss, 'ortho_loss': ortho_loss}

class VTNAffine_CustomPadding(nn.Module):
    def __init__(self, flow_multiplier=1., channels=16):
        super().__init__()
        self.flow_multiplier = flow_multiplier
        self.dim = 3
        self.channels = channels

        c = self.channels
        # conv
        self.conv = nn.ModuleList()
        self.conv.append(conv3_leakyrelu_block(2, c, 3, stride=2, padding=0)) #64 * 64 * 64
        self.conv.append(conv3_leakyrelu_block(c, c*2, 3, stride=2, padding=0)) #32 * 32 * 32
        self.conv.append(conv3_leakyrelu_block(c*2, c*4, 3, stride=2, padding=0)) # 16 * 16 * 16
        self.conv.append(conv3_leakyrelu_block(c*4, c*4, 3, stride=1))#16 * 16 * 16
        self.conv.append(conv3_leakyrelu_block(c*4, c*8, 3, stride=2, padding=0)) #8 * 8 * 8
        self.conv.append(conv3_leakyrelu_block(c*8, c*8, 3, stride=1)) #8 * 8 * 8
        self.conv.append(conv3_leakyrelu_block(c*8, c*16, 3, stride=2, padding=0)) #4 * 4 * 4
        self.conv.append(conv3_leakyrelu_block(c*16, c*16, 3, stride=1)) #4 * 4 * 4
        self.conv.append(conv3_leakyrelu_block(c*16, c*32, 3, stride=2, padding=0)) #2 * 2 * 2
        self.conv.append(conv3_leakyrelu_block(c*32, c*32, 3, stride=1)) #2 * 2 * 2

        self.conv.append(conv3_block(c*32, 9, 2, padding=0, bias=False))
        self.conv.append(conv3_block(c*32, 3, 2, padding=0, bias=False))

    def forward(self, x):
        ori_x = x
        x = custom_padding_3d(x)
        conv1 = self.conv[0](x)
        conv1 = custom_padding_3d(conv1)
        conv2 = self.conv[1](conv1)
        conv2 = custom_padding_3d(conv2)
        conv3 = self.conv[2](conv2)
        conv31 = self.conv[3](conv3)
        conv31 = custom_padding_3d(conv31)
        conv4 = self.conv[4](conv31)
        conv41 = self.conv[5](conv4)
        conv41 = custom_padding_3d(conv41)
        conv5 = self.conv[6](conv41)
        conv51 = self.conv[7](conv5)
        conv51 = custom_padding_3d(conv51)
        conv6 = self.conv[8](conv51)
        conv61 = self.conv[9](conv6)

        conv7W = self.conv[10](conv61)
        conv7b = self.conv[11](conv61)

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = torch.reshape(conv7W, [-1, 3, 3]) * self.flow_multiplier
        b = torch.reshape(conv7b, [-1, 3]) * self.flow_multiplier
        A = W + torch.repeat_interleave(torch.from_numpy(np.array(I)).float().to(W.device), W.shape[0], dim=0)
        sx, sy, sz = ori_x.shape[2:5] # torch tensor = [B, C, H, W, D]
        flow = affine_flow(W, b, sx, sy, sz).permute(0, 4, 1, 2, 3)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = torch.norm(det.float() - 1.0)**2/2
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = torch.matmul(torch.transpose(A, 1, 2), A) + torch.repeat_interleave(torch.from_numpy(np.array(epsI)).float().to(A.device), A.shape[0], dim=0)
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = torch.sum(ortho_loss, dim=0)

        return {'flow': flow, 'W': W, 'b': b, 'det_loss': det_loss, 'ortho_loss': ortho_loss}