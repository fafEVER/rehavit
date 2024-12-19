import torch
from torch import nn
from models.common import reflect_conv
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Haloattention import HaloAttention
matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

def make_odd(value):
    return value if value % 2 == 1 else value + 1

def multiscale_retinex(image):
    scales = [3, 7, 15]  # Example scales
    retinex = image.clone()
    for scale in scales:
        odd_scale = make_odd(scale)
        gaussian_blur = GaussianBlur(kernel_size=(odd_scale, odd_scale), sigma=odd_scale // 2)
        blurred_image = gaussian_blur(image)
        log_i = torch.log1p(image)
        log_blur = torch.log1p(blurred_image)
        retinex = retinex * (log_i - log_blur)
    return retinex



def multiscale_retinex_decomposition(image_tensor, sigma_list=[15, 80, 250]):
    """
    Perform multi-scale Retinex decomposition on an image.

    Args:
        image_tensor (torch.Tensor): Input image as a PyTorch tensor.
        sigma_list (list): List of Gaussian blur sigmas for multi-scale Retinex.

    Returns:
        retinex_normalized (np.ndarray): Normalized retinex result as a NumPy array.
    """
    # Convert the input tensor to a NumPy array and normalize it to [0, 1]
    image_np = image_tensor.detach().cpu().numpy()
    image_np = np.clip(image_np, 0, 255) / 255

    # Convert the NumPy array back to a PyTorch tensor
    image_tensor = torch.tensor(image_np, dtype=torch.float32).to(image_tensor.device)

    retinex = torch.zeros_like(image_tensor, dtype=torch.float32)

    for sigma in sigma_list:
        # Apply Gaussian blur using OpenCV and convert the result back to a PyTorch tensor
        blurred_image_np = cv2.GaussianBlur(image_np, (0, 0), sigma)
        blurred_image_np[blurred_image_np == 0] = 1e-10
        blurred_image_tensor = torch.tensor(blurred_image_np, dtype=torch.float32).to(image_tensor.device)

        log_term = torch.log(image_tensor + 1) - torch.log(blurred_image_tensor + 1)
        retinex += log_term

    retinex = retinex / len(sigma_list)
    lumiance = torch.exp(retinex) * image_tensor

    # Convert the final result back to a NumPy array for normalization
    retinex_np = retinex.detach().cpu().numpy()
    retinex_normalized = (retinex_np - retinex_np.min()) / (retinex_np.max() - retinex_np.min())

    return retinex_normalized


def CMDAF(vi_feature, ir_feature):
    sigmoid = nn.Sigmoid()
    gap = nn.AdaptiveAvgPool2d(1)
    batch_size, channels, _, _ = vi_feature.size()

    sub_vi_ir = vi_feature - ir_feature
    vi_ir_div = sub_vi_ir * sigmoid(gap(sub_vi_ir))

    sub_ir_vi = ir_feature - vi_feature
    ir_vi_div = sub_ir_vi * sigmoid(gap(sub_ir_vi))

    # 特征加上各自的带有简易通道注意力机制的互补特征
    vi_feature += ir_vi_div
    ir_feature += vi_ir_div

    return vi_feature, ir_feature


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=1, out_channels=16, stride=1, padding=0)
        #
        # self.retinex = multiscale_retinex()

        # 引入Halo Attention模块
        self.halo_attention_v = HaloAttention(dim=16, block_size=2, halo_size=1)

        self.vi_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.ir_conv2 = reflect_conv(in_channels=16, kernel_size=3, out_channels=16, stride=1, pad=1)


        self.vi_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.ir_conv3 = reflect_conv(in_channels=16, kernel_size=3, out_channels=32, stride=1, pad=1)

        self.vi_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.ir_conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=64, stride=1, pad=1)

        self.vi_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.ir_conv5 = reflect_conv(in_channels=64, kernel_size=3, out_channels=128, stride=1, pad=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.LeakyReLU()
        vi_out = activate(self.vi_conv1(y_vi_image))
        ir_out = activate(self.ir_conv1(ir_image))
        # vi_out = self.halo_attention_v(vi_out)
        vi_out, ir_out = CMDAF(activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out)))
        vi_out, ir_out = CMDAF(activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out)))
        vi_out, ir_out = CMDAF(activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out)))
        vi_out, ir_out = activate(self.vi_conv5(vi_out)), activate(self.ir_conv5(ir_out))

        return vi_out, ir_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=256, kernel_size=3, out_channels=256, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=256, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=32, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class PIAFusion(nn.Module):
    def __init__(self):
        super(PIAFusion, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.fusion = Fusion(vi_out, ir_out)

    def forward(self, y_vi_image, ir_image):
        # y_vi_image = multiscale_retinex(y_vi_image, visualize=False)
        vi_encoder_out, ir_encoder_out = self.encoder(y_vi_image, ir_image)
        # encoder_out = Fusion(vi_encoder_out, ir_encoder_out)
        vi_encoder_out =multiscale_retinex(vi_encoder_out)
        encoder_out = torch.cat([vi_encoder_out, ir_encoder_out], dim=1)
        fused_image = self.decoder(encoder_out)
        return fused_image


if __name__ == "__main__":
    # encoder = Encoder()
    # decoder = Decoder()
    # y_vi_image = torch.randn(1, 1, 256, 256)
    # ir_image = torch.randn(1, 1, 256, 256)
    # vi_out, ir_out = encoder(y_vi_image, ir_image)
    # print(vi_out.shape, ir_out.shape)
    # feature_fusion = torch.cat([vi_out, ir_out], dim=1)
    # fusion = decoder(feature_fusion)
    # print(fusion.shape)
    piafusion = PIAFusion()
    y_vi_image = torch.randn(1, 1, 256, 256)
    ir_image = torch.randn(1, 1, 256, 256)
    fusion = piafusion(y_vi_image, ir_image)
    print(fusion.shape)
