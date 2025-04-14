import torch
import torch.nn as nn
from torch.nn import Dropout, Flatten
import numpy as np

class AvgMeter:

    def __init__(self):
        self.value = 0
        self.number = 0

    def add(self, v, n):
        self.value += v
        self.number += n

    def avg(self):
        return self.value / self.number

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input > 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
                                           
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        # tmp = torch.ones_like(input)
        # tmp = torch.where(input.abs() < 0.5, 1., 0.)
        grad_input = grad_input * tmp
        return grad_input, None


class DSPIKE(nn.Module):
    def __init__(self, region=1.0):
        super(DSPIKE, self).__init__()
        self.region = region

    def forward(self, x, temp):
        out_bp = torch.clamp(x, -self.region, self.region)
        out_bp = (torch.tanh(temp * out_bp)) / \
                (2 * np.tanh(self.region * temp)) + 0.5     
        out_s = (x >= 0).float()
        return (out_s.float() - out_bp).detach() + out_bp

class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.5, gamma=1.0, dspike=False, soft_reset=True):
        """
        Implementing the LIF neurons.
         mem膜电位。T是时间步
        @param thresh: firing threshold阈值;
        @param tau: membrane potential decay factor衰减因子;
        @param gamma: hyper-parameter for controlling the sharpness in surrogate gradient;
        @param dspike: whether using rectangular gradient of dspike gradient;
        @param soft_reset: whether using soft-reset or hard-reset,True代表soft-reset.
        """
        super(LIFSpike, self).__init__()
        if not dspike:
            self.act = ZIF.apply
        else:
            # using the surrogate gradient function from Dspike:
            # https://proceedings.neurips.cc/paper/2021/file/c4ca4238a0b923820dcc509a6f75849b-Paper.pdf
            self.act = DSPIKE(region=1.0)
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.soft_reset = soft_reset

    def forward(self, x):
        mem = 0
        spike_out = []
        T = x.shape[2]
        for t in range(T):
            mem = mem * self.tau + x[:, :, t]
            spike = self.act(mem - self.thresh, self.gamma)
            mem = mem - spike * self.thresh if self.soft_reset else (1 - spike) * mem
                                                                                            
            spike_out.append(spike)

        return torch.stack(spike_out, dim=2)
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=2):
        super(AFF, self).__init__()

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_fft):
        xa = x + x_fft
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * x_fft * (1 - wei)
        return xo

class SignalIdentification(nn.Module):
    def __init__(self, **kwargs):
        super(SignalIdentification, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=8, stride=1, padding=4)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.lif_spike1 = LIFSpike(**kwargs)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.dropout1 = Dropout(0.35)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=8, stride=1, padding=4)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.lif_spike2 = LIFSpike(**kwargs)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(64, 6, kernel_size=8, stride=1, padding=4)
        self.batchnorm3 = nn.BatchNorm2d(6)
        self.lif_spike3 = LIFSpike(**kwargs)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.dropout = Dropout(0.3)
        self.flatten = Flatten()
        # 全连接层
        self.fc = nn.LazyLinear(6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.lif_spike1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.lif_spike2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.lif_spike3(x)
        x = self.maxpool3(x)

        x = self.dropout(x)

        x = self.flatten(x)
        output_iden = self.fc(x)

        return output_iden

class SAFF(nn.Module):
    def __init__(self, **kwargs):
        super(SAFF, self).__init__()

        self.AFF = AFF()
        self.signal_identification = SignalIdentification(**kwargs)

    def forward(self, f_iq, f_fft):

        attn = self.AFF(f_iq, f_fft)

        # Signal identification
        output = self.signal_identification(attn)
        return output
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
