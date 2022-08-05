import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from torch.utils.tensorboard.writer import SummaryWriter


class GRUC(nn.Module):
    """
    GRU Model
    """

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUC, self).__init__()
        self.hidden_dim = hidden_dim

        self.n_layers = n_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=drop_prob
        )
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, h = self.gru(x)
        # print(f"out1: {out.size()}")
        out = self.linear2(self.relu(out))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return hidden


class NET(nn.Module):
    def __init__(
        self,
        L=20,
        N=256,
        X=8,
        R=4,
        B=256,
        H=512,
        P=3,
        norm="cLN",
        num_spks=1,
        non_linear="relu",
        causal=False
    ):
        super(NET, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d = Conv1D(1, N, L, stride=L // 2, padding=0)
        self.T = (64000 - L) // (L // 2) + 1 # 6399
        self.gru_net = GRUC(
            input_dim=H,
            hidden_dim=H,
            output_dim=B,
            n_layers=2,
            drop_prob=0.2
        )
        # n x N x T => n x N x H
        self.linear1 = nn.Sequential(
            nn.Linear(N, 837),
            nn.ReLU(),
            nn.Linear(837, 637),
            nn.ReLU(),
            nn.Linear(637, H)
        )
        # output 1x1 conv
        # n x B x T => n x N x T
        # NOTE: using ModuleList not python list
        # self.conv1x1_2 = th.nn.ModuleList(
        #     [Conv1D(B, N, 1) for _ in range(num_spks)])
        # n x B x T => n x 2N x T
        self.mask = Conv1D(B, num_spks * N, 1)
        self.linear2 = nn.Sequential(
            nn.Linear(B, 637),
            nn.ReLU(),
            nn.Linear(637, 498),
            nn.ReLU(),
            nn.Linear(498, B),
            nn.ReLU()
        )
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, bias=True)
        self.num_spks = num_spks

    

    def forward(self, x):
        # x.size() = torch.Size([16, 64000])
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
        # n x 1 x S => n x N x T
        w = self.encoder_1d(x)  # w.size(): torch.Size([16, 256, 6399])
        # n x N x T => n x T x N
        y = th.transpose(w, 1, 2)
        # n x T x N => n x T x H
        y = th.tanh(self.linear1(y)) 
        # n x T x H => n x T x B
        y, hidden = self.gru_net(y)

        y = self.linear2(y)

        # n x T x B => n x B x T
        y = th.transpose(y, 1, 2)
        # n x B x T => n x N x T
        
        e = y * w

        # spks x n x S
        output = self.decoder_1d(e, squeeze=True)  # require torch.Size([16, 64000])
        output = th.unsqueeze(output, 0) # 本框架套用分离框架，第0维表示分离出的说话人，因为为语音增强，只有一个干净语音，所以在第0维加个1，[1,16,257,637]?
        # print("output.size: ".format(output.size()))
        return output


class Conv1D(nn.Conv1d):
    """
    1D conv in GRUC
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


# 使用tensorboard工具来可视化查看网络模型
def tensorboard_show_model(model, input=None, model_name=None):
    writer = SummaryWriter(model_name)
    writer.add_graph(model, input)
    writer.close()
