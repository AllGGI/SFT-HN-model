import numpy as np
import scipy.io as sio
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score



#
# def get_logger(filename, verbosity=1, name=None):
#     level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
#     formatter = logging.Formatter(
#         "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
#     )
#     logger = logging.getLogger(name)
#     logger.setLevel(level_dict[verbosity])
#
#     fh = logging.FileHandler(filename, "w")
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#
#     sh = logging.StreamHandler()
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)
#
#     return logger


def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: 真实标签，比如[0,1,2,7,4,5,...]
    @param label_pred: 预测标签，比如[0,5,4,2,1,4,...]
    @param label_name: 标签名字，比如['cat','dog','flower',...]
    @param title: 图标题
    @param pdf_save_path: 是否保存，是则为保存路径pdf_save_path=xxx.png | xxx.pdf | ...等其他plt.savefig支持的保存格式
    @param dpi: 保存到文件的分辨率，论文一般要求至少300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=45)

    plt.tight_layout()

    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

class Neru_att(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Neru_att, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus / (4 * (x_minus.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=5, spatial_kernel=3):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))

#
class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls * 2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls * 2)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls, self.group_chnls * 2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_chnls * 2)
        )
        self.att = CBAMLayer(self.group_chnls * 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out = self.att(out)
        out += self.short_cut(x)
        return F.relu(out)


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """
    # [1, 1, 3], 8, 8
    def __init__(self, layers: object, cardinality, group_depth, num_classes) -> object:
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv2d(5, self.channels, 3, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        self.n1 = Neru_att()
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=1)
        self.n2 = Neru_att()
        self.n3 = Neru_att()
        d3 = d2
        self.conv4 = self.___make_layers(d3, layers[2], stride=1)
    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride))
            self.channels = self.cardinality * d * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.n1(out)
        out = self.conv3(out)
        out = self.n2(out)
        out = self.conv4(out)
        out = self.n3(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        return out


# import scipy.io as sio
import os
# from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# import sklearn
# import sklearn.model_selection as ms

class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.res = ResNeXt([1, 1, 3], 8, 8, 3)
        self.dnes = nn.Linear(2304, 150)

    def forward(self, x):
        tmp = torch.zeros(x.shape[0], 4, 150).cuda()
        for i in range(4):
            img = x[:, i].to(torch.float32)
            img = self.res(img)
            img = torch.flatten(img, start_dim=1)
            # print(img.shape)
            img = self.dnes(img)
            tmp[:, i] = img
        out = torch.cat([tmp[:, i] for i in range(4)], dim=1)
        return out


# Define the MT-CNN model
class MT_CNN(nn.Module):
    def __init__(self, img_size, num_classes):
        super(MT_CNN, self).__init__()
        self.base_network = BaseNetwork(img_size)
        # self.lstm = nn.LSTM(512, 128, batch_first=True)
        self.lstm = nn.LSTM(150, 36, batch_first=True, bidirectional=True)
        self.out = nn.Linear(72, num_classes)
        self.relu1 = nn.ReLU()
        self.sig = nn.Softmax(dim=1)
        self.w = nn.Parameter(torch.Tensor(36 * 2, 1))
        self.tanh2 = nn.Tanh()
        self.tanh1 = nn.Tanh()
        nn.init.uniform_(self.w, -0.1, 0.1)

    def forward(self, x):
        # (128, 5, 8, 9 ,4)
        x = self.base_network(x)
        x = x.view(x.size(0), 4, 150)  # Assumes batch size is a multiple of 6
        # x, _ = self.lstm(x)
        x, (h_n, c_n) = self.lstm(x)
        M = self.tanh1(x)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1)
        out = x * alpha
        out = torch.sum(out, 1)  # [batch_size,hidden_size * 2]
        out = self.tanh2(out)
        t = self.out(out)
        return t


model = MT_CNN((8, 9, 5), 2).cuda()
input = torch.randn((2, 4, 5, 8, 9)).cuda()
out = model(input)


