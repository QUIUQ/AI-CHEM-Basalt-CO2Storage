import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from  VIT import *



torch.set_default_dtype(torch.float32)

# finite difference filters, for calculating 1st & 2nd derivatives
# 1 st derivative of x and y axes
dx = [[[[0, 0, 0],
        [-1 / 2, 0, 1 / 2],
        [0, 0, 0]]]]

dy = [[[[0, -1 / 2, 0],
        [0, 0, 0],
        [0, 1 / 2, 0]]]]

# 2 nd derivative of x and y axes
dxx = [[[[0, 0, 0],
         [1, -2, 1],
         [0, 0, 0]]]]

dyy = [[[[0, 1, 0],
         [0, -2, 0],
         [0, 1, 0]]]]


# -----------------below are non-trainable CNNs for calculating derivatives-----------------------
class Dx(nn.Module):
    def __init__(self, dx_filter, in_channel, out_channel):
        super(Dx, self).__init__()
        self.conv_dx = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(dx_filter).repeat(in_channel,1,1,1).to('cuda'), requires_grad=False)

    def forward(self, x):
        dx_value = self.conv_dx(x)
        return dx_value


class Dxx(nn.Module):
    def __init__(self, dx_filter, in_channel, out_channel):
        super(Dxx, self).__init__()
        self.conv_dx = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dx.weight = nn.Parameter(torch.FloatTensor(dx_filter).repeat(in_channel,1,1,1).to('cuda'), requires_grad=False)

    def forward(self, x):
        dx_value = self.conv_dx(x)
        # dxx_value = self.conv_dx(dx_value)
        return dx_value


class Dy(nn.Module):
    def __init__(self, dy_filter, in_channel, out_channel):
        super(Dy, self).__init__()
        self.conv_dy = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dy.weight = nn.Parameter(torch.FloatTensor(dy_filter).repeat(in_channel,1,1,1).to('cuda'), requires_grad=False)

    def forward(self, x):
        dy_value = self.conv_dy(x)
        return dy_value


class Dyy(nn.Module):
    def __init__(self, dy_filter, in_channel, out_channel):
        super(Dyy, self).__init__()
        self.conv_dyy = nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='replicate', groups=in_channel,
                                 bias=False)
        self.conv_dyy.weight = nn.Parameter(torch.FloatTensor(dy_filter).repeat(in_channel,1,1,1).to('cuda'), requires_grad=False)

    def forward(self, x):
        dyy_value = self.conv_dyy(x)

        return dyy_value


class Total_loss (nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Total_loss, self).__init__()
        dx = [[[[0 ,0 ,0] ,
                [-1 / 2 ,0 ,1 / 2] ,
                [0 ,0 ,0]]]]

        dy = [[[[0 ,-1 / 2 ,0] ,
                [0 ,0 ,0] ,
                [0 ,1 / 2 ,0]]]]

        # 2 nd derivative of x and y axes
        dxx = [[[[0 ,0 ,0] ,
                 [1 ,-2 ,1] ,
                 [0 ,0 ,0]]]]

        dyy = [[[[0 ,1 ,0] ,
                 [0 ,-2 ,0] ,
                 [0 ,1 ,0]]]]
        self.dx = Dx(dx, in_channel, out_channel)
        self.dy = Dy(dy, in_channel, out_channel)
        self.dxx = Dxx(dxx, in_channel, out_channel)
        self.dyy = Dyy(dyy, in_channel, out_channel)
        self.lossfunction = nn.MSELoss()

    def forward(self, label_real, label , pred ):
        # calculate derivatives
        label_dx = self.dx(label)
        label_dy = self.dy(label)
        label_dxx = self.dxx(label)
        label_dyy = self.dyy(label)

        pred_dx = self.dx(pred)
        pred_dy = self.dy(pred)
        pred_dxx = self.dxx(pred)
        pred_dyy = self.dyy(pred)

        ##region penalty
        mask = (label_real!=0)

        # calculate loss
        loss = self.lossfunction(pred, label)
        loss += 0.1*self.lossfunction(pred_dx, label_dx)
        loss += 0.1*self.lossfunction(pred_dy, label_dy)
        #loss += 0.1*self.lossfunction(pred_dxx, label_dxx)
        #loss += 0.1*self.lossfunction(pred_dyy, label_dyy)
        #loss += 0.1*self.lossfunction(pred[mask], label_real[mask])
        return loss

##test
if __name__ == '__main__':
    model = Total_loss(4,4)
    label_real = torch.randn(1,4,256,256)
    label_processed = torch.randn(1,4,256,256)
    pred = torch.randn(1,4,256,256)
    loss = model(label_real, label_processed , pred)

    print(loss.backward())


