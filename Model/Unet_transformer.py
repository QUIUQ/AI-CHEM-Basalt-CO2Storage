import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from  VIT import *

torch.set_default_dtype(torch.float32)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """



    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, out_channels=None):
        """
        :param channels: the number of input channels

        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )


        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        # Initial convolution
        h = self.in_layers(x)
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h



class GroupNorm16(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization

    This is a helper function, with fixed number of groups..
    """
    return GroupNorm16(16, channels)








class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, attention_levels= [ ] ,features=[16, 32, 64, 128],n_head= 4 , n_layer = 2):


        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool =  nn.ModuleList()

        self.TFencoder_layer = nn.TransformerEncoderLayer(d_model=features[-1]*2, nhead=n_head, dim_feedforward=features[-1]*8, dropout=0.1)

        for feature in features:
            self.pool.append(DownSample(feature))


        self.input = nn.Conv2d (in_channels ,features[0] ,3 ,padding=1)

        in_channels = features[0]
        # Down part of UNET
        for i , feature in enumerate(features):
            layers = []
            layers.append(ResBlock(in_channels, feature))

            if i in attention_levels:
               layers.append(SpatialTransformer(channels= feature,n_heads=n_head,n_layers=n_layer))

            self.downs.append(nn.Sequential(*layers))

            in_channels = feature

        # Up part of UNET
        for i in reversed(range(len(features))):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=features[i]*2, out_channels=features[i], kernel_size=2, stride=2)
            )
            layers = []
            layers.append(ResBlock(features[i]*2, features[i]))
            if i in attention_levels:
               layers.append(nn.TransformerEncoder(self.TFencoder_layer,n_layer))

            self.ups.append(nn.Sequential(*layers))

        self.bottom = nn.Sequential(ResBlock(features[-1], features[-1]*2),
                                    SpatialTransformer(channels=  features[-1]*2,n_heads=n_head,n_layers=n_layer)

                                    )

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


        self.final_output = nn.ReLU()





    def forward(self, x):
        skip_connections = []

        x=F.pad(x,(6,6,1,1))

        x= self.input(x)

        # down sampling
        for i ,down in enumerate(self.downs):
            x = down(x)

            skip_connections.append(x)
            x = self.pool[i](x)




        x = self.bottom(x)
        skip_connections = skip_connections[::-1]    # reverse list

        # up sampling
        # notice: we do up + DoubleConv per step
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            # check if the two cat tensors match during skip connection
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        x  = TF.resize(x,size=[30,100])

        x = self.final_conv(x)


        return x

def test():
    x = torch.randn((20, 7, 30, 100)).cuda()
    label= torch.randn((20, 4, 30, 100)).cuda()
    model = UNET(in_channels=7, out_channels=4).cuda()
    mse_loss_fn = nn.MSELoss()
    output= model(x)



    loss = mse_loss_fn(output[0], label)
    loss.backward()
    print(loss.item())

    print(output[0].shape)
    total_params = sum (p.numel () for p in model.parameters ())
    print (f"Total parameters: {total_params:,}")
    print (f'total params: {sum (p.numel () for p in model.parameters () if p.requires_grad)}')
    # assert preds.shape == x.shape

if __name__ == "__main__":
    test()
