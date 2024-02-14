# MIDINET ARC- SKIP CONECCTIONS
from torch import nn
import torch.nn.functional as F



class encoder_decoder_net_notes_first(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = Down_Conv(1, 16, kernel_size=(22, 1), stride=(2, 1))
        self.conv2 = Down_Conv(16, 32, kernel_size=(4, 10), stride=2)
        self.conv3 = Down_Conv(32, 64, kernel_size=(4, 12), stride=4)
        self.conv4 = Down_Conv(64, 128, kernel_size=(4, 6), stride=2)

        # decoder
        self.deconv1 = Up_Conv(128, 64, kernel_size=(4, 6), stride=2)
        self.deconv2 = Up_Conv(
            64, 32, kernel_size=(4, 12), stride=4)
        self.deconv3 = Up_Conv(
            32, 16, kernel_size=(4, 10), stride=2)
        self.deconv4 = Up_Conv(
            16, 1, kernel_size=(22, 1), stride=(2, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Decoder
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.sigmoid(x/0.001)
        #x = F.threshold(x, 0.5, 0 )
        return x


class Up_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.drop = nn.Dropout(0.3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm(self.drop(self.conv(x))))
        return x


class Down_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.drop = nn.Dropout(0.3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.norm(self.drop(self.conv(x))))
        return x

# class encoder_decoder_net(nn.Module):
#     # under the assumtion that we start with 1X16X16
#     def __init__(self, input_dim):
#         super(encoder_decoder_net, self).__init__()

#         # encoder
#         self.conv1 = nn.Conv2d(input_dim, input_dim*2, kernel_size=4, stride=4)
#         self.conv2 = nn.Conv2d(input_dim*2, input_dim *
#                                4, kernel_size=4, stride=4)
#         self.conv3 = nn.Conv2d(input_dim * 4, input_dim *
#                                8, kernel_size=4, stride=2)
#         self.conv4 = nn.Conv2d(input_dim * 8, input_dim *
#                                16, kernel_size=4, stride=2)

#         # decoder
#         self.deconv1 = nn.ConvTranspose2d(
#             input_dim*16, input_dim*8, kernel_size=4, stride=2)
#         self.deconv2 = nn.ConvTranspose2d(
#             input_dim*8, input_dim * 4, kernel_size=4, stride=2)
#         self.deconv3 = nn.ConvTranspose2d(
#             input_dim * 4, input_dim * 2, kernel_size=4, stride=4)
#         self.deconv4 = nn.ConvTranspose2d(
#             input_dim * 2, input_dim, kernel_size=4, stride=4)

#         self.relu = nn.ReLU()

#     def forward(self, x):
#         # Encoder
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))
#         # Decoder
#         x = self.relu(self.deconv1(x))
#         x = self.relu(self.deconv2(x))
#         x = self.relu(self.deconv3(x))
#         x = self.relu(self.deconv4(x))
#         return x
