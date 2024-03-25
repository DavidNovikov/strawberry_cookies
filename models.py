# MIDINET ARC- SKIP CONECCTIONS
from torch import nn
import torch.nn.functional as F
import torch




class Net_with_skip(nn.Module):
    def __init__(self):
        super(Net_with_skip, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(88, 128, (1, 11), 1, (0, 5)),  # 88
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),  # 44
            nn.Conv2d(128, 128, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((1, 2)),  # 22
            nn.Conv2d(128, 256, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),  # 11
            nn.Conv2d(256, 256, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),  # 5
            nn.Conv2d(256, 512, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, (1, 5), 1, 0),
            nn.BatchNorm2d(1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 11), 1, 0, 0),  # 11
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (1, 11), 1, (0, 5)),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(768, 256, (1, 2), 2, 0, 0),  # 22 - skip
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (1, 11), 1, (0, 5)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 128, (1, 2), 2, 0, 0),  # 44 - skip
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, (1, 11), 1, (0, 5)),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, (1, 2), 2, 0, 0),  # 88 - skip
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 88, (1, 11), 1, (0, 5)),
        )

    def forward(self, x):
        encoder_outputs = []

        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)

        skip_connections = [encoder_outputs[5], encoder_outputs[8], encoder_outputs[11]]

        for i, layer in enumerate(self.decoder):
            if i in [6, 12, 18]:
                skip = skip_connections.pop()
                x = torch.cat((x, skip), dim=1)
            x = layer(x)

        #return (x * 25).sigmoid()
        return (x * 25).sigmoid()





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(88, 128, (1, 11), 1, (0, 5)),  # 88
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),  # 44
            #nn.Dropout(0.3),  # Add dropout layer
            nn.Conv2d(128, 128, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((1, 2)),  # 22
            # nn.Dropout(0.3),  # Add dropout layer
            nn.Conv2d(128, 256, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),  # 11
            # nn.Dropout(0.3),  # Add dropout layer
            nn.Conv2d(256, 256, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.MaxPool2d((1, 2)),  # 5
            # nn.Dropout(0.3),  # Add dropout layer
            nn.Conv2d(256, 512, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, (1, 5), 1, 0),
            nn.BatchNorm2d(1024),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (1, 11), 1, 0, 0),  # 11
            nn.ReLU(True),
            nn.Conv2d(512, 512, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, (1, 2), 2, 0, 0),  # 22
            nn.ReLU(True),
            nn.Conv2d(256, 256, (1, 11), 1, (0, 5)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (1,1), 1),  # make the model bigger
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, (1, 2), 2, 0, 0),  # 44
            nn.ReLU(True),
            nn.Conv2d(128, 128, (1, 11), 1, (0, 5)),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, (1, 2), 2, 0, 0),  # 88
            nn.ReLU(True),
            nn.Conv2d(128, 128, (1,1), 1),  # make the model bigger
            nn.ReLU(True),
            nn.Conv2d(128, 88, (1, 11), 1, (0, 5)),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return (x * 25).sigmoid()

class encoder_decoder_net_notes_first(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = Down_Conv(1, 174, kernel_size=(88, 1), stride=(1, 1))
        self.conv2 = Down_Conv(174, 348, kernel_size=(1, 8), stride=4)
        self.conv3 = Down_Conv(
            348, 696, kernel_size=(1, 4), stride=2)  # stride=4
        self.conv4 = Down_Conv(
            696, 1392, kernel_size=(1, 4), stride=2)  # stride=4

        # decoder
        self.deconv1 = Up_Conv(
            1392, 696, kernel_size=(1, 5), stride=2)  # stride=4
        self.deconv2 = Up_Conv(696, 348, kernel_size=(
            1, 5), stride=2)  # stride=4:wq:wq
        self.deconv3 = Up_Conv(
            348, 174, kernel_size=(1, 8), stride=4)
        self.deconv4 = Up_Conv(
            174, 1, kernel_size=(88, 1), using_relu=False, stride=(1, 1))

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
        return x


class Up_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, using_relu=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.drop = nn.Dropout(0.3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.using_relu = using_relu
        if self.using_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.norm(x)
        if self.using_relu:
            x = self.relu(x)
        return x


class Down_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, using_relu=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)
        self.drop = nn.Dropout(0.3)
        self.norm = nn.BatchNorm2d(out_channels)
        self.using_relu = using_relu
        if self.using_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.norm(x)
        if self.using_relu:
            x = self.relu(x)
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
