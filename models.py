# MIDINET ARC- SKIP CONECCTIONS
from torch import nn


class encoder_decoder_net_notes_first(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(1, 2, kernel_size=(22, 1), stride=(2, 1))
        self.conv2 = nn.Conv2d(2, 4, kernel_size=(4, 10), stride=2)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=(4, 12), stride=4)
        self.conv4 = nn.Conv2d(8, 16, kernel_size=(4, 6), stride=2)

        # decoder
        self.deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=(4, 6), stride=2)
        self.deconv2 = nn.ConvTranspose2d(8, 4, kernel_size=(4, 12), stride=4)
        self.deconv3 = nn.ConvTranspose2d(4, 2, kernel_size=(4, 10), stride=4)
        self.deconv4 = nn.ConvTranspose2d(
            2, 1, kernel_size=(22, 1), stride=(2, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        # Decoder
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
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
