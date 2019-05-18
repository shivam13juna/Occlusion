import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, c_in, c_out, f=64, network_mode='upsampling'):
        super(Generator, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.f = f

        self.encoder = nn.Sequential(
            nn.Conv2d(self.c_in, self.f * 2, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.f * 2, self.f * 4, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.f * 4, self.f * 8, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.f * 8, self.f * 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 16),
            nn.LeakyReLU(negative_slope=0.2),
        )

        if network_mode == 'upsampling':
            self.decoder = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.f * 16, self.f * 8, 5, stride=1, padding=2),
                nn.BatchNorm2d(self.f * 8),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.f * 8, self.f * 4, 5, stride=1, padding=2),
                nn.BatchNorm2d(self.f * 4),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.f * 4, self.f * 2, 5, stride=1, padding=2),
                nn.BatchNorm2d(self.f * 2),
                nn.ReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.f * 2, self.f, 5, stride=1, padding=2),
                nn.BatchNorm2d(self.f),
                nn.ReLU(),
                nn.Conv2d(self.f, self.c_out, 5, stride=1, padding=2),
                nn.Tanh()
            )
        elif network_mode == 'transposed_conv':
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.f * 16, self.f * 8, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(self.f * 8),
                nn.ReLU(),
                nn.ConvTranspose2d(self.f * 8, self.f * 4, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(self.f * 4),
                nn.ReLU(),
                nn.ConvTranspose2d(self.f * 4, self.f * 2, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(self.f * 2),
                nn.ReLU(),
                nn.ConvTranspose2d(self.f * 2, self.f, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(self.f),
                nn.ReLU(),
                nn.Conv2d(self.f, self.c_out, 5, stride=1, padding=2),
                nn.Tanh()
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, c, f=64, input_shape=(128, 64)):
        super(Discriminator, self).__init__()
        self.c = c
        self.f = f

        self.input_shape = input_shape
        self.fc_dim = self.f * 16 * (self.input_shape[0] // 16) * (self.input_shape[1] // 16)

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.c, self.f * 2, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.f * 2, self.f * 4, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 4),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.f * 4, self.f * 8, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 8),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.f * 8, self.f * 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(self.f * 16),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.fc = nn.Linear(self.fc_dim, 1)

    def forward(self, x):
        x = self.discriminator(x)
        x = x.view(-1, self.fc_dim)
        x = F.sigmoid(self.fc(x))
        return x
