import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetSingleHead(nn.Module):
    def __init__(self, in_channels, out_channels=4):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        return out


class UNetMultiHead(nn.Module):
    """
    Shared encoder, two independent decoders
    """
    def __init__(self, in_channels):
        super().__init__()

        # Shared encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)

        # BPD decoder
        self.up3_bpd = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3_bpd = ConvBlock(256, 128)
        self.up2_bpd = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2_bpd = ConvBlock(128, 64)
        self.up1_bpd = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1_bpd = ConvBlock(64, 32)
        self.out_bpd = nn.Conv2d(32, 2, 1)

        # OFD decoder
        self.up3_ofd = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3_ofd = ConvBlock(256, 128)
        self.up2_ofd = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2_ofd = ConvBlock(128, 64)
        self.up1_ofd = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1_ofd = ConvBlock(64, 32)
        self.out_ofd = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # BPD Decoder
        d3_bpd = self.up3_bpd(b)
        d3_bpd = self.dec3_bpd(torch.cat([d3_bpd, e3], dim=1))
        d2_bpd = self.up2_bpd(d3_bpd)
        d2_bpd = self.dec2_bpd(torch.cat([d2_bpd, e2], dim=1))
        d1_bpd = self.up1_bpd(d2_bpd)
        d1_bpd = self.dec1_bpd(torch.cat([d1_bpd, e1], dim=1))
        out_bpd = self.out_bpd(d1_bpd)

        # OFD Decoder
        d3_ofd = self.up3_ofd(b)
        d3_ofd = self.dec3_ofd(torch.cat([d3_ofd, e3], dim=1))
        d2_ofd = self.up2_ofd(d3_ofd)
        d2_ofd = self.dec2_ofd(torch.cat([d2_ofd, e2], dim=1))
        d1_ofd = self.up1_ofd(d2_ofd)
        d1_ofd = self.dec1_ofd(torch.cat([d1_ofd, e1], dim=1))
        out_ofd = self.out_ofd(d1_ofd)

        return torch.cat([out_bpd, out_ofd], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        # BPD
        d3b = self.dec3_bpd(torch.cat([self.up3_bpd(b), e3], 1))
        d2b = self.dec2_bpd(torch.cat([self.up2_bpd(d3b), e2], 1))
        d1b = self.dec1_bpd(torch.cat([self.up1_bpd(d2b), e1], 1))
        bpd_hm = self.out_bpd(d1b)

        # OFD
        d3o = self.dec3_ofd(torch.cat([self.up3_ofd(b), e3], 1))
        d2o = self.dec2_ofd(torch.cat([self.up2_ofd(d3o), e2], 1))
        d1o = self.dec1_ofd(torch.cat([self.up1_ofd(d2o), e1], 1))
        ofd_hm = self.out_ofd(d1o)

        # Concatenate in canonical order
        out = torch.cat([bpd_hm, ofd_hm], dim=1)
        return out

class GeometryDiameterNet(nn.Module):
    """
    Predicts:
    - center heatmap (1)
    - direction vector (2)
    - BPD length (1)
    - OFD length (1)
    Total output channels = 5
    """
    def __init__(self, in_channels):
        super().__init__()

        self.encoder = UNetSingleHead(in_channels, out_channels=64)

        self.center_head = nn.Conv2d(64, 1, 1)
        self.dir_head = nn.Conv2d(64, 2, 1)
        self.len_head = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        feat = self.encoder(x)
        center = self.center_head(feat)
        direction = self.dir_head(feat)
        lengths = self.len_head(feat)

        return {
            "center": center,
            "direction": direction,
            "lengths": lengths
        }

class HighResHeatmapNet(nn.Module):
    """
    Maintains higher resolution features longer
    """
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 64)

        self.pool = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(64, 128)
        self.conv5 = ConvBlock(128, 128)

        self.up = nn.ConvTranspose2d(128, 64, 2, 2)
        self.out = nn.Conv2d(64, 4, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        p = self.pool(x3)
        p = self.conv4(p)
        p = self.conv5(p)

        up = self.up(p)
        out = self.out(up)
        return out
