import torch.nn as nn

from .libResnet34_unet import BasicBlock, InputBlock, DecoderBlock, SegmentationHead


class Resnet34Unet(nn.Module):
    def __init__(self, in_channels):
        super(Resnet34Unet, self).__init__()
        self.input_block = InputBlock(in_channels)
        self.enc_conv2 = self._make_resnet_layer(
            BasicBlock, 64, 64, num_blocks=3, stride=1
        )
        self.enc_conv3 = self._make_resnet_layer(
            BasicBlock, 64, 128, num_blocks=4, stride=2
        )
        self.enc_conv4 = self._make_resnet_layer(
            BasicBlock, 128, 256, num_blocks=6, stride=2
        )
        self.enc_conv5 = self._make_resnet_layer(
            BasicBlock, 256, 512, num_blocks=3, stride=2
        )

        # Decoder
        self.dec5 = DecoderBlock(512, 256, 256)
        self.dec4 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64, 64)
        self.dec2 = DecoderBlock(64, 0, 32)
        self.dec1 = DecoderBlock(32, 0, 16)
        self.seg = SegmentationHead(in_channels=16, out_channels=1)

    def _make_resnet_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = [block(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.input_block(x)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        enc4 = self.enc_conv4(enc3)
        enc5 = self.enc_conv5(enc4)

        dec5 = self.dec5(enc5, enc4)
        dec4 = self.dec4(dec5, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3)
        dec1 = self.dec1(dec2)
        seg = self.seg(dec1)
        return seg
