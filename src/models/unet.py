import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, num_classes):
        from .libUnet import DoubleConv, Down, Up, Out
        super(Unet, self).__init__()
        self.enc_1 = DoubleConv(3, 64)
        self.enc_2 = Down(64, 128)
        self.enc_3 = Down(128, 256)
        self.enc_4 = Down(256, 512)
        self.enc_5 = Down(512, 1024)
        self.dec_5 = Up(1024, 512)
        self.dec_4 = Up(512, 256)
        self.dec_3 = Up(256, 128)
        self.dec_2 = Up(128, 64)
        self.out = Out(64, num_classes)

    def forward(self, x):
        x1 = self.enc_1(x)
        x2 = self.enc_2(x1)
        x3 = self.enc_3(x2)
        x4 = self.enc_4(x3)
        x5 = self.enc_5(x4)
        x = self.dec_5(x5, x4)
        x = self.dec_4(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_2(x, x1)
        x = self.out(x)
        return x
