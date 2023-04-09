import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.InstanceNorm2d(out_channels, affine= True),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x



class Generator(nn.Module):
    def __init__(self, in_channels=8, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 3, 1, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, y, attn1, attn2):
        d = torch.cat([x,y, attn1, attn2], dim=1)
        # print("d shape", d.shape)
        d1 = self.initial_down(d)
        # print("d1 shape", d1.shape)
        d2 = self.down1(d1)
        # print("d2 shape", d2.shape)
        d3 = self.down2(d2)
        # print("d3 shape", d3.shape)
        d4 = self.down3(d3)
        # print("d4 shape", d4.shape)
        d5 = self.down4(d4)
        # print("d5 shape", d5.shape)
        d6 = self.down5(d5)
        # print("d6 shape", d6.shape)
        d7 = self.down6(d6)
        # print("d7 shape", d7.shape)
        bottleneck = self.bottleneck(d7)
        # print("bottle shape", bottleneck.shape)
        up1 = self.up1(bottleneck)
        # print("up1 shape", up1.shape)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    x = torch.randn((1, 3, 64, 128))
    y = torch.randn((1, 3, 64, 128))
    a1 = torch.randn((1, 1, 64, 128))
    a2= torch.randn((1, 1, 64, 128))
    model = Generator(in_channels=8, features=64)
    preds = model(x,y, a1, a2)
    print(preds.shape)


if __name__ == "__main__":
    test()
        