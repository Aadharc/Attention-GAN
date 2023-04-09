import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VisionTransformer

class block(nn.Module):
    def __init__(self, in_chan , out_chan, down = True, act = "prelu", use_dropout = False):
        super(block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias = False, padding_mode= 'reflect')
            if down
            else nn.ConvTranspose2d(in_chan, out_chan, 4, 2, 1, bias = False),
            nn.InstanceNorm2d(out_chan, affine = True),
            nn.PReLU() if act == "prelu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


# creating block without downsampling the height and widths
class block1(nn.Module):
    def __init__(self, in_chan , out_chan, act = "prelu", use_dropout = False):
        super(block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3, 1, 1, bias = False, padding_mode= 'reflect'),
            nn.InstanceNorm2d(out_chan, affine = True),
            nn.PReLU() if act == "prelu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x
    
class convblock(nn.Module):
    def __init__(self, in_chan = 3, features = 32):
        super().__init__()
        self.down1 = block1(in_chan, features, act = "prelu", use_dropout= False)
        self.down2 = block1(features, features, act = "prelu", use_dropout= False)
        self.down3 = block1(features, features, act = "prelu", use_dropout= False)
        self.down4 = block1(features, features, act = "prelu", use_dropout= False)
    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x
    


class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.in_channels = in_channels

        # Convolutional layers for extracting query, key, and value features
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Convolutional layer for combining the attended features
        self.combine_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def forward(self, x1, x2):
        # Compute the query, key, and value features from x1 and x2
        batch_size, _, h, w = x1.size()
        query1 = self.query_conv(x1).view(batch_size, -1, h*w).permute(0, 2, 1)
        key1 = self.key_conv(x1).view(batch_size, -1, h*w)
        value1 = self.value_conv(x1).view(batch_size, -1, h*w)
        query2 = self.query_conv(x2).view(batch_size, -1, h*w).permute(0, 2, 1)
        key2 = self.key_conv(x2).view(batch_size, -1, h*w)
        value2 = self.value_conv(x2).view(batch_size, -1, h*w)

        # Compute the attention map and attended features
        attn1 = torch.bmm(query1, key1)
        attn1 = F.softmax(attn1, dim=2)
        attended1 = torch.bmm(value2, attn1.permute(0, 2, 1)).view(batch_size, self.in_channels, h, w)

        # Combine the attended features from x1 and x2
        combined1 = self.combine_conv(torch.cat((x1, attended1), dim=1))

        # Compute the attention map and attended features
        attn2 = torch.bmm(query2, key2)
        attn2 = F.softmax(attn2, dim=2)
        attended2 = torch.bmm(value1, attn2.permute(0, 2, 1)).view(batch_size, self.in_channels, h, w)

        # Combine the attended features from x1 and x2
        combined2 = self.combine_conv(torch.cat((x1, attended2), dim=1))

        return (torch.abs(combined1).sum(dim=1).unsqueeze(1), torch.abs(combined2).sum(dim = 1).unsqueeze(1))
        # return combined1

class convup(nn.Module):
    def __init__(self, in_chan = 3, features = 32):
        super().__init__()
        # self.enc = convblock(in_chan= in_chan, features = 32)
        self.up1 = block( 2 + 32 + 32, features * 2, down = False, act = "prelu", use_dropout= False)
        self.up2 = block(features * 2, features * 4, down = False, act = "prelu", use_dropout= False)
        self.up3 = block(features * 4, features, down = False, act = "prelu", use_dropout= False)
        self.up4 = block(features , in_chan, down = False, act = "prelu", use_dropout= False)
    def forward(self, x, y, attn):
        # d1 = self.enc.down1(x)
        # d2 = self.enc.down2(d1)
        # d3 = self.enc.down3(d2)
        # d4 = self.enc.down4(d3)
        u1 = self.up1(torch.cat((attn,x,y), 1))
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)
        return u4


class Generator(nn.Module):
    def __init__(self, in_chan):
        super().__init__()
        self.enc = convblock(in_chan= in_chan, features = 32)
        self.attn = CrossAttention(32)
        # self.dec = convup(in_chan= 3, features = 32)

    def forward(self, x, y):
        # print(x.shape)
        enc1 = self.enc(x)
        enc2 = self.enc(y)
        # print("enc", enc1.shape)
        attn = self.attn(enc1, enc2)[0]
        print(attn.shape)
        # dec = self.dec(attn, enc1, enc2)
        return attn

def test():
    x = torch.randn((1, 3, 28, 28))
    y = torch.randn((1, 3, 28, 28))
    # model = CrossAttention(in_channels=32)
    model = Generator(3)
    preds = model(x,y)
    print(preds.shape)


if __name__ == "__main__":
    test()