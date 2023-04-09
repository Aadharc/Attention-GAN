import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, num_detections
import torch.nn as nn
import torch.optim as optim
import config
# from dataset import MapDataset
from dataset import CustomDataSet
from Generator import Generator
from Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ssim import SSIM
from Attention_gen import convblock, CrossAttention


def train_fn(disc_ir, disc_vis, gen, cross_attn, convblock, loader, opt_disc_ir, opt_disc_vis, opt_gen, l1_loss, bce, KL):
    loop = tqdm(loader, leave = True)
    for idx, batch in enumerate(loop):
        x, y = batch
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        x_feat = convblock(x)
        y_feat = convblock(y)

        attn1, attn2 = cross_attn(x_feat, y_feat)
        # print(f"attn1 {attn1.shape}")
        
        y_fake = gen(x, y, attn1, attn2)
        # print(f"yfake {y_fake.shape}")
        D_real_ir = disc_ir(y, y, attn2)
        D_real_loss_ir = bce(D_real_ir, torch.ones_like(D_real_ir))
        D_fake_ir = disc_ir(y, y_fake.detach(), attn2)
        D_fake_loss_ir = bce(D_fake_ir, torch.zeros_like(D_fake_ir))
        D_loss_ir = (D_real_loss_ir + D_fake_loss_ir) / 2

        D_real_vis = disc_vis(x, x, attn1)
        D_real_loss_vis = bce(D_real_vis, torch.ones_like(D_real_vis))
        D_fake_vis = disc_vis(x, y_fake.detach(), attn1)
        D_fake_loss_vis = bce(D_fake_vis, torch.zeros_like(D_fake_vis))
        D_loss_vis = (D_real_loss_vis + D_fake_loss_vis) / 2

        disc_ir.zero_grad()
        disc_vis.zero_grad()
        D_loss_ir.backward(retain_graph=True)
        D_loss_vis.backward(retain_graph=True)
        opt_disc_ir.step()
        opt_disc_vis.step()

        D_fake_ir = disc_ir(y, y_fake, attn2)
        D_fake_vis = disc_vis(x, y_fake, attn1)

        G_fake_loss_ir = bce(D_fake_ir, torch.ones_like(D_fake_ir))
        G_fake_loss_vis = bce(D_fake_vis, torch.ones_like(D_fake_vis))
        L1 = (l1_loss(y_fake, y) + l1_loss(y_fake, x))  * config.L1_LAMBDA
        G_loss = G_fake_loss_ir + G_fake_loss_vis + L1 + (KL(y_fake.clone(),y.clone()) + KL(y_fake.clone(),x.clone())) * config.BETA

        opt_gen.zero_grad()
        G_loss.backward(retain_graph=True)
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real_ir=torch.sigmoid(D_real_ir).mean().item(),
                D_fake_ir=torch.sigmoid(D_fake_ir).mean().item(),
                D_real_vis=torch.sigmoid(D_real_vis).mean().item(),
                D_fake_vis=torch.sigmoid(D_fake_vis).mean().item(),
            )

def main():
    conv_block = convblock(in_chan= 3, features = 32)
    cross_attn = CrossAttention(32)
    disc_ir = Discriminator(in_channels=3).to(config.DEVICE)
    disc_vis = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=8, features=64).to(config.DEVICE)
    opt_disc_ir = optim.Adam(disc_ir.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_disc_vis = optim.Adam(disc_vis.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    transform = transforms.Compose([transforms.Resize((64,128),transforms.InterpolationMode.BILINEAR), transforms.ToTensor()])

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    ssim = SSIM()
    KL = nn.KLDivLoss()

    train_dataset = CustomDataSet(config.TRAIN_DIR_VIS, config.TRAIN_DIR_IR,  transform= transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    val_dataset = CustomDataSet(config.VAL_DIR_VIS, config.VAL_DIR_IR,  transform)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
                disc_ir, disc_vis, gen, cross_attn, conv_block, train_loader, opt_disc_ir, opt_disc_vis, opt_gen, L1_LOSS, BCE, KL)

        if config.SAVE_MODEL and epoch % 2 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc_ir, opt_disc_ir, filename=config.CHECKPOINT_DISC_IR)
            save_checkpoint(disc_vis, opt_disc_vis, filename=config.CHECKPOINT_DISC_VIS)
            save_some_examples(conv_block, cross_attn, gen, val_loader, epoch, folder="evaluation")


if __name__ == "__main__":
    main()



