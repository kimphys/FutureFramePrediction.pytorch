import os
import sys

from torch.utils.data import DataLoader

from utils import *
from loss import *
from dataset import SequenceDataset
from networks.unet.unet import UNet
from networks.flownet.models import FlowNet2SD
from networks.discriminator.models import PixelDiscriminator

from tqdm import tqdm

class args():
    
    # training args
    epochs = 600 # "number of training epochs, default is 2"
    save_per_epoch = 5
    batch_size = 6 # "batch size for training/testing, default is 4"
    pretrained = False
    lr_init = 1e-4
    lr_weight_decay = 1e-5
    save_model_dir = "./weights/" #"path to folder where trained model with checkpoints will be saved."
    num_workers = 0
    resume = False

    # generator setting
    g_lr_init = 0.0002
    
    # discriminator setting
    d_lr_init = 0.00002

    # optical flow setting
    flownet_pretrained = 'pretrained/FlowNet2-SD.pth'

    # Dataset setting
    channels = 3
    size = 256
    videos_dir = 'D:\\project\\anomalydetection\\UCSD_Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train\\210716\\'
    time_steps = 10

    # For GPU training
    gpu = 0 # None

def train():
    
    generator = UNet(in_channels=args.channels * (args.time_steps - 1), out_channels=args.channels)
    discriminator = PixelDiscriminator(input_nc=args.channels)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.g_lr_init)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr_init)

    opticalflow = FlowNet2SD()
    opticalflow.load_state_dict(torch.load(args.flownet_pretrained)['state_dict'])

    intensity_loss = IntensityLoss()
    gradient_loss = GradientLoss(args.channels)
    flow_loss = FlowLoss()
    adversarial_loss = GeneratorAdversarialLoss()
    discriminator_loss = DiscriminatorAdversarialLoss()


    if args.resume:
        generator.load_state_dict(torch.load(args.resume)['generator'])
        discriminator.load_state_dict(torch.load(args.resume)['discriminator'])
        optimizer_G.load_state_dict(torch.load(args.resume)['optimizer_G'])
        optimizer_D.load_state_dict(torch.load(args.resume)['optimizer_D'])
        print(f'Pre-trained generator and discriminator have been loaded.\n')

    if torch.cuda.is_available() and args.gpu is not None:
        use_cuda = True
        torch.cuda.set_device(args.gpu)

        generator = generator.cuda()
        discriminator = discriminator.cuda()
        opticalflow = opticalflow.cuda()
    else:
        use_cuda = False
        print('using CPU, this will be slow')

    trainloader = DataLoader(dataset=SequenceDataset(channels=args.channels, size=args.size, videos_dir=args.videos_dir, time_steps=args.time_steps), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with torch.set_grad_enabled(True):
        for ep in range(args.epochs):
            pbar = tqdm(trainloader)

            g_loss_sum = 0
            d_loss_sum = 0

            for i, clips in enumerate(pbar):
                for frames in clips:
                    generator.train()
                    discriminator.train()
                    inputs = frames[:, :args.channels * (args.time_steps - 1), :, :]
                    last = frames[:, args.channels * (args.time_steps - 2):args.channels * (args.time_steps - 1), :, :]
                    target = frames[:, args.channels * (args.time_steps - 1):, :, :]
                    
                    if use_cuda:
                        inputs = inputs.cuda()
                        last = last.cuda()
                        target = target.cuda()

                    generated = generator(inputs)

                    gt_flow_input = torch.cat([last.unsqueeze(2), target.unsqueeze(2)], 2)
                    pred_flow_input = torch.cat([last.unsqueeze(2), generated.unsqueeze(2)], 2)

                    flow_gt = (opticalflow(gt_flow_input * 255.) / 255.).detach()
                    flow_pred = (opticalflow(pred_flow_input * 255.) / 255.).detach()

                    d_t = discriminator(target)
                    d_g = discriminator(generated.detach())

                    if use_cuda:
                        generated = generated.cpu()
                        target = target.cpu()
                        flow_pred = flow_pred.cpu()
                        flow_gt = flow_gt.cpu()
                        d_t = d_t.cpu()
                        d_g = d_g.cpu()

                    g_loss = intensity_loss(generated, target) + \
                                gradient_loss(generated, target) + \
                                2 * flow_loss(flow_pred, flow_gt) + \
                                0.05 * adversarial_loss(d_g)

                    d_loss = discriminator_loss(d_t, d_g)

                    d_loss.backward(retain_graph=True)
                    g_loss.backward(retain_graph=True)

                    optimizer_D.zero_grad()
                    optimizer_G.zero_grad()
                    optimizer_D.step()
                    optimizer_G.step()

                    d_loss_sum += d_loss.item()
                    g_loss_sum += g_loss.item()

            g_loss_mean = g_loss_sum / (len(clips) * len(pbar))
            d_loss_mean = d_loss_sum / (len(clips) * len(pbar))
            print('G Loss: ', g_loss_mean)
            print('D Loss: ', d_loss_mean)

            if (ep + 1) % args.save_per_epoch == 0:
                model_dict = {'generator': generator.state_dict(), 'optimizer_G': optimizer_G.state_dict(),
                            'discriminator': discriminator.state_dict(), 'optimizer_D': optimizer_D.state_dict()}
                torch.save(model_dict, os.path.join(args.save_model_dir, f'ckpt_{ep + 1}_{g_loss_mean}_{d_loss_mean}.pth'))

        

if __name__ == "__main__":
    train()