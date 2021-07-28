import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
from networks.unet.unet import UNet

from tqdm import tqdm

import cv2
import imageio

class args():
    
    # Model setting
    checkpoint = 'weights/sample.pth'

    # Dataset setting
    channels = 3
    size = 256
    videos_dir = 'datasets/Test015'
    time_steps = 5

    # For GPU training
    gpu = 0 # None


def evaluate():

    generator = UNet(in_channels=args.channels * (args.time_steps - 1), out_channels=args.channels)
    generator.load_state_dict(torch.load(args.checkpoint)['generator'])
    print(f'The pre-trained generator has been loaded from ', args.checkpoint)

    testloader = DataLoader(dataset=TestDataset(channels=args.channels, size=args.size, videos_dir=args.videos_dir, time_steps=args.time_steps), batch_size=1, shuffle=False, num_workers=0)

    if torch.cuda.is_available() and args.gpu is not None:
        use_cuda = True
        torch.cuda.set_device(args.gpu)

        generator = generator.cuda()
    else:
        use_cuda = False
        print('using CPU, this will be slow')

    heatmaps = []
    originals = []
    with torch.no_grad():
        for i, datas in enumerate(tqdm(testloader)):
            frames, o_frames = datas[0], datas[1]
            generator.eval()
            inputs = frames[:, :args.channels * (args.time_steps - 1), :, :]
            target = frames[:, args.channels * (args.time_steps - 1):, :, :]

            if use_cuda:
                inputs = inputs.cuda()

            generated = generator(inputs)

            if use_cuda:
                generated = generated.cpu()

            diffmap = torch.sum(torch.abs(generated - target).squeeze(), 0)
            diffmap -= diffmap.min() 
            diffmap /= diffmap.max()
            diffmap *= 255
            diffmap = diffmap.detach().numpy().astype('uint8')
            
            heatmap = cv2.applyColorMap(diffmap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmaps.append(heatmap)

            original = o_frames[-1].squeeze().detach().numpy().astype('uint8')
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            originals.append(original)

    imageio.mimsave(f'results/heatmap.gif', heatmaps, fps=30) 
    imageio.mimsave(f'results/original.gif', originals, fps=30) 

if __name__ == "__main__":
    evaluate()