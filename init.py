import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torch.utils.data import DataLoader, Dataset  # Gives easier dataset managment and creates mini batches
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data.distributed import DistributedSampler

from models import Discriminator, Generator


def run():
    # Hyperparameters
    lr = 0.0002
    batch_size = 64
    image_size = 64
    channels_img = 3
    channels_noise = 128
    num_epochs = 4096

    # For how many channels Generator and Discriminator should use
    features_d = 64
    features_g = 64

    moose_images = torchvision.datasets.ImageFolder('dataset', transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    data_loader = torch.utils.data.DataLoader(moose_images,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device ", device)

    # Create discriminator and generator
    netD = Discriminator(channels_img, features_d).to(device)
    netG = Generator(channels_noise, channels_img, features_g).to(device)
    netD.load_state_dict(torch.load('d.pth'))
    netG.load_state_dict(torch.load('g.pth'))

    # Setup Optimizer for G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))

    netG.train()
    netD.train()

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(image_size, channels_noise, 1, 1).to(device)
    writer_real = SummaryWriter('runs/GAN_MOOSE/test_real')

    print("Starting Training...")

    real_label = 1
    fake_label = 0

    for epoch in range(num_epochs):
        for i, data in enumerate(data_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, channels_noise, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Print losses ocassionally and print to tensorboard
            if i % batch_size == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch {i}/{len(data_loader)} \
                      Loss D: {errD:.4f}, loss G: {errG:.4f} D(x): {D_x:.4f}')

                with torch.no_grad():
                    fake = netG(fixed_noise)

                    img_grid_real = torchvision.utils.make_grid(real_cpu[:32], normalize=True)
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    writer_real.add_image('Mnist Real Images', img_grid_real, epoch)
                    writer_real.add_image('Mnist Fake Images', img_grid_fake, epoch)
                    writer_real.add_scalar('Loss D', errD, epoch)
                    writer_real.add_scalar('Loss G', errG, epoch)
                    writer_real.add_scalar('D(x)', D_x, epoch)

    torch.save(netD.state_dict(), 'd.pth')
    torch.save(netG.state_dict(), 'g.pth')


if __name__ == '__main__':
    run()
