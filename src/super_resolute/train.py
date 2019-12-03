import os
import torch.optim as optim
import torch
import torch.nn as nn
from model import Generator, Discriminator
import utils
from data_loader import get_super_resolute_data_loader
import argparse



def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

def print_models(Generator, Discriminator):
    """Prints model information for the generators and discriminators.
    """
    print("                 Generator             ")
    print("---------------------------------------")
    print(Generator)
    print("---------------------------------------")

    print("             Discriminator             ")
    print("---------------------------------------")
    print(Discriminator)
    print("---------------------------------------")

def create_model(opts):
    """Builds the generators and discriminators.
    """
    G = Generator(conv_dim=opts.g_conv_dim, scale_factor=opts.scale_factor)
    D = Discriminator(conv_dim=opts.d_conv_dim)

    print_models(G, D)

    if torch.cuda.is_available():
        D.cuda()
        D.cuda()
        print('Models moved to GPU.')

    return G.float(), D


def checkpoint(iteration, G, D, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G_.pth')
    D_path = os.path.join(opts.checkpoint_dir, 'D_.pth')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def training_loop(dataloader_train, dataloader_valid, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    if opts.load:
        G, D = load_checkpoint(opts)
    else:
        G, D = create_model(opts)

    g_params = list(G.parameters()) # Get generator parameters
    # Get discriminator parameters
    d_params = list(D.parameters())

    # Create optimizers for the generator and discriminator
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    # Create loss functions for generator and discriminator
    d_loss_criterion = nn.MSELoss()
    g_loss_criterion = nn.BCELoss()

    train_iter = iter(dataloader_train)

    # valid_iter = iter(dataloader_valid)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    # fixed_X = utils.to_var(test_iter_X.next()[0])
    # fixed_Y = utils.to_var(test_iter_Y.next()[0])

    iter_per_epoch = len(train_iter)
    mean_discriminator_loss = 0
    mean_generator_loss = 0

    for iteration in range(1, opts.train_iters+1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            train_iter = iter(dataloader_train)
        images_HR, images_LR = train_iter.next()
        # print("images", images_LR.shape, images_HR.shape)
        images_HR, images_LR = utils.to_var(images_HR).float().squeeze(), utils.to_var(images_LR).float().squeeze()
        # print("images", images_LR.shape, images_HR.shape, images_LR.type(), images_HR.type())
        # print("talking about u", images_LR.size(0))
        # valid, fake = utils.generate_random_groundtruth(opts)


        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================
        # Train with real images
        d_optimizer.zero_grad()
        # 1. Compute the discriminator losses on real images


        # Train with fake images

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_img = G(images_LR)

        d_optimizer.zero_grad()
        # 3. Compute the loss for D_X
        real_out = D(images_HR)
        fake_out = D(fake_img)
        print(fake_out.shape, fake_out.size(0))

        print(real_out.shape, images_HR.shape)

        d_fake_loss = d_loss_criterion(real_out, fake_out)
        d_fake_loss.backward(retain_graph=True)
        mean_discriminator_loss += d_fake_loss.item()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================
        g_optimizer.zero_grad()

        g_loss = g_loss_criterion(real_out, valid)
        g_loss.backward()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_img = G(images_LR)
        fake_out = D(fake_img).mean()

        # 2. Compute the generator loss
        mean_generator_loss += g_loss.item()
        print("mean_generator_loss", mean_generator_loss)
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                  'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                      iteration, opts.train_iters, d_real_loss.data[0], D_Y_loss.data[0],
                      D_X_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G, D, opts)


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    train_dloader, test_dloader = get_super_resolute_data_loader(opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(train_dloader, test_dloader, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=100,
                        help='The side length N to convert images to NxN.')
    parser.add_argument('--g_conv_dim', type=int, default=8)
    parser.add_argument('--d_conv_dim', type=int, default=8)
    parser.add_argument('--use_cycle_consistency_loss', action='store_true', default=False,
                        help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--init_zero_weights', action='store_true', default=False,
                        help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')
    parser.add_argument('--scale_factor', type=int, default=4)


    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=100,
                        help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--training_data_path', type=str, help='Path to training data',
                        default='data/DIV2K_train_HR')
    parser.add_argument('--validation_data_path', type=str, help='Path to training data',
                        default='data/DIV2K_valid_HR')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int, default=10)
    # parser.add_argument('--sample_every', type=int, default=100)
    parser.add_argument('--checkpoint_every', type=int, default=800)

    return parser


if __name__ == '__main__':

    parser = create_parser()
    opts = parser.parse_args()

    if opts.use_cycle_consistency_loss:
        opts.sample_dir = 'samples_cyclegan_cycle'

    if opts.load:
        opts.sample_dir = '{}_pretrained'.format(opts.sample_dir)
        opts.sample_every = 20

    print_opts(opts)
    main(opts)
