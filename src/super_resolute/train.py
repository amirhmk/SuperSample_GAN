import os
import torch.optim as optim
import torch
import torch.nn as nn
from model import Generator, Discriminator


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

    return G, D


def checkpoint(iteration, G, D, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_path = os.path.join(opts.checkpoint_dir, 'G_.pth')
    D_path = os.path.join(opts.checkpoint_dir, 'D_.pth')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
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

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr, [opts.beta1, opts.beta2])

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = utils.to_var(test_iter_X.next()[0])
    fixed_Y = utils.to_var(test_iter_Y.next()[0])

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    for iteration in range(1, opts.train_iters+1):

        # Reset data_iter for each epoch
        if iteration % iter_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, labels_X = iter_X.next()
        images_X, labels_X = utils.to_var(
            images_X), utils.to_var(labels_X).long().squeeze()

        images_Y, labels_Y = iter_Y.next()
        images_Y, labels_Y = utils.to_var(
            images_Y), utils.to_var(labels_Y).long().squeeze()

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        #########################################
        ##             FILL THIS IN            ##
        #########################################

        # Train with real images
        d_optimizer.zero_grad()
        # print("HELLO", opts)
        batch_size = opts.batch_size
        # 1. Compute the discriminator losses on real images
        D_X_loss = np.sum(np.power(D_X(images_X) - 1, 2)) / batch_size
        D_Y_loss = np.sum(np.power(D_Y(images_Y) - 1, 2)) / batch_size

        d_real_loss = D_X_loss + D_Y_loss
        d_real_loss.backward()
        d_optimizer.step()

        # Train with fake images
        d_optimizer.zero_grad()

        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 3. Compute the loss for D_X
        D_X_loss = np.sum(np.power(D_X(fake_X), 2)) / batch_size

        # 4. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 5. Compute the loss for D_Y
        D_Y_loss = np.sum(np.power(D_Y(fake_Y), 2)) / batch_size

        d_fake_loss = D_X_loss + D_Y_loss
        d_fake_loss.backward()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        #########################################
        ##    FILL THIS IN: Y--X-->Y CYCLE     ##
        #########################################
        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        g_loss = np.sum(np.power(D_X(fake_X) - 1, 2)) / batch_size

        if opts.use_cycle_consistency_loss:
            reconstructed_Y = G_XtoY(fake_X)
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.sum(
                (images_Y - reconstructed_Y) ** 2) / batch_size
            g_loss += cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()

        #########################################
        ##    FILL THIS IN: X--Y-->X CYCLE     ##
        #########################################

        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain Y
        g_loss = np.sum(np.power(D_Y(fake_Y) - 1, 2)) / batch_size

        if opts.use_cycle_consistency_loss:
            reconstructed_X = G_YtoX(fake_Y)
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.sum(
                (images_X - reconstructed_X) ** 2) / batch_size
            g_loss += cycle_consistency_loss

        g_loss.backward()
        g_optimizer.step()

        # Print the log info
        if iteration % opts.log_step == 0:
            print('Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | '
                  'd_fake_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                      iteration, opts.train_iters, d_real_loss.data[0], D_Y_loss.data[0],
                      D_X_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

        # Save the generated samples
        if iteration % opts.sample_every == 0:
            save_samples(iteration, fixed_Y, fixed_X, G_YtoX, G_XtoY, opts)

        # Save the model parameters
        if iteration % opts.checkpoint_every == 0:
            checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts)


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(
        emoji_type=opts.X, opts=opts)
    dataloader_Y, test_dataloader_Y = get_emoji_loader(
        emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    utils.create_dir(opts.checkpoint_dir)
    utils.create_dir(opts.sample_dir)

    # Start training
    training_loop(dataloader_X, dataloader_Y,
                  test_dataloader_X, test_dataloader_Y, opts)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=32,
                        help='The side length N to convert images to NxN.')
    parser.add_argument('--g_conv_dim', type=int, default=32)
    parser.add_argument('--d_conv_dim', type=int, default=32)
    parser.add_argument('--use_cycle_consistency_loss', action='store_true', default=False,
                        help='Choose whether to include the cycle consistency term in the loss.')
    parser.add_argument('--init_zero_weights', action='store_true', default=False,
                        help='Choose whether to initialize the generator conv weights to 0 (implements the identity function).')

    # Training hyper-parameters
    parser.add_argument('--train_iters', type=int, default=600,
                        help='The number of training iterations to run (you can Ctrl-C out earlier if you want).')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Data sources
    parser.add_argument('--X', type=str, default='Apple', choices=[
                        'Apple', 'Windows'], help='Choose the type of images for domain X.')
    parser.add_argument('--Y', type=str, default='Windows', choices=[
                        'Apple', 'Windows'], help='Choose the type of images for domain Y.')

    # Saving directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints_cyclegan')
    parser.add_argument('--sample_dir', type=str, default='samples_cyclegan')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=100)
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