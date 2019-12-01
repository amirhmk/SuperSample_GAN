import torch.optim as optim
import torch
import torch.nn as nn
from model import Generator, Discriminator

def create_model(opts):
    """Builds the generators and discriminators.
    """
    G_XtoY = CycleGenerator(conv_dim=opts.g_conv_dim,
                            init_zero_weights=opts.init_zero_weights)
    G_YtoX = CycleGenerator(conv_dim=opts.g_conv_dim,
                            init_zero_weights=opts.init_zero_weights)
    D_X = DCDiscriminator(conv_dim=opts.d_conv_dim)
    D_Y = DCDiscriminator(conv_dim=opts.d_conv_dim)

    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    if torch.cuda.is_available():
        G_XtoY.cuda()
        G_YtoX.cuda()
        D_X.cuda()
        D_Y.cuda()
        print('Models moved to GPU.')

    return G_XtoY, G_YtoX, D_X, D_Y


def checkpoint(iteration, G_XtoY, G_YtoX, D_X, D_Y, opts):
    """Saves the parameters of both generators G_YtoX, G_XtoY and discriminators D_X, D_Y.
    """
    G_XtoY_path = os.path.join(opts.checkpoint_dir, 'G_XtoY.pkl')
    G_YtoX_path = os.path.join(opts.checkpoint_dir, 'G_YtoX.pkl')
    D_X_path = os.path.join(opts.checkpoint_dir, 'D_X.pkl')
    D_Y_path = os.path.join(opts.checkpoint_dir, 'D_Y.pkl')
    torch.save(G_XtoY.state_dict(), G_XtoY_path)
    torch.save(G_YtoX.state_dict(), G_YtoX_path)
    torch.save(D_X.state_dict(), D_X_path)
    torch.save(D_Y.state_dict(), D_Y_path)


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts):
    """Runs the training loop.
        * Saves checkpoint every opts.checkpoint_every iterations
        * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    if opts.load:
        G_XtoY, G_YtoX, D_X, D_Y = load_checkpoint(opts)
    else:
        G_XtoY, G_YtoX, D_X, D_Y = create_model(opts)

    g_params = list(G_XtoY.parameters()) + \
        list(G_YtoX.parameters())  # Get generator parameters
    # Get discriminator parameters
    d_params = list(D_X.parameters()) + list(D_Y.parameters())

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
