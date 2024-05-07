import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
import argparse
import numpy as np
import time
#from tensorboardX import SummaryWriter
from deep_ion_registration.networks.primal_dual_straight import PrimalDualStraight, LoggerPrimalDual
from deep_ion_registration.dataset import IonDataset, load_system_matrices

# class DataLoaderX(DataLoader):
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())

def train():
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(description="Train a network for ...")
    # add your arguments here
    opt = parser.parse_args()

    # set flags / seeds
    torch.backends.cudnn.benchmark = True
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.is_available()

    # add code for datasets
    train_dataset = IonDataset(opt.train_data_dir)
    train_data_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_dataset = IonDataset(opt.test_data_dir)
    test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # instantiate network
    system_matrix = load_system_matrices(opt.system_matrix_path)
    system_matrix_normalized = load_system_matrices(opt.system_matrix_normalized_path)
    net = PrimalDualStraight(system_matrix=system_matrix, system_matrix_normalized=system_matrix_normalized)

    # create losses
    criterion = nn.MSELoss()

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    # create optimizers
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr)

    # load checkpoint if needed/ wanted
    start_epoch = 0
    if opt.resume:
        checkpoint = torch.load(opt.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    # if we want to run experiment on multiple GPUs we move the models there
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # typically we use tensorboardX to keep track of experiments
    # writer = SummaryWriter(opt.log_dir)

    # now we start the main loop
    for epoch in range(start_epoch, opt.epochs):
        # set models to train mode
        net.train()

        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        start_time = time.time()

        # for loop going through dataset
        for i, data in pbar:
            # data preparation
            xray_ct, projection_angle, dual_initial_guess, transformed_ionct = data
            if use_cuda:
                xray_ct = xray_ct.cuda()
                projection_angle = projection_angle.cuda()
                dual_initial_guess = dual_initial_guess.cuda()
                transformed_ionct = transformed_ionct.cuda()

            # forward and backward pass
            optim.zero_grad()
            output = net(xray_ct, projection_angle, dual_initial_guess)
            loss = criterion(output, transformed_ionct)
            loss.backward()
            optim.step()

            # udpate tensorboardX
            # writer.add_scalar('Train/Loss', loss.item(), epoch)

            # compute computation time and *compute_efficiency*
            process_time = start_time-time.time()
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
                process_time/(process_time+start_time), epoch, opt.epochs))
            start_time = time.time()

        # save checkpoint if needed
        if epoch % opt.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
            }, opt.checkpoint_path)

if __name__ == '__main__':
    main()