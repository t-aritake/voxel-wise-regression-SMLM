# -*- coding: utf-8 -*-
import argparse
import numpy
import torch
import datasets
import models
import utils
import loss
import pickle
import augmentation


def make_savedir(parent_dir='./'):
    """
    make directory to save results

    Params
    ----------
    parent_dir : str
        parent directory to save results
    """

    if parent_dir[-1] != '/':
        parent_dir += '/'
    import datetime
    import os
    orig_dirname = datetime.datetime.now().strftime('%y%m%d_%H%M')
    orig_dirname = parent_dir + orig_dirname
    dirname = orig_dirname
    count = 1
    while os.path.exists(dirname):
        dirname = orig_dirname + str(count)
        count += 1
    os.mkdir(dirname)

    return dirname


def train(args, model, device, train_loader, optimizer, loss_func, epoch):
    model.train()
    train_loss = 0
    running_loss = 0
    running_datasize = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = [target.to(device) for target in targets]
        optimizer.zero_grad()
        predictions = model(data)
        location_loss, confidence_loss = loss_func(predictions, targets)

        loss = location_loss + confidence_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        running_loss += loss.item()
        running_datasize += 1

        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ([{:.0f}%)]\tLoss: {:.4e}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader),
                running_loss / running_datasize))
            running_loss = 0
            running_datasize = 0

    return train_loss


def test(args, model, device, test_loader, loss_func):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, targets in test_loader:
            # data, target_coord, target_weight = data.to(device), target_coord.to(device), target_weight.to(device)
            data = data.to(device)
            targets = [target.to(device) for target in targets]
            predictions = model(data)
            # test_loss += torch.nn.functional.binary_cross_entropy(output, target, reduction='sum').item()
            # test_loss += laplacian_kernel_loss(est_coord, est_weight, target_coord, target_weight, 192, reduction='sum')
            location_loss, confidence_loss = loss_func(predictions, targets)
            test_loss += (location_loss + confidence_loss).item()
            # 粒子のインデックスの確認するならここ？

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.3g}'.format(test_loss))
    print('-' * 20)

    return test_loss


def main():
    parser = argparse.ArgumentParser(
        description='DeepLoco')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
            help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
            help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
            help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='M',
            help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--low-patchsize', type=int, default=64, metavar='N',
            help='patch size of data in low resolution (default: 16)')
    parser.add_argument('--low-depth', type=int, default=4, metavar='N',
            help='depth of data in low resolution (default: 4)')
    parser.add_argument('--resolution-xy-low', type=int, default=192, metavar='N',
            help='resolution(nm/pixel) of xy plane in low resolution image (default: 192)')
    parser.add_argument('--resolution-depth-low', type=int, default=400, metavar='N',
            help='resolution(nm/plane) of depth in low resolution image (default: 400)')
    parser.add_argument('--scale-xy', type=int, default=8, metavar='N',
            help='scaling factor of horizontal direction (default: 8)')
    parser.add_argument('--scale-depth', type=int, default=8, metavar='N',
            help='scaling factor of depth (default: 8)')
    parser.add_argument('--min-weight', type=float, default=1.0, metavar='F',
            help='minimum weight of a molecule (default: 1.0)')
    parser.add_argument('--max-weight', type=float, default=1.0, metavar='F',
            help='maximum weight of a molecule (default: 1.0)')
    parser.add_argument('--num-particle-train', type=int, default=5, metavar='N',
            help='number of fluorescent particles in one frame in training data (default: 3)')
    parser.add_argument('--num-particle-test', type=int, default=3, metavar='N',
            help='number of fluorescent particles in one frame int test data (default: 1)')
    parser.add_argument('--num-data-train', type=int, default=10000, metavar='N',
            help='number of training data (default: 10000)')
    parser.add_argument('--num-data-test', type=int, default=1000, metavar='N',
            help='number of test data (default: 1000)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--savedir', default='./data/learned_models/', help='save directory is created inside this directory (default="data/learned_models")')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers':1, 'pin_memory': True} if use_cuda else {}

    # random lateral drifts within plus/minus 48 nm
    resolution_high = args.resolution_xy_low / args.scale_xy
    z_list = numpy.linspace(0, args.resolution_depth_low * (args.low_depth-1), args.low_depth)
    aug = augmentation.RandomTranslation(2, 2, z_list, resolution_high)

    # train set
    trainset = datasets.RandomDataset(
        low_patchsize=args.low_patchsize, low_depth=args.low_depth,
        resolution_xy_low=args.resolution_xy_low, resolution_depth_low=args.resolution_depth_low,
        scale_xy=args.scale_xy, scale_depth=args.scale_depth,
        min_weight=args.min_weight, max_weight=args.max_weight,
        num_particles=args.num_particle_train, data_size=args.num_data_train, augmentation=aug)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=datasets.collate_fn, **kwargs)

    # test set (parameters are same as train set)
    testset = datasets.RandomDataset(
        low_patchsize=args.low_patchsize, low_depth=args.low_depth,
        resolution_xy_low=args.resolution_xy_low, resolution_depth_low=args.resolution_depth_low,
        scale_xy=args.scale_xy, scale_depth=args.scale_depth,
        min_weight=args.min_weight, max_weight=args.max_weight,
        num_particles=args.num_particle_test, data_size=args.num_data_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, collate_fn=datasets.collate_fn, **kwargs)

    # minimum and maximum coordinate of x, y, z of the target space
    xlim = (0, args.resolution_xy_low * args.low_patchsize)
    ylim = (0, args.resolution_xy_low * args.low_patchsize)
    zlim = (0, args.resolution_depth_low * args.low_depth)
    # minimum coordinate
    coord_min = torch.Tensor((xlim[0], ylim[0], zlim[0]))
    # maximum coordinate
    coord_max = torch.Tensor((xlim[1], ylim[1], zlim[1]))
    # regression model
    model = models.Network().to(device)

    # size of the image
    image_size = torch.Tensor([args.low_patchsize, args.low_patchsize, args.low_depth]).to(device)
    # minimum coordinates of each voxel
    defaults = utils.gen_defaults(1/image_size, device, centering=False)
    # loss function (binary cross entropy + l1 distance)
    lossfunc = loss.LocationAndConfidenceLoss(defaults, 1/image_size, default_centering=False, negapos_ratio=3).to(device)

    trainloss_history = []
    testloss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # iterate train and test
    for epoch in range(1, args.epochs + 1):
        trainloss = train(args, model, device, train_loader, optimizer, lossfunc, epoch)
        testloss = test(args, model, device, test_loader, lossfunc)

        trainloss_history.append(trainloss)
        testloss_history.append(testloss)

        # save initial model
        if epoch == 1 and args.save_model:
            dirname = make_savedir(args.savedir)

    if args.save_model:
        model = model.cpu()
        torch.save(model.state_dict(), dirname + '/cnn_microscopy.pth')
        with open(dirname + '/args.pkl', mode='wb') as f:
            pickle.dump([args, train_loader, test_loader, trainloss_history, testloss_history], f)


if __name__ == '__main__':
    main()
