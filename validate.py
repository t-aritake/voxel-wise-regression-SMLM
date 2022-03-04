# -*- coding: utf-8 -*-
import torch
import datasets
import models
import augmentation
import pickle
import time
import numpy
import utils
import loss
import sys
numpy.random.seed(0)
torch.manual_seed(0)


def pairwise_l1_dist(x):
    D = x.unsqueeze(2) - x.unsqueeze(1)
    return D.abs().sum(-1)


def laplacian_kernel_loss(
        coord1, weights1, coord2, weights2, sigma,
        reduce=None, reduction='mean'):
    D = pairwise_l1_dist(torch.cat((coord1, coord2), 1))
    K = torch.exp(-D / sigma)

    weight_vec = torch.cat((weights1, -weights2), 1).unsqueeze(-1)

    dist = torch.matmul(torch.matmul(weight_vec.permute(0, 2, 1), K),
                        weight_vec)

    if reduce is False:
        return dist

    if reduction == 'mean':
        return torch.mean(dist)

    if reduction == 'sum':
        return torch.sum(dist)


# calc Jaccard index
def jaccard(x, y):
    intersection = numpy.intersect1d(x, y, assume_unique=True)
    union = numpy.union1d(x, y)

    return len(intersection) / len(union)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# specify directory of a learned model
loaddir = sys.argv[1]
if loaddir[-1] != '/':
    loaddir += '/'

# load trained parameters to a network
with open(loaddir + 'args.pkl', 'rb') as f:
    variables = pickle.load(f)
args = variables[0]

params = torch.load(loaddir + 'cnn_microscopy.pth', map_location=device)

model = models.Network().to(device)
model.load_state_dict(params)
model.eval()

# image size of a test image
image_size = torch.Tensor([64, 64, 4]).to(device)
image_size_large = image_size * 8
target_voxel_size = torch.Tensor([24, 24, 50]).to(device)
# minimum coordinates of voxels of test image
defaults = utils.gen_defaults(1/image_size, device, centering=False)
lossclass = loss.LocationAndConfidenceLoss(
    defaults, 1/image_size, default_centering=False, negapos_ratio=3)\
    .to(device)


# add random or fixed lateral drift
# translation = numpy.random.randint(-2, 2, size=(4, 2))
# translation[2] = [0, 0]
resolution_high = args.resolution_xy_low / args.scale_xy
z_list = numpy.linspace(
    0, args.resolution_depth_low * (args.low_depth-1), args.low_depth)
# aug = augmentation.FixedTranslation(translation, 2, z_list, resolution_high)
aug = augmentation.RandomTranslation(2, 2, z_list, resolution_high)
# aug = None

# validate set
validateset = datasets.RandomDataset(
    low_patchsize=64, low_depth=4,
    resolution_xy_low=192, resolution_depth_low=400,
    scale_xy=8, scale_depth=8,
    min_weight=0.3, max_weight=1.0,
    num_particles=8, data_size=1000, augmentation=aug)
# validateset = datasets.PreGeneratedData('./data/datasets/size_64_noaug/')
validate_loader = torch.utils.data.DataLoader(
    validateset, batch_size=50, shuffle=False, collate_fn=datasets.collate_fn)

# threshold value for binarize classification results
threshold = 0.9

# timer for count computation time
start_time = time.time()
print("start")
tdiffs = []
closest_log = []
MMD_log = []
jaccard_intersect = 0
jaccard_union = 0

# size of the target space
target_space_size = torch.Tensor([
    validateset._low_patchsize * args.resolution_xy_low,
    validateset._low_patchsize * args.resolution_xy_low,
    validateset._low_depth * args.resolution_depth_low]).to(device)
# minimum coordinate of the target space
target_space_min = torch.Tensor([
    validateset._xymin,
    validateset._xymin,
    validateset._zmin]).to(device)
with torch.no_grad():

    for i, (data, targets) in enumerate(validate_loader):
        # computational time for a batch
        inner_start = time.time()
        # transfer data
        data = data.to(device)
        targets = [target.to(device) for target in targets]
        # predict
        predictions = model(data).to(float)

        # rescale normalized coordinate to original scale
        predicted_locations = []
        rescaled_locations =\
            (predictions[:, :, :3] * (1/image_size) + defaults)\
            * target_space_size + target_space_min

        # computational time for a batch
        tdiff_inner = time.time() - inner_start
        tdiffs.append(tdiff_inner)

        # analyze each image
        for batch in range(len(targets)):
            # indices of positive voxels
            idx = torch.where(predictions[batch][:, 3] > threshold)[0]
            # true indices of positive voxels
            true_idx = (targets[batch][:, :3] * image_size).to(int)
            true_idx = true_idx[:, 0] + true_idx[:, 1] * data.shape[3]\
                + true_idx[:, 2] * (data.shape[2]*data.shape[3])
            # if predictions is empty, skip rest of the process
            if idx.shape[0] == 0:
                continue
            # estimated coordinates of molecules at positive voxels
            est_coord = rescaled_locations[batch][idx]
            # true coordinates of molecules (rescale normalized coordinates)
            true_coord = (
                targets[batch][:, :3] * target_space_size + target_space_min)

            # calculate distance between true molecules and estimated molecules
            diff = true_coord.cpu().numpy() - est_coord.cpu().numpy()[:, None]
            # dist = numpy.abs(diff).sum(2)
            dist = (diff**2).sum(2)
            # record distance to closest true molcule from estimated molecule
            min_idx = numpy.argmin(dist, 1)
            closest_coords = diff[numpy.arange(est_coord.shape[0]), min_idx]
            closest_log.append(closest_coords)

            # discretize estimation
            est_voxels = torch.div(est_coord - target_space_min,
                                   target_voxel_size, rounding_mode='floor')
            true_voxels = torch.div(true_coord - target_space_min,
                                    target_voxel_size, rounding_mode='floor')

            est_voxels = est_voxels[:, 0]\
                + est_voxels[:, 1] * image_size_large[0]\
                + est_voxels[:, 2] * image_size_large[0] * image_size_large[1]
            true_voxels = true_voxels[:, 0]\
                + true_voxels[:, 1] * image_size_large[0]\
                + true_voxels[:, 2] * image_size_large[0] * image_size_large[1]

            est_voxels = est_voxels.cpu().numpy()
            true_voxels = true_voxels.cpu().numpy()

            jaccard_union += len(numpy.union1d(est_voxels, true_voxels))
            jaccard_intersect\
                += len(numpy.intersect1d(est_voxels, true_voxels))
            if idx.shape[0] == 0:
                continue

            weight1 = torch.ones(est_coord.shape[0]).unsqueeze(0).to(device)
            weight2 = torch.ones(true_coord.shape[0]).unsqueeze(0).to(device)
            mmd = laplacian_kernel_loss(
                est_coord.unsqueeze(0).float(), weight1,
                true_coord.unsqueeze(0).float(), weight2, 12)
            MMD_log.append(mmd.item())


end_time = time.time()
tdiff = end_time - start_time
print('elapsed time:', tdiff)
print('fps={0:.1f}'.format(len(validate_loader.dataset)/tdiff))

closest_log = numpy.concatenate(closest_log)

numpy.save('./data/results/validate/closest_log.npy', closest_log)
print(numpy.mean(MMD_log), numpy.std(MMD_log), jaccard_intersect/jaccard_union)
