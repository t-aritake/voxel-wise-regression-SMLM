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


# calc Jaccard index
def jaccard(x, y):
    intersection = numpy.intersect1d(x, y, assume_unique=True)
    union = numpy.union1d(x, y)

    return len(intersection) / len(union)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# specify directory of a learned model
loaddir = sys.argsv[1]

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
# minimum coordinates of voxels of test image
defaults = utils.gen_defaults(1/image_size, device, centering=False)
lossclass = loss.LocationAndConfidenceLoss(defaults, 1/image_size, default_centering=False, negapos_ratio=3).to(device)


# add random or fixed lateral drift
# translation = numpy.random.randint(-2, 2, size=(4, 2))
# translation[2] = [0, 0]
resolution_high = args.resolution_xy_low / args.scale_xy
z_list = numpy.linspace(0, args.resolution_depth_low * (args.low_depth-1), args.low_depth)
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
validate_loader = torch.utils.data.DataLoader(validateset, batch_size=50, shuffle=False, collate_fn=datasets.collate_fn)

# threshold value for binarize classification results
threshold = 0.9

# timer for count computation time
start_time = time.time()
print("start")
tdiffs = []
jaccard_log = []
location_error_log = []

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
    confusion_log = []
    closest_log = []

    for i, (data, targets) in enumerate(validate_loader):
        print(i)
        # computational time for a batch
        inner_start = time.time()
        # transfer data
        data = data.to(device)
        targets = [target.to(device) for target in targets]
        # predict
        predictions = model(data).to(float)

        # rescale normalized coordinate to original scale
        predicted_locations = []
        rescaled_locations = (predictions[:, :, :3] * (1/image_size) + defaults) * target_space_size + target_space_min

        # computational time for a batch
        tdiff_inner = time.time() - inner_start
        tdiffs.append(tdiff_inner)

        # analyze each image
        for batch in range(len(targets)):
            # indices of positive voxels
            idx = torch.where(predictions[batch][:, 3] > threshold)[0]
            # true indices of positive voxels
            true_idx = (targets[batch][:, :3] * image_size).to(int)
            true_idx = true_idx[:, 0]  + true_idx[:, 1] * data.shape[3] + true_idx[:, 2] * (data.shape[2]*data.shape[3])
            # calculate Jaccard index
            jaccard_log.append(jaccard(idx.cpu().numpy(), true_idx.cpu().numpy()))
            # if predictions is empty, skip rest of the process
            if idx.shape[0] == 0:
                continue
            # estimated coordinates of molecules at positive voxels
            est_coord = rescaled_locations[batch][idx].cpu().numpy()
            # true coordinates of molecules (rescale normalized coordinates)
            true_coord = (targets[batch][:, :3] * target_space_size + target_space_min).cpu().numpy()

            # calculate distance between true molecules and estimated molecules
            diff = true_coord - est_coord[:, None]
            # dist = numpy.abs(diff).sum(2)
            dist = (diff**2).sum(2)
            # record distance to closest true molcule from estimated molecule
            min_idx = numpy.argmin(dist, 1)
            closest_coords = diff[numpy.arange(est_coord.shape[0]), min_idx]

            closest_log.append(closest_coords)

        if (i+1) % 100 == 0:
            print(i+1)


print(tdiffs)
end_time = time.time()
tdiff = end_time - start_time
print('elapsed time:', tdiff)
print('fps={0:.1f}'.format(len(validate_loader.dataset)/tdiff))

closest_log = numpy.concatenate(closest_log)

numpy.save('./diff.npy'.format(idx), closest_log)
numpy.save('./jaccard.npy'.format(idx), numpy.array(jaccard_log))
