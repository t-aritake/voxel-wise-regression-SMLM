import torch
import os
import datasets
import models
import pickle
import time
import numpy
import utils
import loss
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load parameters of a learned model
loaddir = sys.argv[1]
savedir = sys.argv[2]

if not os.path.exists(savedir):
    os.mkdir(dirname)

with open(loaddir + 'args.pkl', 'rb') as f:
    variables = pickle.load(f)
args = variables[0]

params = torch.load(loaddir + 'cnn_microscopy.pth', map_location=device)

model = models.Network().to(device)
model.load_state_dict(params)
model.eval()

# load real data as validataset
validateset = datasets.Tubulin_191216()
# validateset = datasets.Tubulin_191216_masked()
validate_loader = torch.utils.data.DataLoader(validateset, batch_size=50, shuffle=False)

# size of a real image
image_size = torch.Tensor([256, 256, 4]).to(device)
# minimum coordinates of each voxel of input image
defaults = utils.gen_defaults(1/image_size, device, centering=False)
lossclass = loss.LocationAndConfidenceLoss(defaults, 1/image_size, default_centering=False, negapos_ratio=3).to(device)
# thresholding value for binarizing classification results
thresh = 0.5

start_time = time.time()
# output = numpy.array([])
particles = numpy.empty(shape=[0, 3])
num_particles = []

# size of the target space
target_space_size = torch.Tensor([
    256 * 192, 256 * 192, 4 * 400]).to(device)
# the minimum coordinate of the target space
target_space_min = torch.Tensor([0, 0, -200]).to(device)

with torch.no_grad():
    for i, data in enumerate(validate_loader):
        if (i+1) % 10 == 0:
            print(i+1)
            end_time = time.time()
            tdiff = end_time - start_time
            print('elapsed time:', tdiff)
            print('fps={0:.1f}'.format(i*validate_loader.batch_size/tdiff))
        data = data.to(device)
        predictions = model(data).to(float)

        # rescale predictions
        predictions[:, :, :3] = (predictions[:, :, :3] * (1/image_size) + defaults) *target_space_size + target_space_min

        # save predicted coordinates as a text file
        f = open(savedir + 'coord_{0:.2f}.txt'.format(thresh), 'a')
        for pred in predictions:
            coordinates = pred[pred[:, 3] > thresh].cpu().numpy()
            numpy.savetxt(f, coordinates, delimiter=',')
        f.close()

end_time = time.time()
tdiff = end_time - start_time
print('elapsed time:', tdiff)
print('fps={0:.1f}'.format(len(validate_loader.dataset)/tdiff))
