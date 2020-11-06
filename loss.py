# -*- coding: utf-8 -*-
import torch
import numpy
import pdb


class LocationAndConfidenceLoss(torch.nn.Module):
    def __init__(self, defaults, default_interval, default_centering=False, negapos_ratio=3):
        """
        Loss function class to calculate classification loss and location loss

        Parameters
        ----------
        defaults : torch.Tensor
            default location of molecule in each voxel
        default_interval : torch.Tensor
            normalized interval size (normalized size of a voxel)
        negapos_ratio : int
            ratio of negative and positive data which are used to calculate classification loss
        """
        super().__init__()

        # register default location of molecules in voxels as a buffer
        self.register_buffer('defaults', defaults)
        # register default interval as a buffer
        self.register_buffer('default_interval', default_interval)
        num_voxels = (1 / self.default_interval).to(int)
        # multiplier used for calculating index of voxels
        self.register_buffer('index_multiplier', torch.Tensor([1, num_voxels[0], num_voxels[0]*num_voxels[1]]).to(int))
        self._default_centering = default_centering
        self._negapos_ratio = negapos_ratio


    def forward(self, predictions, targets):
        """
        Calculate loss function


        Parameters
        ----------
        predictions:
            predicted value by a network

        targets:
            true voxel class and moleucle coordinates



        Returns
        ----------
        location_loss:
            loss for molecule coordinates

        confidence_loss:
            loss for voxel classifications
        """

        # num_targets = [x.shape[0] for x in targets]

        selected_predictions = []
        targets_locations = []
        targets_confidence = torch.zeros_like(predictions[:, :, 3])
        positive_indices = []
        num_defaults = (1 / self.default_interval).to(int)
        for batch in range(len(targets)):
            # convert target values to indices along each axis
            indices = (targets[batch][:, :3] * num_defaults).to(int)
            # convert indices along axis to voxel indices
            indices = (indices * self.index_multiplier).sum(1).flatten()

            # convert target coordinate into offset for belonging voxels
            if self._default_centering:
                location_diff = (targets[batch][:, :3]-self.defaults[indices]) / (self.default_interval/2)
            else:
                location_diff = (targets[batch][:, :3]-self.defaults[indices]) / self.default_interval
            # add offset to list
            targets_locations.append(location_diff)
            # add indices of positive label to list
            positive_indices.append(indices)
            # add predicted value at voxels which contains ground-truth molecule to list
            selected_predictions.append(predictions[batch, indices])
            # set targets confidence of the voxel which contains a molecule as 1
            targets_confidence[batch, indices] = 1


        # concatenate selected predictions
        selected_predictions = torch.cat(selected_predictions)
        # locations of selected predictions
        selected_locations = selected_predictions[:, :3]
        # confidences of selected predictions
        selected_confidence = selected_predictions[:, 3]
        targets_locations = torch.cat(targets_locations)

        # calculate error for true and predicted location in a voxel
        location_loss = torch.nn.functional.l1_loss(selected_locations, targets_locations, reduction='sum')

        # calculate classification error of each voxel
        confidence_loss = torch.nn.functional.binary_cross_entropy(predictions[:, :, 3], targets_confidence, reduction='none')
        # mask of voxels where a molecule is contained
        positive_mask = (targets_confidence == 1)
        # ignore confidence in which label of the voxel is positive
        confidence_loss[positive_mask] = 0
        # sort confidence loss of negative label in descending order
        _, loss_idx = confidence_loss.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # sum of targets_confidence equal a number of positive label
        # the number of negative elements to calculate classification error is _negapos_ration times of the number of positive elements
        # torch.clamp is used not to exceed the number of voxels = targets_confidence.shape[1]
        num_negative = torch.clamp(targets_confidence.sum(1) * self._negapos_ratio, max= targets_confidence.shape[1])
        # mask of negative elements whose rank is less than num_negative
        negative_mask = idx_rank < num_negative.unsqueeze(1)

        # get predicted confidence and target confidence of positive and selected negative voxels
        masked_predictions_confidence = predictions[positive_mask + negative_mask][:, 3]
        masked_targets_confidence = targets_confidence[positive_mask + negative_mask]

        # calculate confidence loss
        confidence_loss = torch.nn.functional.binary_cross_entropy(masked_predictions_confidence, masked_targets_confidence, reduction='sum')

        # calculate average loss by dividing the loss by number of batches
        location_loss /= predictions.shape[0]
        confidence_loss /= predictions.shape[0]

        return location_loss, confidence_loss


if __name__ == '__main__':
    # test codes to calculate loss
    torch.manual_seed(0)

    import datasets
    import utils
    num_particles = numpy.random.randint(40, 50, size=100)
    ds = datasets.RandomDataset(
        low_patchsize=64, low_depth=4, resolution_xy_low=192, resolution_depth_low=400,
        min_weight=0.3, max_weight=1.0,
        lmbd=20, xymin=0, zmin=0,
        scale_xy=8, scale_depth=8, num_particles=num_particles, data_size=10)
    loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, collate_fn=datasets.collate_fn)


    # predictions
    predictions = torch.rand(10, 64*64*4, 4)
    # target values
    target = torch.rand(10, 20, 4)

    intervals = torch.Tensor([1.0/64, 1.0/64, 1.0/4])
    defaults = utils.gen_defaults(intervals)
    loss = LocationAndConfidenceLoss(intervals)
    # loss(prediction, target)

    for batch_idx, (img, annotation) in enumerate(loader):
        print(loss(predictions, annotation, defaults))
