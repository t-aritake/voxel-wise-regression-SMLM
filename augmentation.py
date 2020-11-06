import torch
import pdb
import numpy

""" class to add noise to the observation coordinates
"""

class FixedTranslation(torch.nn.Module):
    def __init__(self, translation, base_plane, observe_z, resolution):
        super().__init__()
        # 固定ずれ配列
        self._translation = translation
        # base plane which is not affected by noise
        self._base_plane = base_plane
        # candidate of z
        self._observe_z = observe_z
        # nm / px along x and y axes
        self._resolution = resolution

    def forward(self, observe_coordinates):
        coord = numpy.copy(observe_coordinates)
        # add noise to observed coordinate
        for d in range(len(self._observe_z)):
            coord[coord[:, 2] == self._observe_z[d], :2] += self._translation[d] * self._resolution

        # indices of depth that is not a base plane
        rows = [i for i in range(0, len(self._observe_z)) if i != self._base_plane]

        return coord, self._translation[rows].flatten()

class RandomTranslation(torch.nn.Module):
    def __init__(self, max_translation, base_plane, observe_z, resolution):
        super().__init__()
        # maximum pixel of translation
        self._max_translation = max_translation
        # base plane which is not affected by noise
        self._base_plane = base_plane
        # candidate of z
        self._observe_z = observe_z
        # nm / px along x and y axes
        self._resolution = resolution

    def forward(self, observe_coordinates):
        coord = numpy.copy(observe_coordinates)
        # sample random translation
        translation = numpy.random.randint(-self._max_translation, self._max_translation, size=(len(self._observe_z), 2))
        # set translation of base plane as 0
        translation[self._base_plane] = 0
        # add noise
        for d in range(len(self._observe_z)):
            coord[coord[:, 2] == self._observe_z[d], :2] += translation[d] * self._resolution

        # indices of depth that is not a base plane
        rows = [i for i in range(0, len(self._observe_z)) if i != self._base_plane]

        return coord, translation[rows].flatten()
