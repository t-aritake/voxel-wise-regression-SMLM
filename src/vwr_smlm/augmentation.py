"""Lateral (xy) drift between focal planes, applied to the observation coordinates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class _PlaneTranslation:
    def __init__(self, base_plane: int, observe_z: Sequence[float], resolution: float) -> None:
        # plane that is never shifted
        self.base_plane = base_plane
        # z coordinate of every focal plane [nm]
        self.observe_z = np.asarray(observe_z, dtype=float)
        # nm / px of the shift unit
        self.resolution = resolution

    def _apply(self, observe_coordinates: np.ndarray, translation: np.ndarray) -> np.ndarray:
        coord = observe_coordinates.copy()
        for z, shift in zip(self.observe_z, translation, strict=False):
            coord[np.isclose(coord[:, 2], z), :2] += shift * self.resolution
        return coord

    def __call__(self, observe_coordinates: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FixedTranslation(_PlaneTranslation):
    """Shift every plane by a fixed amount ``translation[d]`` (in units of ``resolution``)."""

    def __init__(
        self,
        translation: np.ndarray,
        base_plane: int,
        observe_z: Sequence[float],
        resolution: float,
    ) -> None:
        super().__init__(base_plane, observe_z, resolution)
        self.translation = np.asarray(translation)

    def __call__(self, observe_coordinates: np.ndarray) -> np.ndarray:
        return self._apply(observe_coordinates, self.translation)


class RandomTranslation(_PlaneTranslation):
    """Shift every plane except ``base_plane`` by a random integer amount.

    Shifts are drawn from ``[-max_translation, max_translation)`` (same as the original code).
    """

    def __init__(
        self, max_translation: int, base_plane: int, observe_z: Sequence[float], resolution: float
    ) -> None:
        super().__init__(base_plane, observe_z, resolution)
        self.max_translation = max_translation

    def __call__(self, observe_coordinates: np.ndarray) -> np.ndarray:
        translation = np.random.randint(
            -self.max_translation, self.max_translation, size=(len(self.observe_z), 2)
        )
        translation[self.base_plane] = 0
        return self._apply(observe_coordinates, translation)
