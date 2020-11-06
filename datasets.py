# -*- coding: utf-8 -*-
import torch
import torch.utils.data
import numpy
from PIL import Image
import pdb


def psf(observe_coordinates, molecule_coordinates):
    """ Point Spread Function

    Parameters
    ----------
    observe_coordinates : numpy.ndarray (n, 3)
        array of observation coordinates
    molecule_coordinate :  numpy.ndarray (3,)
        array of molecule coordinates
    """
    w0 = 133.0525
    d = 302.3763
    A = 0.000737
    B = 0.122484
    adash = 5e7

    diff = observe_coordinates[None, :, :] - molecule_coordinates[:, None, :]
    
    z_diff = diff[:, :, 2] / d
    w = w0 * numpy.sqrt(1 + z_diff**2 + A * z_diff**3 + B * z_diff**4)
    a = adash / (2 * numpy.pi * w**2)
    b = 0

    G = a * numpy.exp(-numpy.sum((diff[:, :, :2]/w[:, :, None])**2/2, axis=2)) + b
    
    return G




class PreGeneratedData(torch.utils.data.Dataset):
    """
    dataset class to load saved data
    """
    def __init__(self, dirpath):
        self._dirpath = dirpath
        if dirpath[-1] != '/':
            self._dirpath = dirpath + '/'
        import pathlib
        p = pathlib.Path(dirpath)
        self._data_size = len(list(p.glob('data_*')))

    def __len__(self):
        return self._data_size

    def __getitem__(self, index):
        observe, target = torch.load(self._dirpath + 'data_{0:d}.pt'.format(index))

        return observe, target


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, low_patchsize, low_depth, resolution_xy_low, resolution_depth_low, scale_xy, scale_depth,
                 min_weight = 1.0, max_weight = 1.0,
                 xymin = 0, zmin=0, 
                 num_particles=3, data_size=10000, augmentation=None, lmbd=10, sigma=3):
        """
        Dataset class to randomly distribute molecules in the target space

        Parameters
        ----------
        low_patchsize : int
            image size of each depth of low resolution tensor
        low_depth : int
            depth of low resolution tensor
        resolution_xy_low : int
            nm/pixel in low resolution image (horizontal)
        resolution_depth_low : int
            nm/pixel in low resolution image (depth)
        scale_xy : int
            scaling factor of super resolution (horizontal)
        scale_depth : int
            scaling factor of super resolution (depth)
        min_weight : float
            minimum weight of the molecule peak fluorescence
        max_weight : float
            maximum weight of the molecule peak fluorescence
        xymin : int
            minimum(starting) value of xy coordinate
        zmin : int
            minimum(starting) value of z coordinate
        num_particles : int or numpy.ndarray
            number of fluorescent particles (default 3)
        data_size : int
            number of data to be generated (default 10000)
        augmentation : torch.nn.Module
            augumentation method to apply for observe coordinate for data augmentation
        lmbd : float
            parameter of shot noise (poisson distribution)
        sigma : float
            parameter of observation noise (normal distribution)
        """
        super().__init__()

        self._low_patchsize = low_patchsize
        self._low_depth = low_depth
        # nm per pixel in low resolution
        self._resolution_xy_low = resolution_xy_low
        self._resolution_depth_low = resolution_depth_low
        # target scale
        self._scale_xy = scale_xy
        self._scale_depth = scale_depth

        resolution_xy_high, resolution_depth_high = self._calc_resolution_high()
        # 画像の範囲
        self._xymin = xymin
        self._xymax = resolution_xy_low * low_patchsize + self._xymin

        # self._zmin = zmin - resolution_
        # self._zmax = resolution_depth_low * (low_depth-1) + zmin
        self._zmin = zmin - resolution_depth_high * (scale_depth//2)
        # upsample前のzmin, zmaxを計算
        zmax = resolution_depth_low * (low_depth-1)
        self._zmax = zmax + resolution_depth_high * int((scale_depth-0.1)//2)

        # データ生成用パラメータ
        if type(num_particles) == int:
            self._num_particles = numpy.repeat(num_particles, data_size)
        else:
            self._num_particles = num_particles
        self._data_size = data_size
        # self._observe_translation = observe_translation
        self._augmentation = augmentation
        self._lmbd = lmbd
        self._sigma = sigma

        # データ生成のために必要な情報を生成（分子の座標，係数の強さ）
        # 観測座標は共通
        self._observe_coordinates = self._gen_observation_coordinates()
        self._particle_coordinates = numpy.array([
            numpy.random.uniform(self._xymin, self._xymax, size=self._num_particles.sum()),
            numpy.random.uniform(self._xymin, self._xymax, size=self._num_particles.sum()),
            numpy.random.uniform(self._zmin, self._zmax, size=self._num_particles.sum())]).T
        # self._weight = numpy.ones(shape=(num_particles * data_size, ))
        self._weight = numpy.random.uniform(min_weight, max_weight, size=(self._num_particles.sum(), ))


    def __len__(self):
        return self._data_size

    def __getitem__(self, index):
        high_patchsize, high_depth = self._calc_highres_voxel_nums()
        resolution_xy_high, resolution_depth_high = self._calc_resolution_high()
        # 低解像度データ用配列 0で初期化
        x = numpy.zeros(shape=(self._low_patchsize, self._low_patchsize, self._low_depth))
        # 正解データ（座標と確信度の配列）
        _return_values = numpy.zeros(shape=(self._num_particles[index], 4))
        # 正解データの確信度は1.0で初期化
        _return_values[:, 3] = 1.0

        # 座標，重み配列の初期index
        index_start = self._num_particles[:index].sum()
        index_end = self._num_particles[:index+1].sum()
        tmp_weight = self._weight[index_start:index_end]

        # 対象とする空間サイズで[0, 1]に正規化された座標を正解データとして返す
        # 厳密な座標に戻す際は画像サイズの情報を使って戻さざるをえない．
        _return_values[:, :3] = self._particle_coordinates[index_start:index_end]
        # size of space
        denom = numpy.array([self._xymax - self._xymin, self._xymax - self._xymin, self._zmax - self._zmin])
        if denom[2] == 0:
            denom[2] = 1
        _return_values[:, :3] = (_return_values[:, :3] - numpy.array([self._xymin, self._xymin, self._zmin])) / denom

        if self._augmentation is None:
            observe_coordinates = self._observe_coordinates
        else:
            observe_coordinates, aug_params = self._augmentation(self._observe_coordinates)

        count = 0
        kernels = psf(observe_coordinates, self._particle_coordinates[index_start:index_end])
        x = (kernels * tmp_weight[:, None]).sum(0).reshape(x.shape, order='F')

        # 観測データにショットノイズと観測雑音を加える
        x += numpy.random.poisson(lam=self._lmbd, size=x.shape)
        x += numpy.random.normal(size=x.shape) * self._sigma

        # torchのテンソルに変換して返す
        x = torch.from_numpy(x.transpose(2, 0, 1)).float()

        # if self._augmentation is not None:
        #     aug_params = torch.from_numpy(aug_params).float()
        #     return x, y, aug_params
        # return x, (torch.from_numpy(_coordinate[:, :-1]).float(), torch.from_numpy(_weight).float())

        return x, _return_values

    def _calc_highres_voxel_nums(self):
        # 高解像でのピクセル数
        high_patchsize = self._low_patchsize * self._scale_xy
        # 高解像での深さ数
        high_depth = self._low_depth * self._scale_depth

        return high_patchsize, high_depth

    def _calc_resolution_high(self):
        # nm per pixel in high-res
        resolution_xy_high = self._resolution_xy_low // self._scale_xy
        # nm per depth in high-res
        resolution_depth_high = self._resolution_depth_low // self._scale_depth

        return resolution_xy_high, resolution_depth_high


    def _gen_observation_coordinates(self):
        resolution_xy_high, resolution_z_high = self._calc_resolution_high()
        observe_xy = (numpy.arange(self._low_patchsize) + 0.5) * self._resolution_xy_low
        observe_z = numpy.arange(self._low_depth) * self._resolution_depth_low

        # z, y, xの順で変化していく順
        # observe_coordinates = numpy.array(numpy.meshgrid(observe_xy, observe_xy, observe_z)).transpose(2,1,3,0).reshape([-1, 3])
        # x, y, zの順で変化
        observe_coordinates = numpy.array(numpy.meshgrid(observe_xy, observe_xy, observe_z)).transpose(3,1,2,0).reshape([-1, 3])

        return observe_coordinates


def collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))

    imgs = torch.stack(imgs,dim=0)

    return imgs, targets

class Tubulin_191216(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 30000

    def __getitem__(self, index):
        im2 = numpy.array(Image.open('./data/191216/cam2/{:05d}.tiff'.format(index)))
        im1 = numpy.array(Image.open('./data/191216/cam1/{:05d}.tiff'.format(index)))
        im3 = numpy.array(Image.open('./data/191216/cam3/{:05d}.tiff'.format(index)))
        im4 = numpy.array(Image.open('./data/191216/cam4/{:05d}.tiff'.format(index)))

        im = numpy.array([
            im2,
            im1,
            im3,
            im4]) 
        
        x = torch.from_numpy(im).float()

        return x

class Tubulin_191216_masked(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 30000

    def __getitem__(self, index):
        im2 = numpy.array(Image.open('./data/datasets/191216/masked_cam2/{:05d}.tiff'.format(index)))
        im1 = numpy.array(Image.open('./data/datasets/191216/masked_cam1/{:05d}.tiff'.format(index)))
        im3 = numpy.array(Image.open('./data/datasets/191216/masked_cam3/{:05d}.tiff'.format(index)))
        im4 = numpy.array(Image.open('./data/datasets/191216/masked_cam4/{:05d}.tiff'.format(index)))

        im = numpy.array([
            im2,
            im1,
            im3,
            im4]) 
        
        x = torch.from_numpy(im).float()

        return x




if __name__ == '__main__':
    # test codes
    numpy.random.seed(0)

    ds = RandomDataset(
        low_patchsize=64, low_depth=4, resolution_xy_low=192, resolution_depth_low=400,
        min_weight=0.3, max_weight=1.0,
        lmbd=20, xymin=0, zmin=0,
        scale_xy=8, scale_depth=1, num_particles=50, data_size=100)

    x, annotation = ds.__getitem__(0)


    print(x.shape)
    print(annotation)

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(annotation[:, 0] * 64*192, annotation[:, 1]*64*192, annotation[:, 2]*4*400, marker='x')
    ax.set_xlabel('x [nm]')
    ax.set_ylabel('y [nm]')
    ax.set_zlabel('z [nm]')
    ax.set_xlim(0, 64*192)
    ax.set_ylim(0, 64*192)
    ax.set_zlim(0, 1600)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
