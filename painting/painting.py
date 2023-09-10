import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image
from tqdm import tqdm

import pdb

from mmseg.apis import inference_model, init_model
import mmcv

import calibration_kitti

TRAINING_PATH = "../data/kitti/training/"
TWO_CAMERAS = True


class Painter:
    def __init__(self):
        self.root_split_path = TRAINING_PATH
        self.save_path = TRAINING_PATH + "painted_lidar/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.model = None

        """
        download config file and checkpoint file
        mim download mmsegmentation --config deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024 --dest .
        """

        print(f'Using Segmentation Network -- deeplabv3plus')
        config_file = './config/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py'
        checkpoint_file = './checkpoint/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')

    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne_reduced/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, left):
        output_reassign_softmax = None

        filename = self.root_split_path + left + ('%s.png' % idx)
        result = inference_model(self.model, filename)
        # person 11, rider 12, vehicle 13/14/15/16, bike 17/18
        output_permute = torch.tensor(result.seg_logits.data).permute(1, 2, 0)  # H, W, 19
        sf = torch.nn.Softmax(dim=2)

        output_reassign = torch.zeros(output_permute.size(0), output_permute.size(1), 5)
        output_reassign[:, :, 0], _ = torch.max(output_permute[:, :, :11], dim=2)  # background
        output_reassign[:, :, 1], _ = torch.max(output_permute[:, :, [17, 18]], dim=2)  # bicycle
        output_reassign[:, :, 2], _ = torch.max(output_permute[:, :, [13, 14, 15, 16]], dim=2)  # car
        output_reassign[:, :, 3] = output_permute[:, :, 11]  # person
        output_reassign[:, :, 4] = output_permute[:, :, 12]  # rider
        output_reassign_softmax = sf(output_reassign).cpu().numpy()

        return output_reassign_softmax

    def get_calib(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        return calibration_kitti.Calibration(calib_file)

    def get_calib_fromfile(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        calib = calibration_kitti.get_calib_from_file(calib_file)
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        calib['Tr_velo2cam'] = np.concatenate([calib['Tr_velo2cam'], np.array([[0., 0., 0., 1.]])], axis=0)
        return calib

    def cam_to_lidar(self, pointcloud, projection_mats):
        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1])  # copy reflectances column
        lidar_velo_coords[:, -1] = 1  # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo2cam'].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances

        return lidar_cam_coords

    def create_cyclist(self, augmented_lidar):
        rider_idx = np.where(augmented_lidar[:, 8] >= 0.3)[0]  # 0, 1(bike), 2, 3(person), 4(rider)
        rider_points = augmented_lidar[rider_idx]
        bike_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
        bike_total = (np.argmax(augmented_lidar[:, -5:], axis=1) == 1)
        for i in range(rider_idx.shape[0]):
            bike_mask = (np.linalg.norm(augmented_lidar[:, :3] - rider_points[i, :3], axis=1) < 1) & bike_total
            bike_mask_total |= bike_mask
        augmented_lidar[bike_mask_total, 8] = augmented_lidar[bike_mask_total, 5]
        augmented_lidar[bike_total ^ bike_mask_total, 4] = augmented_lidar[bike_total ^ bike_mask_total, 5]

        return augmented_lidar[:, [0, 1, 2, 3, 4, 8, 6, 7]]

    def augment_lidar_class_scores_both(self, class_scores, lidar_raw, projection_mats):

        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)  # (20285, 4)

        # left
        lidar_cam_coords[:, -1] = 1  # homogenous coords for projection
        points_projected = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected = points_projected.transpose()
        points_projected = points_projected / (points_projected[:, 2].reshape(-1, 1))

        true_x_on_img = (0 < points_projected[:, 0]) & (points_projected[:, 0] < class_scores.shape[1])  # x in img coords is cols of img
        true_y_on_img = (0 < points_projected[:, 1]) & (points_projected[:, 1] < class_scores.shape[0])
        true_point = true_x_on_img & true_y_on_img

        points_projected = points_projected[true_point]
        points_projected = np.floor(points_projected).astype(int)  # using floor so you don't end up indexing num_rows+1th row or col
        points_projected = points_projected[:,:2]  # drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        # indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        # socre dimesion: point_scores.shape[2]
        point_scores = class_scores[points_projected[:, 1], points_projected[:, 0]].reshape(-1, class_scores.shape[2])

        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores.shape[2]))), axis=1)
        augmented_lidar[true_point, -class_scores.shape[2]:] += point_scores
        augmented_lidar[true_point, -class_scores.shape[2]:] = 0.5 * augmented_lidar[true_point, -class_scores.shape[2]:]

        augmented_lidar = self.create_cyclist(augmented_lidar)  # (20285, 8)

        return augmented_lidar

    def run(self):
        num_image = 7481
        for idx in tqdm(range(num_image)):
            sample_idx = "%06d" % idx

            # points: N * 4(x, y, z, r)
            points = self.get_lidar(sample_idx)

            # get segmentation score from network
            scores_from_cam = self.get_score(sample_idx, "image_2/")

            # get calibration data
            calib_fromfile = self.get_calib_fromfile(sample_idx)

            # paint the point clouds
            # points: N * 8
            points = self.augment_lidar_class_scores_both(scores_from_cam, points, calib_fromfile)

            np.save(self.save_path + ("%06d.npy" % idx), points)


if __name__ == '__main__':
    painter = Painter()
    painter.run()
