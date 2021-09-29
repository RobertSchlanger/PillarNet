from typing import List
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.utils.data_utils import Sequence

from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys


def select_best_anchors(arr):
    dims = np.indices(arr.shape[1:])
    # arr[..., 0:1] gets the occupancy value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box}
    ind = (np.argmax(arr[..., 0:1], axis=0),) + tuple(dims)
    return arr[ind]


class SimpleDataGenerator(Sequence):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, batch_size: int, lidar_files: List[str], label_files: List[str] = None,
                 calibration_files: List[str] = None):
        super(SimpleDataGenerator, self).__init__()
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.lidar_files = lidar_files
        self.label_files = label_files
        self.calibration_files = calibration_files

        assert (calibration_files is None and label_files is None) or \
               (calibration_files is not None and label_files is not None)

        if self.calibration_files is not None:
            assert len(self.calibration_files) == len(self.lidar_files)
            assert len(self.label_files) == len(self.lidar_files)

    def __len__(self):
        return len(self.lidar_files) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = np.arange(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
        #         print("inside getitem")
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []

        for i in file_ids:
            lidar = self.data_reader.read_lidar(self.lidar_files[i])
            # For each file, dividing the space into a x-y grid to create pillars
            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(lidar)

            pillars.append(pillars_)
            voxels.append(voxels_)

            if self.label_files is not None:
                label = self.data_reader.read_label(self.label_files[i])
                R, t = self.data_reader.read_calibration(self.calibration_files[i])
                # Labels are transformed into the lidar coordinate bounding boxes
                # Label has 7 values, centroid, dimensions and yaw value.
                label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)
                # These definitions can be found in point_pillars.cpp file
                # We are splitting a 10 dim vector that contains this information.
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    label_transformed)

                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.label_files is not None:
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        #         print("inside epoch")
        if self.label_files is not None:
            self.lidar_files, self.label_files, self.calibration_files = \
                shuffle(self.lidar_files, self.label_files, self.calibration_files)
