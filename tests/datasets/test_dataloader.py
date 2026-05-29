# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from src.datasets.utils.dataloader import ConcatIndices

try:
    from src.datasets.video_dataset import VideoDataset
except ModuleNotFoundError:
    VideoDataset = None


class TestConcatIndices(unittest.TestCase):
    def test_concat_indices(self):
        sizes = [10, 20, 30, 40]
        total_size = sum(sizes)
        concat_indices = ConcatIndices(sizes)

        # -1 is outside the total range
        with self.assertRaises(ValueError):
            concat_indices[-1]
        # 0-9 map to dataset 0
        self.assertEqual(concat_indices[0], (0, 0))
        self.assertEqual(concat_indices[9], (0, 9))
        # 10-29 map to dataset 1
        self.assertEqual(concat_indices[10], (1, 0))
        self.assertEqual(concat_indices[29], (1, 19))
        # 30-59 map to dataset 2
        self.assertEqual(concat_indices[30], (2, 0))
        self.assertEqual(concat_indices[59], (2, 29))
        # 60-99 map to dataset 3
        self.assertEqual(concat_indices[60], (3, 0))
        self.assertEqual(concat_indices[99], (3, 39))
        # 100 is outside the total range
        with self.assertRaises(ValueError):
            concat_indices[total_size]


class TestVideoDatasetMetadataCrop(unittest.TestCase):
    @unittest.skipIf(VideoDataset is None, "video dataset dependencies are not installed")
    def test_metadata_crop_returns_view_with_expected_pixels(self):
        buffer = np.arange(2 * 5 * 7 * 3, dtype=np.uint8).reshape(2, 5, 7, 3)
        metadata = {
            "crop_x": 2,
            "crop_y": 1,
            "crop_width": 3,
            "crop_height": 2,
        }

        cropped = VideoDataset._apply_metadata_crop(buffer, metadata)

        np.testing.assert_array_equal(cropped, buffer[:, 1:3, 2:5, :])
        self.assertTrue(np.shares_memory(cropped, buffer))
