from collections import Counter

import medmnist
import numpy as np
from medmnist import INFO
from torchvision.transforms import transforms


class MedMnistData:

    def __init__(self, data_flag, download=True):
        self.data_flag = data_flag
        self.download = download
        self.info = INFO[data_flag]
        self.task = self.info['task']
        self.n_channels = self.info['n_channels']
        self.n_classes = len(self.info['label'])
        self.DataClass = getattr(medmnist, self.info['python_class'])

        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        # load the data
        self.train_dataset = self.DataClass(
            split='train', transform=data_transform, download=download)
        self.test_dataset = self.DataClass(
            split='test', transform=data_transform, download=download)

        if self.n_channels == 1:
            rgb_images = np.stack((self.train_dataset.imgs,) * 3, axis=-1)
            self.train_dataset.imgs = rgb_images
            rgb_images = np.stack((self.test_dataset.imgs,) * 3, axis=-1)
            self.test_dataset.imgs = rgb_images

    def select_n_classes(self, n_classes):
        class_labels = self.train_dataset.labels
        flat_class_labels = class_labels.flatten()
        class_counts = Counter(flat_class_labels)
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1],
                                reverse=True)
        selected_classes = sorted_classes[:n_classes]

        # Create a dataset with only the selected classes
        self.trim_data_by_labels(self.train_dataset, flat_class_labels,
                                 selected_classes)
        test_labels = self.test_dataset.labels.flatten()
        self.trim_data_by_labels(self.test_dataset, test_labels,
                                 selected_classes)

    def trim_data_by_labels(self, dataset, flat_class_labels, selected_labels):
        dataset.imgs = \
            dataset.imgs[np.isin(flat_class_labels, selected_labels)]
        filtered_labels = dataset.labels[np.isin(flat_class_labels,
                                                 selected_labels)]
        unique_targets, inverse = np.unique(filtered_labels,
                                            return_inverse=True)
        continuous_indices = inverse.reshape(filtered_labels.shape)
        dataset.labels = continuous_indices
