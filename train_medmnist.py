import random
import time

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights

from proyecto.medmnist_data import MedMnistData
from proyecto.model_trainer import ModelTrainer

if __name__ == "__main__":

    # Model parameters
    num_classes = 6
    base_model = 'resnet_50_finetune'
    preprocess_resnet = False
    # Optimizer parameters
    learning_rate = 0.001
    momentum = 0.9
    # Scheduler parameters
    gamma = 0.9
    step_size = 10
    # Training parameters
    num_epochs = 10
    trim_train_dataset = True
    trim_test_dataset = True
    # Regularization parameters
    ewc_lambda = 0
    rehearsal = 0
    dropout = 0.4

    # Load and preprocess your datasets
    dataset_names = ['PathMNIST', 'DermaMNIST', 'BloodMNIST', 'TissueMNIST',
                     'OrganCMNIST', 'OrganAMNIST', 'OrganSMNIST']
    random.Random().shuffle(dataset_names)
    datasets = []
    smaller_size = 1e6
    smaller_test_size = 0

    for name in dataset_names:
        dataset = MedMnistData(name.lower())
        dataset.select_n_classes(num_classes)
        dataset_size = len(dataset.train_dataset.imgs)

        if dataset_size < smaller_size:
            smaller_size = dataset_size
            smaller_test_size = len(dataset.test_dataset.imgs)

        if preprocess_resnet:
            weights = ResNet50_Weights.DEFAULT
            preprocess = weights.transforms()
            imgs = dataset.train_dataset.imgs
            n_samples = len(imgs)

            # Create an empty tensor to hold the transformed images
            transformed_images = torch.empty(n_samples, 3, 224, 224)

            for i in range(n_samples):
                img = Image.fromarray(imgs[i])
                img = preprocess(img)
                transformed_images[i] = img

            dataset.train_dataset.imgs = transformed_images

        datasets.append(dataset)

    datasets = sorted(datasets, key=lambda dataset: len(
        dataset.test_dataset.imgs), reverse=True)
    random_index = random.randint(0, len(datasets) - 1)
    # Pop the dataset at the random index
    random_dataset = datasets.pop(random_index)
    # Append the random dataset to the end of the list
    datasets.append(random_dataset)

    for dataset in datasets:
        if trim_train_dataset:
            imgs = dataset.train_dataset.imgs
            labels = dataset.train_dataset.labels
            n_samples = len(imgs)
            # Randomly sample a subset of the data
            random_idx = np.random.choice(n_samples, smaller_size,
                                          replace=False)
            # Concatenate the subset to your dataset
            dataset.train_dataset.imgs = imgs[random_idx]
            dataset.train_dataset.labels = labels[random_idx]

        if trim_test_dataset:
            imgs = dataset.test_dataset.imgs
            labels = dataset.test_dataset.labels
            n_samples = len(imgs)
            # Randomly sample a subset of the data
            random_idx = np.random.choice(n_samples, smaller_test_size,
                                          replace=False)
            # Concatenate the subset to your dataset
            dataset.test_dataset.imgs = imgs[random_idx]
            dataset.test_dataset.labels = labels[random_idx]

    # Initialize the class
    model_trainer = ModelTrainer(num_classes=num_classes,
                                 model_name=base_model,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 gamma=gamma,
                                 step_size=step_size,
                                 dropout=dropout)

    # Train on the datasets specified parameters
    accuracy, f1_macro, f1_weighted, ari, nmi = \
        model_trainer.train_on_datasets(datasets=datasets[:-1],
                                        num_epochs=num_epochs,
                                        eval_dataset=datasets[-1],
                                        rehearsal=rehearsal,
                                        ewc_lambda=ewc_lambda)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_name = 'medmnist_model_' + timestr + '.onnx'
    model_trainer.save_model(model_name, 'onnx')

    report = f"""\
{timestr}
# Model parameters
num_classes = {num_classes}
base_model = '{base_model}'
preprocess_resnet = {preprocess_resnet}

# Optimizer parameters
learning_rate = {learning_rate}
momentum = {momentum}

# Scheduler parameters
gamma = {gamma}
step_size = {step_size}

# Training parameters
num_epochs = {num_epochs}
trim_train_dataset = {trim_train_dataset}
trim_test_dataset = {trim_test_dataset}

# Regularization parameters
ewc_lambda = {ewc_lambda}
rehearsal = {rehearsal}
dropout = {dropout}

# Datasets order
dataset_names = {dataset_names}

# Model metrics
accuracy = {accuracy}
f1_macro = {f1_macro}
f1_weighted = {f1_weighted}
ari = {ari}
nmi = {nmi}

# Output model name
model_name = '{model_name}'

--------------------------------------------------------------------
"""

    with open('training_report.txt', 'a+') as f:
        f.write(report)
