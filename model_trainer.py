import copy
import os
import time
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, \
    accuracy_score, f1_score
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm


class ModelTrainer:
    def __init__(self, num_classes, model_name, learning_rate=0.001,
                 momentum=0.9, step_size=3, gamma=0.9, dropout=0):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.dropout = dropout
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.params = None
        self.model = self.initialize_model(model_name)

        if self.params is None:
            self.params = self.model.parameters()

        self.optimizer = optim.SGD(self.params, lr=learning_rate,
                                   momentum=momentum)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma)

        # EWC
        self.fisher_dict = {}
        self.optpar_dict = {}
        self.trained_tasks = []

    def initialize_model(self, model_name='resnet50'):
        if model_name == 'resnet_50_finetune':
            return self.get_resnet_model_finetune()

    def get_resnet_model_finetune(self):
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features

        if self.dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(num_features, self.num_classes)
            )
        else:
            model.fc = nn.Linear(num_features, self.num_classes)
        self.params = model.fc.parameters()

        return model.to(self.device)

    def train_model(self, train_dataloader, test_dataloader, num_epochs,
                    ewc_lambda):
        since = time.time()
        task_type = train_dataloader.dataset.info['task']

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir,
                                                  'best_model_params.pt')
            torch.save(self.model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                # Each epoch has a training and validation phase
                for phase in ['train']:
                    if phase == 'train':
                        self.model.train()  # Set model to training mode
                        dataloader = train_dataloader
                    else:
                        self.model.eval()  # Set model to evaluate mode
                        dataloader = test_dataloader

                    running_loss = 0.0
                    running_loss_crit = 0.0
                    running_loss_ewc = 0.0
                    running_corrects = 0

                    for inputs, targets in dataloader:
                        inputs = inputs.to(self.device)

                        if task_type == 'multi-label, binary-class':
                            labels = targets.to(torch.float32)
                            labels = labels.to(self.device)
                            criterion = nn.BCEWithLogitsLoss()
                        else:
                            labels = targets.squeeze().long()
                            labels = labels.to(self.device)
                            criterion = nn.CrossEntropyLoss()

                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model(inputs)
                            _, predicted = torch.max(outputs, 1)
                            loss_crit = criterion(outputs, labels)
                            loss_ewc = self.calculate_ewc_loss(ewc_lambda)
                            loss = loss_crit + loss_ewc

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                self.optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_loss_crit += loss_crit * inputs.size(0)
                        running_loss_ewc += loss_ewc * inputs.size(0)
                        running_corrects += (predicted == labels).sum().item()

                    if phase == 'train':
                        self.scheduler.step()

                    num_samples = len(dataloader.dataset.imgs)
                    epoch_loss = running_loss / num_samples
                    epoch_loss_crit = running_loss_crit / num_samples
                    epoch_loss_ewc = running_loss_ewc / num_samples
                    epoch_acc = running_corrects / num_samples

                    print(f'{phase} Loss: {epoch_loss:.4f}'
                          f' Loss crit: {epoch_loss_crit:.4f}'
                          f' Loss ewc: {epoch_loss_ewc:.4f}'
                          f' Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(self.model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m'
              f' {time_elapsed % 60:.0f}s')
        print(f'Best Acc on same dataset: {best_acc:4f}')

    def train_on_datasets(self, datasets, num_epochs, eval_dataset,
                          rehearsal=0, ewc_lambda=0):
        eval_dataloader = DataLoader(eval_dataset.test_dataset, batch_size=32,
                                     shuffle=True)
        print(f"Before training:")
        self.evaluate_model(eval_dataloader)
        accuracy, f1_macro, f1_weighted, ari, nmi = None, None, None, None, None

        for i, dataset in enumerate(datasets):
            print(f'\n[{i}] train on {dataset.data_flag}')
            original_dataset = dataset.train_dataset
            train_dataset = original_dataset

            # If rehearsal>0, carry over part of the previous dataset
            if rehearsal > 0 and i > 0:
                train_dataset = self.add_data_from_previous_task(
                    datasets, i, rehearsal)

            dataloader = DataLoader(train_dataset, batch_size=32,
                                    shuffle=True, drop_last=True)
            test_dataloader = DataLoader(dataset.test_dataset, batch_size=32,
                                         shuffle=True, drop_last=True)
            self.train_model(dataloader, test_dataloader, num_epochs,
                             ewc_lambda)

            if ewc_lambda > 0:
                # ewc_dataloader = DataLoader(train_dataset, batch_size=32,
                #                             shuffle=True, drop_last=True)
                self.on_task_update(test_dataloader)

            self.evaluate_model(test_dataloader)
            accuracy, f1_macro, f1_weighted, ari, nmi = \
                self.evaluate_model(eval_dataloader)

        return accuracy, f1_macro, f1_weighted, ari, nmi

    def evaluate_model(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        predicted_all = []
        labels_all = []
        task_type = dataloader.dataset.info['task']

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                if task_type == 'multi-label, binary-class':
                    labels = labels.to(torch.float32)
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    labels = labels.squeeze().long()
                    criterion = nn.CrossEntropyLoss()

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                predicted_all.append(predicted.cpu())
                labels_all.append(labels.cpu())
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        predicted_all = np.concatenate(predicted_all)
        labels_all = np.concatenate(labels_all)
        ari = adjusted_rand_score(labels_all, predicted_all)
        nmi = normalized_mutual_info_score(labels_all, predicted_all)
        epoch_loss = running_loss / len(dataloader.dataset.imgs)
        accuracy_sk = accuracy_score(labels_all, predicted_all, normalize=True)
        f1_macro = f1_score(labels_all, predicted_all, average='macro')
        f1_weighted = f1_score(labels_all, predicted_all, average='weighted')

        print(f"Accuracy on {dataloader.dataset.flag}: "
              f"{correct}/{total} = {accuracy}")
        print(f"Accuracy sklearn on {dataloader.dataset.flag}: {accuracy_sk}")
        print(f"F1 (macro) score on {dataloader.dataset.flag}: {f1_macro}")
        print(
            f"F1 (weighted) score on {dataloader.dataset.flag}: {f1_weighted}")
        print(f"ARI on {dataloader.dataset.flag}: {ari}")
        print(f"NMI on {dataloader.dataset.flag}: {nmi}")
        print(f"Average loss {epoch_loss}")
        return accuracy, f1_macro, f1_weighted, ari, nmi

    def on_task_update(self, dataloader):
        task_type = dataloader.dataset.info['task']
        self.model.train()
        self.optimizer.zero_grad()

        for inputs, labels in dataloader:
            with torch.set_grad_enabled(True):

                if task_type == 'multi-label, binary-class':
                    labels = labels.to(torch.float32)
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    labels = labels.squeeze().long()
                    criterion = nn.CrossEntropyLoss()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

        task_id = dataloader.dataset.flag
        self.optpar_dict[task_id] = {}
        self.fisher_dict[task_id] = {}
        self.trained_tasks.append(task_id)

        # gradients accumulated can be used to calculate fisher
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.optpar_dict[task_id][name] = param.data.clone()
                self.fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

    def calculate_ewc_loss(self, ewc_lambda):
        ewc_loss = 0

        for id in self.trained_tasks:
            for name, param in self.model.named_parameters():
                if not name in self.fisher_dict[id]:
                    continue
                fisher = self.fisher_dict[id][name]
                optpar = self.optpar_dict[id][name]
                ewc_loss += \
                    (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

        return ewc_loss

    def add_data_from_previous_task(self, datasets, index, rehearsal):
        train_dataset = copy.deepcopy(datasets[index].train_dataset)
        n_samples = len(train_dataset.imgs)
        n_subset = int(n_samples * rehearsal)

        for i in range(index):
            imgs = train_dataset.imgs
            labels = train_dataset.labels
            prev_dataset = datasets[i]
            n_samples_prev = len(prev_dataset.train_dataset.imgs)

            if n_samples_prev < n_subset:
                curr_subset = n_samples_prev
            else:
                curr_subset = n_subset

            # Randomly sample a subset of the data
            random_idx = np.random.choice(n_samples_prev, curr_subset,
                                          replace=False)
            print(f"imgs before {imgs.shape}, labels before {labels.shape}")

            # Concatenate the subset to your dataset
            imgs_prev = prev_dataset.train_dataset.imgs
            labels_prev = prev_dataset.train_dataset.labels
            train_dataset.imgs = np.concatenate(
                (train_dataset.imgs, imgs_prev[random_idx]), axis=0)
            train_dataset.labels = np.concatenate(
                (train_dataset.labels, labels_prev[random_idx]), axis=0)

            print(f"img subset {imgs_prev[random_idx].shape}, "
                  f"labels subset {labels_prev[random_idx].shape}")
            print(f"imgs {train_dataset.imgs.shape}, "
                  f"labels {train_dataset.labels.shape}")

        return train_dataset

    def save_model(self, filename, model_format):
        if model_format == 'onnx':
            self.model.eval()
            x = torch.randn(1, 3, 28, 28).to(self.device)
            dynamic_axes_dict = {
                'input': {
                    0: 'bs',
                    2: 'img_x',
                    3: 'img_y'
                },
                'output': {
                    0: 'bs'
                }
            }

            torch.onnx.export(
                self.model,  # model being run
                x,  # model input
                filename,
                input_names=['input'],  # the model's input names
                output_names=['output'],  # the model's output names
                export_params=True,
                dynamic_axes=dynamic_axes_dict)
