import numpy as np
import onnx
import onnxruntime
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, \
    accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from proyecto.medmnist_data import MedMnistData


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    onnx_model = onnx.load("../data/medmnist_model_20231031-001127.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(
        "../data/medmnist_model_20231031-001127.onnx", providers=["CPUExecutionProvider"])

    # Load and preprocess your datasets
    dataset_names = ['PathMNIST', 'DermaMNIST', 'BloodMNIST', 'TissueMNIST',
                     'OrganCMNIST', 'OrganAMNIST', 'OrganSMNIST']
    num_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in dataset_names:
        dataset = MedMnistData(name.lower())
        dataset.select_n_classes(num_classes)
        predicted_all = []
        labels_all = []
        correct = 0
        total = 0

        dataloader = DataLoader(dataset.test_dataset, batch_size=32,
                                     shuffle=True, drop_last=True)

        for inputs, labels in tqdm(dataloader):
            task_type = dataloader.dataset.info['task']
            inputs = inputs.to(device)
            labels = labels.to(device)

            if task_type == 'multi-label, binary-class':
                labels = labels.to(torch.float32)
            else:
                labels = labels.squeeze().long()

            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
            ort_outputs = ort_session.run(None, ort_inputs)

            predicted = ort_outputs[0].argmax(axis=1)
            predicted_all.append(predicted)
            labels_all.append(labels.cpu().numpy())
            correct += (predicted == labels.cpu().numpy()).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        predicted_all = np.concatenate(predicted_all)
        labels_all = np.concatenate(labels_all)
        ari = adjusted_rand_score(labels_all, predicted_all)
        nmi = normalized_mutual_info_score(labels_all, predicted_all)
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
