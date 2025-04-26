import torch
from torch.utils.data import TensorDataset
import argparse
import wandb
from sklearn.datasets import fetch_olivetti_faces  # 🔵 nuevo import

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    Load the Olivetti Faces data
    """
    
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    images, targets = data.images, data.target  # images: (400, 64, 64), targets: (400,)

    x = torch.tensor(images, dtype=torch.float32)  # (400, 64, 64)
    y = torch.tensor(targets, dtype=torch.long)    # (400,)

    n_total = x.shape[0]
    n_train = int(n_total * train_size)
    n_val = int((n_total - n_train) / 2)
    n_test = n_total - n_train - n_val

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    x_test = x[n_train+n_val:]
    y_test = y[n_train+n_val:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    test_set = TensorDataset(x_test, y_test)

    datasets = [training_set, validation_set, test_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "olivetti-faces-raw", type="dataset",
            description="raw Olivetti Faces dataset, split into train/val/test",
            metadata={"source": "sklearn.datasets.fetch_olivetti_faces",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
