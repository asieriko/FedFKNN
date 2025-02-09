import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath("."))
from datasets.load_dataset import read_keel_dat
import json
from fedartml import SplitAsFederatedData
from sklearn.model_selection import train_test_split

ds = [
    "abalone", "ailerons", "ANACALT", "autoMPG6", "autoMPG8", "baseball",
    "california", "compactiv", "concrete", "dee", "delta_ail", "delta_elv",
    "diabetes", "ele-1", "ele-2", "elevators", "forestFires", "friedman",
    "house", "laser", "machineCPU", "mortgage", "mv", "plastic", "pole",
    "puma32h", "quake", "stock", "tic", "treasury", "wankara", "wizmir"
]

# configure partitions
num_clients_ls = [5] #[3,5,10]
alpha_ls=[0.5]#[0.005,0.01,0.05,0.5,1,10,50,100,500,1000,10000,100000]#[0.5,500]
folds_ls = [5]#[5,10]#[5,10]

if len(sys.argv) > 1:
    ds = [sys.argv[1]]


for dataset_name in ds:
  for folds in folds_ls:
    for num_clients in num_clients_ls:
      for alpha_feat_split in alpha_ls:
        # print(alpha_feat_split, num_clients)
        path = Path(f"/home/asier/NUP/Ikerketa/Projects/FedFuzzyKNN/datasets/regression/{dataset_name}")
        Path.mkdir(path, parents=True, exist_ok=True)
        try:
          # load dataset
          dataset = read_keel_dat(path/dataset_name)
          x_train = dataset.data
          y_train = dataset.target

          distances_all_folds = {}
          for i in range(folds):
            # print(f"Fold {i+1}")
            partitions = {}
            my_federater = SplitAsFederatedData(random_state=i+10)
            clients_glob, list_ids_sampled, miss_class_per_node, distances = my_federater.create_clients(image_list=x_train, label_list=y_train, num_clients=num_clients,
              prefix_cli='Local_node',feat_skew_method="hist-dirichlet",alpha_feat_split=alpha_feat_split,method='no-label-skew')
            for ci, clients_ids in enumerate(list_ids_sampled["without_class_completion"]):
              train_ids, test_ids = train_test_split(clients_ids)
              partitions[f"C{ci+1}"] = {"train":train_ids,"test":test_ids}
            if not distances_all_folds:
              distances_all_folds = distances["without_class_completion"]
            else:
              for k in distances["without_class_completion"]:
                distances_all_folds[k] += distances["without_class_completion"][k]
            with open(path / f"{dataset_name}_a{alpha_feat_split}_n{num_clients}_f{i+1}-{folds}.json","w") as f:
              json.dump(partitions, f, indent=4)
            with open(path / f"{dataset_name}_a{alpha_feat_split}_n{num_clients}_f{i+1}-{folds}.txt","a") as f:
              f.write(f"{miss_class_per_node}")
              json.dump(distances["without_class_completion"], f, indent=4)
          for k in distances_all_folds:
                distances_all_folds[k] = distances_all_folds[k]/folds
          with open(path / f"{dataset_name}_a{alpha_feat_split}_n{num_clients}_f{folds}_distAVG.txt","w") as f:
            json.dump(distances_all_folds, f, indent=4)
          print(f"{alpha_feat_split}: {distances_all_folds['jensen-shannon']}")
        except Exception as e:
          print("Error:", dataset_name)
          print(e)
