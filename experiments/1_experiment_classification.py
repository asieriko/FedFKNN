import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os
import itertools
from pathlib import Path
sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.FedFuzzyKNN.FedFuzzyKNN import FuzzyKNN, CliFuzzyKNN, ServerFuzzyKNN
from src.FedFuzzyKNN.KNN import ClientKNN, ServerKNN
from datasets.load_dataset import read_json_dataset, generate_one_client

def test_classification(dataset_name ="iris", alpha = 0.5, nclients = 3, n_rep = 5, k = 3, m = 2, datasets_path="datasets"):

    federated_accuracies = []
    local_accuracies = []
    central_accuracies = []
    knn_accuracies = []

    for i in range(n_rep):
        dataset_path = Path(datasets_path) / dataset_name / f"{dataset_name}_a{alpha}_n{nclients}_f{i + 1}-{n_rep}.json"
        clients_datasets = read_json_dataset(dataset_name, dataset_path)
        one_client = generate_one_client(clients_datasets)
        X_test = one_client[1]
        y_test = one_client[3]

        # FL FkNN
        # print("-- FL FkNN --")
        clients_fknn = []
        for i in range(nclients):
            fknn = CliFuzzyKNN(k=k, m=m)
            fknn.fit(clients_datasets[i][0], clients_datasets[i][2])
            clients_fknn.append(fknn)

        # Set up the server
        sfknn = ServerFuzzyKNN(k=k, m=m, clients=clients_fknn)
        y_pred = sfknn.predict(X_test)
        y_true = y_test

        # Evaluation
        federated_accuracies.append(accuracy_score(y_true, y_pred))

        # Clients Locally
        # print("-- LL FkNN --")
        clients_accuracies_fold = []
        for client in clients_fknn:
            y_pred = client.predict(X_test)
            clients_accuracies_fold.append(accuracy_score(y_true, y_pred))
        local_accuracies.append(np.average(clients_accuracies_fold))

        # Centralized
        # print("-- CL FkKNN --")
        fknn = FuzzyKNN(k=k, m=m)
        fknn.fit(one_client[0],one_client[2])
        y_pred = fknn.predict(X_test)
        central_accuracies.append(accuracy_score(y_true, y_pred))

        # KNN
        # print("-- FL kNN --")
        clients_knn = []
        for i in range(nclients):
            knn = ClientKNN(k=k)
            knn.fit(clients_datasets[i][0], clients_datasets[i][2])
            clients_knn.append(knn)

        sknn = ServerKNN(k=k, clients=clients_knn)
        y_pred = sknn.predict(X_test)
        knn_accuracies.append(accuracy_score(y_true, y_pred))

    print(f"{n_rep} folds accuracy (FL): {np.average(federated_accuracies):.3f} \u00B1 {np.std(federated_accuracies):.3f}")
    print(f"{n_rep} folds accuracy (LL): {np.average(local_accuracies):.3f} \u00B1 {np.std(local_accuracies):.3f}")
    print(f"{n_rep} folds accuracy (CL): {np.average(central_accuracies):.3f} \u00B1 {np.std(central_accuracies):.3f}")
    print(f"{n_rep} folds accuracy (NF): {np.average(knn_accuracies):.3f} \u00B1 {np.std(knn_accuracies):.3f}")
    # one_client = generate_one_client(clients_datasets)

    return [[np.average(federated_accuracies),np.std(federated_accuracies)],
            [np.average(local_accuracies),np.std(local_accuracies)],
            [np.average(central_accuracies),np.std(central_accuracies)],
            [np.average(knn_accuracies),np.std(knn_accuracies)]]


# Example usage
if __name__ == "__main__":
    columns = ['Dataset', 'FL_Acc', 'FL_Std', 'LL_Acc', 'LL_Std', 'CL_Acc', 'CL_Std', 'NF_Acc', 'NF_Std']
    df = pd.DataFrame(columns=columns)
    datasets_path = "/home/asier/NUP/Ikerketa/Projects/FederatedRuleLearning/datasets"
    datasets = ["iris"]
    ds = ['adult', 'balance', 'coil2000', 'dermatology', 'german', 'ionosphere', 'lymphography', 'mammographic',
          'nursery', 'penbased', 'satimage', 'spambase', 'thyroid', 'wdbc', 'wisconsin', 'appendicitis', 'car', 'crx',
          'flare', 'heart', 'iris', 'monk-2', 'optdigits', 'phoneme', 'ring', 'segment', 'spectfheart', 'titanic',
          'wine', 'australian', 'chess', 'housevotes', 'mushroom', 'page-blocks', 'pima', 'saheart', 'shuttle',
          'splice', 'twonorm']
    datasets = ['appendicitis', 'balance', 'banana', 'breast', 'bupa', 'car', 'contraceptive',  # 'ecoli',
                'glass', 'haberman',  # 'hayes-roth',
                'iris', 'mammographic',  # 'monk-2', 'newthyroid', #'nursery',
                'phoneme', 'pima',  # 'post-operative',
                'saheart',  # 'shuttle',
                'tic-tac-toe', 'titanic', 'wisconsin']
    for dataset_name in datasets:
        print(dataset_name)
        results = test_classification(dataset_name=dataset_name, k=5, datasets_path=datasets_path)
        results = list(itertools.chain(*results))
        df = df.append(pd.DataFrame([[dataset_name] + results], columns=columns),
                       ignore_index=True)

    print(df)
    df.to_csv("results_classification.csv")
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6313426&tag=1
    # https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1698_2014-Derrac-INS.pdf.pdf
    # https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2007-FSS-Sarkar.pdf

