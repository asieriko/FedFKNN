import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os
import itertools
from pathlib import Path
sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.FedFuzzyKNN.FedFuzzyKNN import FuzzyKNN, CliFuzzyKNN, ServerFuzzyKNN
from src.FedFuzzyKNN.KNN import ClientKNN, ServerKNN
from datasets.load_dataset import read_json_dataset, generate_one_client

def test_regression(dataset_name = "autoMPG6", alpha = 0.5, nclients = 5, n_rep = 5, k = 3, m = 2, datasets_path="datasets/regression"):

    federated_mse = []
    local_mse = []
    central_mse = []
    knn_mse = []

    for i in range(n_rep):
        dataset_path = Path(datasets_path) / dataset_name / f"{dataset_name}_a{alpha}_n{nclients}_f{i + 1}-{n_rep}.json"
        # dataset_path = Path(datasets_path)  / dataset_name / dataset_name
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

        federated_mse.append(mean_squared_error(y_true, y_pred))

        # Clients Locally
        # print("-- LL FkNN --")
        local_mse_fold = []
        for client in clients_fknn:
            y_pred = client.predict(X_test)
            local_mse_fold.append(mean_squared_error(y_true, y_pred))
        local_mse.append(np.average(local_mse_fold))

        # Centralized
        # print("-- CL FkKNN --")
        fknn = FuzzyKNN(k=k, m=m)
        fknn.fit(one_client[0],one_client[2])
        y_pred = fknn.predict(X_test)
        central_mse.append(mean_squared_error(y_true, y_pred))

        # KNN
        # print("-- FL kNN --")
        clients_knn = []
        for i in range(nclients):
            knn = ClientKNN(k=k)
            knn.fit(clients_datasets[i][0], clients_datasets[i][2])
            clients_knn.append(knn)

        sknn = ServerKNN(k=k, clients=clients_knn)
        y_pred = sknn.predict(X_test)
        knn_mse.append(mean_squared_error(y_true, y_pred))

    print(f"{n_rep} folds MSE (FL): {np.average(federated_mse):.3f} \u00B1 {np.std(federated_mse):.3f}")
    print(f"{n_rep} folds MSE (LL): {np.average(local_mse):.3f} \u00B1 {np.std(local_mse):.3f}")
    print(f"{n_rep} folds MSE (CL): {np.average(central_mse):.3f} \u00B1 {np.std(central_mse):.3f}")
    print(f"{n_rep} folds MSE (NF): {np.average(knn_mse):.3f} \u00B1 {np.std(knn_mse):.3f}")
    # one_client = generate_one_client(clients_datasets)

    return [[np.average(federated_mse),np.std(federated_mse)],
            [np.average(local_mse),np.std(local_mse)],
            [np.average(central_mse),np.std(central_mse)],
            [np.average(knn_mse),np.std(knn_mse)]]


# Example usage
if __name__ == "__main__":
    # test_regression()

    columns = ['Dataset','FL_Acc', 'FL_Std', 'LL_Acc', 'LL_Std', 'CL_Acc', 'CL_Std', 'NF_Acc', 'NF_Std']
    df = pd.DataFrame(columns=columns)
    datasets_path="/home/asier/NUP/Ikerketa/Projects/FedFuzzyKNN/datasets/regression"
    datasets = [
        "abalone", "ailerons", "ANACALT", "autoMPG6", "autoMPG8", "baseball",
        "california", "compactiv", "concrete", "dee", "delta_ail", "delta_elv",
        "diabetes", "ele-1", "ele-2", "elevators", "forestFires", "friedman",
        "house", "laser", "machineCPU", "mortgage", "mv", "plastic", "pole",
        "puma32h", "quake", "stock", "tic", "treasury", "wankara", "wizmir"
    ]
    # datasets = ["abalone"]
    for dataset_name in datasets:
        print(dataset_name)
        results = test_regression(dataset_name=dataset_name,k=5,datasets_path=datasets_path)
        results = list(itertools.chain(*results))
        df = df.append(pd.DataFrame([[dataset_name]+results], columns=columns),
                       ignore_index=True)

    print(df)
    df.to_csv("results_regression.csv")
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6313426&tag=1
    # https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1698_2014-Derrac-INS.pdf.pdf
    # https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2007-FSS-Sarkar.pdf

