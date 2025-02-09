import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os
import itertools
from functools import partial
from pathlib import Path
sys.path.append(os.path.abspath("."))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_percentage_error as mean_squared_error
# https://www.analyticsvidhya.com/blog/2021/10/evaluation-metric-for-regression-models/#h-list-of-top-13-evaluation-metrics
from src.FedFuzzyKNN.FedFKNNreg import FuzzyKNNreg, CliFuzzyKNNreg, ServerFuzzyKNNreg
from src.FedFuzzyKNN.KNNreg import ClientKNNreg, ServerKNNreg, KNNreg
from scipy.spatial import distance
from datasets.load_dataset import read_json_dataset, generate_one_client

def test_regression(dataset_name = "autoMPG6", alpha = 0.5, nclients = 5, n_rep = 5, k = 3, m = 2, datasets_path="datasets/regression",normalize=False):

    federated_mse = []
    local_mse = []
    central_mse = []
    knn_mse = []
    central_nonf_mse = []

    for i in range(n_rep):
        dataset_path = Path(datasets_path) / dataset_name / f"{dataset_name}_a{alpha}_n{nclients}_f{i + 1}-{n_rep}.json"
        # dataset_path = Path(datasets_path)  / dataset_name / dataset_name
        clients_datasets = read_json_dataset(dataset_name, dataset_path,normalize)
        one_client = generate_one_client(clients_datasets)
        X_test = one_client[1]
        y_test = one_client[3]

        # FL FkNN
        # print("-- FL FkNN --")
        clients_fknn = []
        for i in range(nclients):
            fknn = CliFuzzyKNNreg(k=k, m=m)
            fknn.fit(clients_datasets[i][0], clients_datasets[i][2])
            clients_fknn.append(fknn)

        # Set up the server
        sfknn = ServerFuzzyKNNreg(k=k, m=m, clients=clients_fknn)
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
        fknn = FuzzyKNNreg(k=k, m=m)
        fknn.fit(one_client[0],one_client[2])
        y_pred = fknn.predict(X_test)
        central_mse.append(mean_squared_error(y_true, y_pred))

        # KNN
        # print("-- FL kNN --")
        clients_knn = []
        for i in range(nclients):
            knn = ClientKNNreg(k=k)
            knn.fit(clients_datasets[i][0], clients_datasets[i][2])
            clients_knn.append(knn)

        sknn = ServerKNNreg(k=k, clients=clients_knn)
        y_pred = sknn.predict(X_test)
        knn_mse.append(mean_squared_error(y_true, y_pred))

        # Centralized
        # print("-- CL kKNNreg --")
        knnreg = KNNreg(k=k)
        knnreg.fit(one_client[0],one_client[2])
        y_pred = knnreg.predict(X_test)
        central_nonf_mse.append(mean_squared_error(y_true, y_pred))

    print(f"{n_rep} folds MSE (FL): {np.average(federated_mse):.3f} \u00B1 {np.std(federated_mse):.3f}")
    print(f"{n_rep} folds MSE (LL): {np.average(local_mse):.3f} \u00B1 {np.std(local_mse):.3f}")
    print(f"{n_rep} folds MSE (CL): {np.average(central_mse):.3f} \u00B1 {np.std(central_mse):.3f}")
    print(f"{n_rep} folds MSE (FNF): {np.average(knn_mse):.3f} \u00B1 {np.std(knn_mse):.3f}")
    print(f"{n_rep} folds MSE (NF): {np.average(central_nonf_mse):.3f} \u00B1 {np.std(central_nonf_mse):.3f}")
    # one_client = generate_one_client(clients_datasets)

    return [[np.average(federated_mse),np.std(federated_mse)],
            [np.average(local_mse),np.std(local_mse)],
            [np.average(central_mse),np.std(central_mse)],
            [np.average(knn_mse),np.std(knn_mse)],
            [np.average(central_nonf_mse),np.std(central_nonf_mse)]]


# Example usage
if __name__ == "__main__":
    # test_regression()
    minkowski15 = partial(distance.minkowski, p=1.5)
    # p = 0.25 / 0.5 / 4 # 1 manahatan 2 euclidean
    distance_functions = [distance.euclidean, distance.cityblock, minkowski15]
    columns = ['Dataset','FL_Acc', 'FL_Std', 'LL_Acc', 'LL_Std', 'CL_Acc', 'CL_Std', 'FNF_Acc', 'FNF_Std','CNF_Acc', 'CNF_Std']
    df = pd.DataFrame(columns=columns)
    datasets_path="./datasets/regression"
    datasets = [
    "diabetes", "machineCPU", "baseball", "dee", "autoMPG6", "autoMPG8",
    "ele-1", "forestFires", "stock", "laser", "concrete", "mortgage",
    "treasury", "ele-2", "friedman", "wizmir", "wankara", "plastic",
    "quake", "ANACALT", "abalone", "delta_ail", "compactiv", "puma32h",
    "delta_elv", "tic", "ailerons", "pole", "elevators", "california",
    "house", "mv"]
    normalize = True
    # datasets = ["abalone"]
    for dataset_name in datasets:
        print(dataset_name)
        results = test_regression(dataset_name=dataset_name,k=5,datasets_path=datasets_path,normalize=normalize)
        results = list(itertools.chain(*results))
        df = df.append(pd.DataFrame([[dataset_name]+results], columns=columns),
                       ignore_index=True)
        df.to_csv("results_regression_MAPE.csv")

    print(df)
    df.to_csv("results_regression_MAPE.csv")
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6313426&tag=1
    # https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1698_2014-Derrac-INS.pdf.pdf
    # https://sci2s.ugr.es/keel/pdf/algorithm/articulo/2007-FSS-Sarkar.pdf

