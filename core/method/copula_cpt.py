import numpy as np
import pandas as pd
import torch
from copulae.core import pseudo_obs
import torch.nn as nn
from tqdm import trange

class CP(nn.Module):
    def __init__(self, dimension, epsilon):
        super(CP, self).__init__()
        self.alphas = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon
        self.relu = torch.nn.ReLU()

    def forward(self, pseudo_data):
        coverage = torch.mean(
            torch.relu(
                torch.prod(torch.sigmoid((self.alphas - pseudo_data) * 1000), dim=1)
            )
        )
        return torch.abs(coverage - 1 + self.epsilon)


def search_alpha(alpha_input, epsilon, epochs=500):
    # pseudo_data = torch.tensor(pseudo_obs(alpha_input))
    pseudo_data = torch.tensor(alpha_input)
    dim = alpha_input.shape[-1]
    cp = CP(dim, epsilon)
    optimizer = torch.optim.Adam(cp.parameters(), weight_decay=1e-4)

    with trange(epochs, desc="training", unit="epochs") as pbar:
        for i in pbar:
            optimizer.zero_grad()
            loss = cp(pseudo_data)

            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.detach().numpy())

    return cp.alphas.detach().numpy()

class copulaCPTS:
    def __init__(self, calibration_df, horizon):
        """
        Copula conformal prediction with two-step calibration.
        """
        self.cali_score = None
        self.copula_score = None
        #Nonconformity scores and Copula
        self.horizon = horizon
        self.split_cali(calibration_df)
        self.results_dict = {}

    def split_cali(self, calibration_df, split=0.6, seed=42):
        df_cali = calibration_df.sample(frac=split, random_state=seed)
        df_copula = calibration_df.drop(df_cali.index)
        h_cols = [i for i in range(self.horizon)]
        self.cali_score = df_cali[h_cols].to_numpy(dtype=float)
        self.copula_score = df_copula[h_cols].to_numpy(dtype=float)
    

    def predict(self, epsilon=0.1):

        scores = self.copula_score
        # print(scores)
        alphas = []
        print(self.cali_score.shape)
        for i in range(scores.shape[0]):
            a = (scores[i] > self.cali_score).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)

        threshold = search_alpha(alphas, epsilon, epochs=800)

        mapping_shape = self.cali_score.shape[0]
        mapping = {
            i: sorted(self.cali_score[:, i].tolist()) for i in range(alphas.shape[1])
        }

        quantile = []
        mapping_shape = self.cali_score.shape[0]

        for i in range(alphas.shape[1]):
            idx = int(threshold[i] * mapping_shape) + 1
            if idx >= mapping_shape:
                idx = mapping_shape - 1
            quantile.append(mapping[i][idx])

        radius = np.array(quantile)

        self.results_dict[epsilon] = {"radius": radius}
        print(radius.shape)

        return radius
    
    def calc_area(self, radius):

        area = sum([np.pi * r**2 for r in radius])

        return area

    def calc_area_l1(self, radius):

        area = sum([2 * r**2 for r in radius])

        return area

    def calc_coverage(self, radius, y_pred, y_test):

        testnonconformity = torch.norm((y_pred - y_test), p=2, dim=-1).detach().numpy()

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage