import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from copulae.core import pseudo_obs
import numpy as np
import torch
# import matplotlib.pyplot as plt

def gumbel_copula_loss(x, cop, data, epsilon):
    return np.fabs(cop.cdf([x] * data.shape[1]) - 1 + epsilon)


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1
            )
        )
        - 1
        + epsilon
    )


def empirical_copula_loss_new(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return (
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1
            )
        )
        - 1
        + epsilon
    )


# def mace(cov):
#     x_axis = [i/10.0 for i in range(1,10)]
#     return np.mean([abs(x_axis[8-i] - cov[i]) for i in range(9)])


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
            # if i%10 == 0:
            #     print(cp.alphas)
            #     print(loss)
            pbar.set_postfix(loss=loss.detach().numpy())

    return cp.alphas.detach().numpy()


def empirical_copula_loss(x, data, epsilon):
    pseudo_data = pseudo_obs(data)
    return np.fabs(
        np.mean(
            np.all(
                np.less_equal(pseudo_data, np.array([x] * pseudo_data.shape[1])), axis=1
            )
        )
        - 1
        + epsilon
    )


class copulaCPTS:
    def __init__(self, optional_cp_args):
        """
        Copula conformal prediction with two-step calibration.
        """
        self.window = optional_cp_args.get('window')
        
    def split_cali(self, cali_scores, split=0.6):
        size = cali_scores.shape[0]
        halfsize = int(split * size)

        idx = np.random.choice(range(size), halfsize, replace=False)

        self.cali_scores = cali_scores[idx]
        self.copula_scores = cali_scores[list(set(range(size)) - set(idx))]
    
    def predict(self, cali_scores, epsilon=0.1):
        if self.window is not None and self.window < len(cali_scores):
            windowed_cali_scores = cali_scores[-self.window:]
        else:
            windowed_cali_scores = cali_scores
        self.split_cali(windowed_cali_scores)
        self.nonconformity = self.cali_scores

        # alphas = self.nonconformity
        scores = self.copula_scores # shape (N x H)
        alphas = []
        for i in range(scores.shape[0]):
            a = (scores[i] > self.nonconformity).mean(axis=0)
            alphas.append(a)
        alphas = np.array(alphas)
        # print(self.nonconformity)
        # print(f'scores shape: {scores.shape}')
        # print(f'alphas shape {alphas.shape}')
        # print(f'alphas are {alphas}')
        # print(f'alphas is {alphas}')
        num_dims = alphas.shape[1]
        
        # fig, axes = plt.subplots(1, num_dims, figsize=(5 * num_dims, 4))
        # if num_dims == 1:
        #     axes = [axes]
        # for i in range(num_dims):
        #     axes[i].hist(alphas[:, i], bins=30, alpha=0.7)
        #     axes[i].set_title(f'Alpha Distribution - Dimension {i+1}')
        #     axes[i].set_xlabel('Alpha')
        #     axes[i].set_ylabel('Frequency')
        # plt.tight_layout()
        # plt.show()

        threshold = search_alpha(alphas, epsilon, epochs=800)
        # print(f"Current threshold: {threshold}, epsilon: {epsilon}")

        mapping_shape = self.nonconformity.shape[0]
        mapping = {
            i: sorted(self.nonconformity[:, i].tolist()) for i in range(alphas.shape[1])
        }

        quantile = []
        mapping_shape = self.nonconformity.shape[0]

        for i in range(alphas.shape[1]):
            idx = int(threshold[i] * mapping_shape) + 1
            if idx >= mapping_shape:
                idx = mapping_shape - 1
            quantile.append(mapping[i][idx])

        radius = np.array(quantile)

        return radius

    def calc_area(self, radius):

        area = sum([np.pi * r**2 for r in radius])

        return area

    def calc_area_l1(self, radius):

        area = sum([2 * r**2 for r in radius])

        return area

    def calc_area_3d(self, radius):

        area = sum([4 / 3.0 * np.pi * r**3 for r in radius])

        return area

    def calc_area_1d(self, radius):

        area = sum(radius)

        return area

    def calc_coverage(self, radius, y_pred, y_test):

        testnonconformity = torch.norm((y_pred - y_test), p=2, dim=-1).detach().numpy()

        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_l1(self, radius, y_pred, y_test):
        testnonconformity = (
            torch.norm((y_pred - y_test), p=1, dim=-1).detach().numpy()
        )  # change back to p=2
        circle_covs = []
        for j in range(y_test.shape[-2]):
            circle_covs.append(testnonconformity[:, j] < radius[j])

        circle_covs = np.array(circle_covs)
        coverage = np.mean(np.all(circle_covs, axis=0))
        return coverage

    def calc_coverage_3d(self, radius, y_pred, y_test):

        return self.calc_coverage(radius, y_pred, y_test)

    def calc_coverage_1d(self, radius, y_pred, y_test):

        return self.calc_coverage(radius, y_pred, y_test)
