"""
Author: 2022 Anna M. Puszkarska
SPDX-License-Identifier: Apache-2.0
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn import metrics
from typing import List, Optional


def get_plot(y_test: List[np.ndarray],
             y_pred: np.ndarray,
             path_to_figs: Path,
             name: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.set_context("paper", font_scale=1.7)

    plt.scatter(y_pred[0], y_test[0], c='r', s=60, edgecolors=(0, 0, 0))
    plt.scatter(y_pred[1], y_test[1], c='m', s=60, edgecolors=(0, 0, 0))

    mse1 = metrics.mean_squared_error(y_pred[0], y_test[0])
    mse2 = metrics.mean_squared_error(y_pred[1], y_test[1])

    ax.set_aspect(1)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    plt.xlabel(r"Model predicted $\log_{10} EC50$ [M]")
    plt.ylabel(r"Experimentally measured $\log_{10} EC50$ [M]")

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.legend(loc="best", labels=['GCGR', 'GLP-1R'])

    ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title(f'Multi-task model performance\n'
                 f'RMSE1={np.sqrt(mse1).round(2)}'
                 f'\nRMSE2={np.sqrt(mse2).round(2)}')

    plt.tight_layout()
    plt.show()
    fig.savefig(str(Path(path_to_figs,
                         f'multi-task_model_predictions'
                         f'_{name}').with_suffix('.png')), dpi=300)
    plt.close(fig=fig)


def get_voting_reg_plot(predictions: np.ndarray,
                        avg_votes: List,
                        y_test: List,
                        s: bool,
                        name: str,
                        out_path=None,
                        ):
    num_votes = np.asarray(predictions).shape[0]
    names = list(map(lambda i: f'model{i}',
                     range(num_votes)))

    fig = plt.figure(figsize=(10, 5))
    sns.set_context("paper", font_scale=1.4)
    for x in range(0, np.asarray(predictions).shape[0]):
        plt.plot(predictions[x], 'o', label=names[x])

    plt.plot(avg_votes, 'r*', label='average')
    plt.plot(y_test, 'bd', label='y true')
    plt.ylabel('Prediction')
    plt.xlabel('Test samples')
    plt.legend(loc="best", ncol=2, bbox_to_anchor=(1, 0.5))
    plt.title('Comparison of individual\npredictions '
              'with the average ensemble prediction.')
    plt.xticks(np.arange(0, len(predictions[0]), 1),
               np.arange(1, len(predictions[0]) + 1, 1))
    plt.tight_layout()
    if s is True:
        fig.savefig(str(Path(out_path,
                             f'regressors_votes_{name}').with_suffix('.png')), dpi=300)
    plt.close(fig=fig)


def get_training_curves(model_history,
                        s: bool,
                        out_path: Optional[Path] = None,
                        name: Optional[str] = None
                        ):
    fig = plt.figure(figsize=(5, 3))
    sns.set_context("paper", font_scale=1.5)

    plt.plot(model_history.history['loss'], c='b')
    plt.plot(model_history.history['val_loss'], c='r')
    ax = plt.gca()
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss [MSE]', fontsize=13)
    ax.set_title('Model training')
    ax.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    if s:
        fig.savefig(str(Path(out_path, name).with_suffix('.png')), dpi=300)
        plt.close(fig=fig)
