# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dataset definitions for machine learning benchmarks."""

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd

# Synthetic dataset for regression is generated as a noisy sine wave.
# For more details please refer to the "Neural Network Classifier & Regressor" tutorial
# in Qiskit Machine Learning.
DATASET_SYNTHETIC_REGRESSION_FEATURES = np.array(
    [
        [-1.0],
        [-0.61017247],
        [0.3437984],
        [-0.57425039],
        [-0.15067904],
        [0.54669445],
        [-0.16936095],
        [-0.90157044],
        [0.37708312],
        [0.99143454],
        [-0.49085348],
        [0.53117172],
        [1.0],
        [0.02066802],
        [-0.6564818],
        [-0.76721681],
        [-0.64475023],
        [0.43324292],
        [0.71212452],
        [0.80997021],
        [-0.07795483],
        [0.32083733],
        [0.17454994],
        [0.56743491],
        [-0.12885006],
    ]
)

DATASET_SYNTHETIC_REGRESSION_LABELS = np.array(
    [
        [-0.07742259],
        [-1.01861621],
        [0.9456131],
        [-1.15321842],
        [-0.60170763],
        [1.03743838],
        [-0.53921747],
        [-0.53976918],
        [0.99681555],
        [0.29210661],
        [-1.09853086],
        [0.89345683],
        [0.12046636],
        [-0.03882522],
        [-0.88622102],
        [-0.83758335],
        [-0.98542812],
        [1.07553076],
        [0.70927951],
        [0.64405885],
        [-0.36920025],
        [0.89642704],
        [0.41763132],
        [1.13432333],
        [-0.56352804],
    ]
)


def load_ccpp() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads the Combined Cycle Power Plant dataset. See the `UCI Machine Learning Repository
    <https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant>`_ web site for more
    details.

    Returns:
        a tuple with features and labels as numpy arrays.
    """
    # the benchmarks are run in a temp directory, but we have to reference a file with the dataset.
    abs_path = pathlib.Path(__file__).parent.resolve()
    ccpp_df = pd.read_csv(f"{abs_path}/CCPP_data.csv")
    ccpp_features = ccpp_df[["AT", "V", "AP", "RH"]].to_numpy()
    ccpp_labels = ccpp_df["PE"].to_numpy()
    return ccpp_features, ccpp_labels
