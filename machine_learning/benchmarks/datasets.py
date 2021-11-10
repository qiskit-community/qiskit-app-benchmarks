# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Dataset definitions for machine learning benchmarks"""

import pathlib
import numpy as np
import pandas as pd

DATASET_SYNTHETIC_REGRESSION_FEATURES = np.array(
    [
        [-0.96719663, -0.21727823],
        [-0.90723752, 0.58081368],
        [0.37498856, -0.4353039],
        [-1.0, 0.42412708],
        [-0.44021603, -0.17820959],
        [-0.980195, 0.28240407],
        [0.18312228, -0.84245566],
        [0.05608778, -1.0],
        [-0.41079079, -0.56186934],
        [-0.21762447, -0.23754949],
        [-0.50941294, 0.85148572],
        [-0.82497265, -0.03625427],
        [-0.62816957, -0.86370768],
        [-0.4997361, 0.18054064],
        [-0.44049446, 1.0],
        [-0.1803079, 0.89166501],
        [-0.45615344, 0.43701559],
        [0.31078677, 0.80052221],
        [1.0, 0.94282486],
        [-0.32555285, -0.5833288],
    ]
)

DATASET_SYNTHETIC_REGRESSION_LABELS = np.array(
    [
        [-0.77273448],
        [-0.23856758],
        [-0.19872884],
        [-0.38635744],
        [-0.46888328],
        [-0.46507602],
        [-0.55670191],
        [-0.72319316],
        [-0.69480141],
        [-0.38828153],
        [0.14263396],
        [-0.58341376],
        [-1.0],
        [-0.27459338],
        [0.27264398],
        [0.34232601],
        [-0.09004883],
        [0.54519332],
        [1.0],
        [-0.66314067],
    ]
)


# Combined Cycle Power Plant Data Set
# https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
abs_path = pathlib.Path(__file__).parent.resolve()
ccpp_df = pd.read_csv(f"{abs_path}/CCPP_data.csv")
ccpp_features = ccpp_df[["AT", "V", "AP", "RH"]]
ccpp_labels = ccpp_df["PE"]

DATASET_CCPP_FEATURES = ccpp_features[:35].to_numpy()

DATASET_CCPP_LABELS = ccpp_labels[:35].to_numpy()
