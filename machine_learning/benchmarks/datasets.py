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

import numpy as np

DATASET_SYNTHETIC_CLASSIFICATION_FEATURES = np.array(
    [
        [-0.5662535427032733, -0.9506985204545293],
        [0.9237926141124049, -0.21024034471405928],
        [-0.5363927962538539, -0.6302006194568324],
        [-0.2977115347918018, 0.7846970052821041],
        [0.21787357489553272, -0.7196039773174467],
        [-0.010251183455284751, 0.396711608321084],
        [0.42017133504711857, -0.8924068543685355],
        [0.13563265892971055, 0.25827213877148214],
        [-0.36481296618769576, 0.03628785960039349],
        [0.3192543835725403, -0.1446297087477213],
        [-0.925488106852753, -0.4419621857074916],
        [-0.4224764941409678, -0.30310071927735405],
        [-0.7589609745678978, -0.1886743530929469],
        [-0.47958582394439997, 0.7546226885186544],
        [-0.798675382973272, -0.15556541510766309],
        [-0.38321155225715753, -0.023299505759131423],
        [-0.7851734061535027, -0.9130207147701899],
        [0.3841410493379849, 0.8008340382312655],
        [-0.2914294558218786, 0.2355021627215368],
        [0.5199932916423333, 0.6951624888684251],
        [0.3588191281355948, -0.04488150511315059],
        [-0.5102264261410945, -0.7506684295154553],
        [-0.9568730382417594, 0.5771183134462541],
        [0.2764265583218535, 0.2632603202395736],
        [0.8982101657724386, -0.31681068006920854],
    ]
)

DATASET_SYNTHETIC_CLASSIFICATION_LABELS = np.array(
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
)
