# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base class for QSVC based classifier benchmarks."""
from abc import ABC
from typing import Optional

from qiskit.algorithms.optimizers import Optimizer
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, QSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from .base_classifier_benchmark import BaseClassifierBenchmark


class QsvcBaseClassifierBenchmark(BaseClassifierBenchmark, ABC):
    """Base class for QSVC benchmark"""

    def __init__(self) -> None:
        reshaper = FunctionTransformer(lambda x: x.reshape(-1, 1))
        encoder = OneHotEncoder(sparse=False)
        super().__init__(
            synthetic_label_encoder=Pipeline([("reshape", reshaper), ("one hot", encoder)]),
            iris_num_classes=2,
            iris_label_encoder=Pipeline([("reshape", reshaper), ("one hot", encoder)]),
        )

    #I just built 1 function for method, I don't want to differentiate datasets
    def _construct_QuantumKernel_classical_classifier(self,                                             
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None,
        method = "quantumclassical", #do not modify
        num_qubits = 1, ) -> QuantumKernel:
        """This method just create the matrix from the quantum kernel. Later will be applied the classical SVC"""
        kernelmatrix = self._construct_QuantumKernel(num_qubits, quantum_instance_name, method)
        #put here the function calling the quantum kernel matrix (Quantum Kernel)
        return kernelmatrix

    def _construct_QuantumKernelTrainer(self,
        quantum_instance_name: str,
        optimizer: Optional[Optimizer] = None, 
        loss_function: str = None,
        method = "quantum", #do not modify
        num_qubits = 1,) -> QuantumKernelTrainer:
        """This method returns the QuantumKernelTrainer"""
        kernel = self._construct_QuantumKernel(num_qubits, quantum_instance_name, method)
        optimizer = optimizer #eg L_BFGS_B(maxiter=20) but we need also learning rate, perturbation etc.
        # Instantiate a quantum kernel trainer.
        #look up how to put random initial parameters, seems simple just check
        qkt = QuantumKernelTrainer(quantum_kernel=kernel, loss=loss_function, optimizer = optimizer, initial_point=[np.pi / 2]) #initial random point
        return qkt #use the QuantumKernel to return the classifier method

    def _construct_QuantumKernel(
        self,
        num_inputs: int,
        quantum_instance_name: str,
        method: str,
        optimizer: Optional[Optimizer] = None
    ) -> QuantumKernel:
        """Construct a QuantumKernel"""
        #here we can consider to add functions to be called for the kind of ansatz
        # or the ansatz as input here whatever
        #we should also personalize the parameters in the quantum method
        if method == "quantumclassical":
            feature_map = ZZFeatureMap(num_inputs, reps=2, entanglement="linear")
            #quantum kernel, not parametrized
            qk = QuantumKernel(feature_map=feature_map, quantum_instance=self.backends[quantum_instance_name])
            return qk
        elif method == "quantum":
            #super dumb parametrized start
            #<<<<<<<<<<<<<<<<< any number of qubits
            user_params = ParameterVector("θ", 1)
            fm0 = QuantumCircuit(num_inputs)
            for i in range(num_inputs):
                fm0.ry(user_params[0], i)
            fm1 = ZZFeatureMap(num_inputs, reps=2, entanglement="linear")
            feature_map = fm0.compose(fm1)
            #quantum kernel, parametrized
            qk = QuantumKernel(feature_map = feature_map, user_parameters=user_params, quantum_instance=self.backends[quantum_instance_name])
            return qk
        else:
            return ValueError(f"Unsupported method: {method}")

