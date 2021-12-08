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

"""Circuit QNN Classifier score benchmarks."""
from itertools import product
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import NELDER_MEAD
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class CircuitQnnScoreClassifierBenchmarks(BaseClassifierBenchmark):
    """Circuit QNN Classifier Score benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_synthetic", "dataset_iris"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["dataset", "backend name"]

    def setup_dataset_synthetic(self, X_train, X_test, y_train, num_inputs, quantum_instance_name):
        """Training CircuitQNN function for synthetic dataset."""

        self.output_shape = 2  # corresponds to the number of classes, possible outcomes of the (
        # parity) mapping.

        # parity maps bitstrings to 0 or 1
        def parity(x):
            return f"{x:b}".count("1") % 2

        # construct feature map
        feature_map = ZZFeatureMap(num_inputs)

        # construct ansatz
        ansatz = RealAmplitudes(num_inputs, reps=1)

        # construct quantum circuit
        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        # construct QNN
        self.circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=parity,
            output_shape=self.output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.circuit_classifier_fitted = NeuralNetworkClassifier(
            neural_network=self.circuit_qnn, optimizer=NELDER_MEAD()
        )

        try:
            self.circuit_classifier_fitted._fit_result = load(
                f"/tmp/dataset_synthetic_classification_{quantum_instance_name}.obj"
            )
        except FileNotFoundError:
            self.circuit_classifier_fitted.fit(X_train, y_train)

        self.y_predict = self.circuit_classifier_fitted.predict(X_test)

    def setup_dataset_iris(self, X_train, X_test, y_train, num_inputs, quantum_instance_name):
        """Training CircuitQNN function for iris dataset."""

        self.output_shape = 3

        # creating feature map
        feature_map = ZZFeatureMap(num_inputs)

        # creating ansatz
        ansatz = RealAmplitudes(num_inputs)

        qc = QuantumCircuit(num_inputs)
        qc.append(feature_map, range(num_inputs))
        qc.append(ansatz, range(num_inputs))

        def three_class(x):
            return f"{x:b}".count("1") % 3

        # construct QNN
        circuit_qnn = CircuitQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            interpret=three_class,
            output_shape=self.output_shape,
            quantum_instance=self.backends[quantum_instance_name],
        )

        self.circuit_classifier_fitted = NeuralNetworkClassifier(
            neural_network=circuit_qnn,
            optimizer=NELDER_MEAD(),
        )
        try:
            self.circuit_classifier_fitted._fit_result = load(
                f"/tmp/dataset_iris_{quantum_instance_name}.obj"
            )
        except FileNotFoundError:
            self.circuit_classifier_fitted.fit(X_train, y_train)

        self.y_predict = self.circuit_classifier_fitted.predict(X_test)

    def setup(self, dataset, quantum_instance_name):
        """Setup"""

        self.X = self.datasets[dataset]["features"]
        num_inputs = len(self.X[0])
        self.y01 = self.datasets[dataset]["labels"]

        if dataset == "dataset_iris":
            # scaling data
            scaler = MinMaxScaler((-1, 1))
            self.X = scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y01, test_size=0.25
        )

        if dataset == "dataset_synthetic":
            self.setup_dataset_synthetic(
                self.X_train, self.X_test, self.y_train, num_inputs, quantum_instance_name
            )
        elif dataset == "dataset_iris":
            self.setup_dataset_iris(
                self.X_train, self.X_test, self.y_train, num_inputs, quantum_instance_name
            )

    def setup_cache(self):
        """Cache CircuitQNN fitted model"""
        for dataset, backend in product(*self.params):
            self.setup(dataset, backend)

            dump(self.circuit_classifier_fitted._fit_result, f"/tmp/{dataset}_{backend}.obj")

    def track_overall_accuracy_circuit_qnn_classifier(self, _, __):
        """Tracks the overall accuracy of the classification results."""
        acc_score = accuracy_score(self.y_test, self.y_predict)
        return acc_score

    def track_cohen_kappa_circuit_qnn_clasifier(self, _, __):
        """Tracks the cohen kappa score of the classification results."""
        cohen_kappa = cohen_kappa_score(self.y_test, self.y_predict)
        return cohen_kappa

    def track_f1_score_circuit_qnn_classifier(self, _, __):
        """Tracks the f1 score for each class of the classification results."""
        f1 = f1_score(self.y_test, self.y_predict, average="macro")
        return f1


if __name__ == "__main__":
    for dataset, backend in product(*CircuitQnnScoreClassifierBenchmarks.params):
        bench = CircuitQnnScoreClassifierBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue
