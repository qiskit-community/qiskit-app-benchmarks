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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# pylint: disable=redefined-outer-name, invalid-name, attribute-defined-outside-init
from .base_classifier_benchmark import BaseClassifierBenchmark


class CircuitQnnScoreClassifierBenchmarks(BaseClassifierBenchmark):
    """Circuit QNN Classifier Score benchmarks."""

    version = 1
    timeout = 1200.0
    params = [["dataset_synthetic"], ["qasm_simulator", "statevector_simulator"]]
    param_names = ["backend name"]

    def __init__(self):
        super().__init__()

        self.output_shape = 2  # corresponds to the number of classes, possible outcomes of the (
        # parity) mapping.

    def setup(self, dataset, quantum_instance_name):
        """setup"""
        self.X = self.datasets[dataset]["features"]
        num_inputs = len(self.X[0])
        self.y01 = self.datasets[dataset]["labels"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y01, test_size=0.25
        )

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
            neural_network=self.circuit_qnn, optimizer=COBYLA()
        )
        self.circuit_classifier_fitted.fit(self.X, self.y01)
        self.circuit_classifier_fitted.score(self.X, self.y01)
        y_predict = self.circuit_classifier_fitted.predict(self.X_test)

        # create confusion matrix of results of classification
        self.conf_mat = confusion_matrix(self.y_test, y_predict)

        # calculating false positive, false negative, true positive and true negative
        self.tn, self.fp, self.fn, self.tp = self.conf_mat.ravel()

    def track_false_positive_rate_circuit_qnn_classifier(self, _, __):
        """Tracks the false positive rate of classification results from confusion matrix."""
        false_positive_rate = self.fp / (self.fp + self.tn)
        return false_positive_rate

    def track_false_negative_rate_circuit_qnn_classifier(self, _, __):
        """Tracks the false negative rate of classification results from confusion matrix."""
        false_negative_rate = self.fn / (self.tp + self.fn)
        return false_negative_rate

    def track_true_positive_rate_circuit_qnn_classifier(self, _, __):
        """Tracks the true positive rate of classification of results from confusion matrix."""
        true_positive_rate = self.tp / (self.tp + self.fn)
        return true_positive_rate

    def track_true_negative_rate_circuit_qnn_classifier(self, _, __):
        """Tracks the true negative rate of classification of results from confusion matrix."""
        true_negative_rate = self.tn / (self.tn + self.fp)
        return true_negative_rate

    def track_overall_accuracy_circuit_qnn_classifier(self, _, __):
        """Tracks the overall accuracy of the classification results from confusion matrix."""
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        return accuracy


if __name__ == "__main__":
    for dataset, backend in product(*CircuitQnnScoreClassifierBenchmarks.params):
        bench = CircuitQnnScoreClassifierBenchmarks()
        try:
            bench.setup(dataset, backend)
        except NotImplementedError:
            continue
