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

"""Protein Folding Problem benchmarks."""
from itertools import product
from timeit import timeit

from qiskit_nature.problems.sampling.protein_folding import (
    PenaltyParameters,
    Peptide,
    ProteinFoldingProblem,
    RandomInteraction,
    MiyazawaJerniganInteraction,
    MixedInteraction,
)


# pylint: disable=redefined-outer-name, invalid-name


class ProteinFoldingProblemBenchmarks:
    """Protein Folding Problem benchmarks."""

    version = 1
    params = [
        ["Neuropeptide", "NeuropeptideDummySide", "Angiotensin", "AngiotensinDummySide"],
        ["MiyazawaJerniganInteraction", "RandomInteraction", "MixedInteraction"],
    ]
    param_names = ["peptide", "interaction type"]

    def __init__(self):
        self.peptides = None
        self.interactions = None
        self.main_chain_residue_sequence = None
        self.side_chain_residue_sequences = None
        self.protein_folding_problem = None

    def _initialization(self):
        if self.peptides is None:
            self.peptides = {
                "Neuropeptide": ("APRLRFY", [""] * 7),  #
                "NeuropeptideDummySide": ("APRLRFY", ["", "", "R", "", "T", "W", ""]),
                # Neuropeptide with dummy side chains
                "Angiotensin": ("DRVYIHPFHL", [""] * 10),  # Angiotensin I, human
                "AngiotensinDummySide": (
                    "DRVYIHPFHL",
                    ["", "", "P", "R", "L", "H", "Y", "", "I", ""],
                ),
            }  # Angiotensin I, human with dummy side chains

        if self.interactions is None:
            self.interactions = {
                "MiyazawaJerniganInteraction": MiyazawaJerniganInteraction(),
                "RandomInteraction": RandomInteraction(),
                "MixedInteraction": MixedInteraction(),
            }

    def setup(self, peptide_id, interaction_id):
        """setup"""
        self._initialization()
        self.main_chain_residue_sequence = self.peptides[peptide_id][0]
        self.side_chain_residue_sequences = self.peptides[peptide_id][1]
        peptide = Peptide(self.main_chain_residue_sequence, self.side_chain_residue_sequences)
        interaction = self.interactions[interaction_id]
        self.protein_folding_problem = ProteinFoldingProblem(
            peptide, interaction, PenaltyParameters()
        )

    def time_generate_peptide(self, _, __):
        """Time generation of a peptide."""
        return Peptide(self.main_chain_residue_sequence, self.side_chain_residue_sequences)

    def time_generate_full_qubit_operator(self, _, __):
        """Time generation of full protein folding qubit operator."""
        return self.protein_folding_problem._qubit_op_full()

    def time_generate_compressed_qubit_operator(self, _, __):
        """Time generation of compressed protein folding qubit operator."""
        return self.protein_folding_problem.qubit_op()


if __name__ == "__main__":
    for peptide_id, interaction_id in product(*ProteinFoldingProblemBenchmarks.params):
        bench = ProteinFoldingProblemBenchmarks()
        try:
            bench.setup(peptide_id, interaction_id)
        except NotImplementedError:
            continue
        for method in set(dir(ProteinFoldingProblemBenchmarks)):
            if method.startswith("time_"):
                elapsed = timeit(f"bench.{method}(None, None)", number=10, globals=globals())
                print(
                    f"main_chain_residue_seq="
                    f"{bench.peptides[peptide_id][0]}, "
                    f"side_chain_residue_sequences={bench.peptides[peptide_id][1]}, "
                    f"interaction={bench.interactions[interaction_id]} {method}:\t{elapsed}"
                )
