# Qiskit Application Benchmarks

[![License](https://img.shields.io/github/license/Qiskit/qiskit-app-benchmarks.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://github.com/Qiskit/qiskit-app-benchmarks/workflows/Application%20Benchmarks%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-app-benchmarks/actions?query=workflow%3A"Application%20Benchmarks%20Tests"+branch%3Amain+event%3Apush)

## Usage

In order to run benchmarks, run:

* Finance: `make benchmark TARGET=finance`
* Machine Learning: `make benchmark TARGET=machine_learning`
* Optimization: `make benchmark TARGET=optimization`
* Nature: `make benchmark TARGET=nature`

Before any benchmarking, you need to set once your machine info.
If you accept defaults, for finance for instance, run `make machine TARGET=finance ASVOPTS=--yes`
Another option is to run in development mode as a validation: `run benchmark_dev TARGET=machine_learning`

----------------------------------------------------------------------------------------------------

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](./CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-app-benchmarks/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Authors and Citation

Application Benchmarks were inspired, authored and brought about by the collective work of a team of researchers.
Application Benchmarks continues to grow with the help and work of
[many people](https://github.com/Qiskit/qiskit-app-benchmarks/graphs/contributors), who contribute
to the project at different levels.
If you use Qiskit, please cite as per the provided
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-app-benchmarks/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](LICENSE.txt).

                                                                          
