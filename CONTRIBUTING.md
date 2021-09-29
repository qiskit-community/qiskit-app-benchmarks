# Contributing

**We appreciate all kinds of help, so thank you!**

First please read the overall project contributing guidelines. These are
included in the Qiskit documentation here:

https://qiskit.org/documentation/contributing_to_qiskit.html

## Contributing to Qiskit Application Benchmarks

In addition to the general guidelines above there are specific details for
contributing to Qiskit Application Benchmarks, these are documented below.

### Project Code Style.

Code in Qiskit Application Benchmarks should conform to PEP8 and style/lint checks are run to validate
this.  Line length must be limited to no more than 100 characters. Docstrings
should be written using the Google docstring format.

Every Benchmark class should have  a `version` property. If the benchmark class changes in a way that would invalidate previous
results, the `version` should change in order to reset previous results and start fresh from next commit.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the _code style_ of this project and successfully
   passes the _unit tests_. Application Benchmarks uses [Pylint](https://www.pylint.org) and
   [PEP8](https://www.python.org/dev/peps/pep-0008) style guidelines.
   
   You can run
   ```shell script
   make lint
   make style 
   ```
   from the root of the Application Benchmarks repository clone for lint and style conformance checks.

   If your code fails the local style checks (specifically the black
   code formatting check) you can use `make black` to automatically
   fix update the code formatting.
   
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
   
   You can run `make spell` locally to check spelling though you would need to
   [install pyenchant](https://pyenchant.github.io/pyenchant/install.html) and be using
   hunspell-en-us as is used by the CI. 
   
   For some words, such as names, technical terms, referring to parameters of the method etc., 
   that are not in the en-us dictionary and get flagged as being misspelled, despite being correct,
   there is a [.pylintdict](./.pylintdict) custom word list file, in the root of the Application Benchmarks repo,
   where such words can be added, in alphabetic order, as needed.
   
3. If it makes sense for your change that you have added new tests that
   cover the changes and any new function.

4. Ensure all code, has the copyright header. The copyright
   date will be checked by CI build. The format of the date(s) is _year of creation,
   last year changed_. So for example:
   
   > \# (C) Copyright IBM 2018, 2021.

   If the _year of creation_ is the same as _last year changed_ then only
   one date is needed, for example:

   > \# (C) Copyright IBM 2021.
                                                                                                                                                                                                 
   If code is changed in a file make sure the copyright includes the current year.
   If there is just one date and it's a prior year then add the current year as the 2nd date, 
   otherwise simply change the 2nd date to the current year. The _year of creation_ date is
   never changed.
   
### Branches

* `main`:

The main branch is used for development of the next version of qiskit-app-benchmarks.
It will be updated frequently and should not be considered stable. The API
can and will change on main as we introduce and refine new features.

* `stable/*`:
The stable branches are used to maintain the most recent released versions of
qiskit-app-benchmarks. It contains the versions of the code corresponding to the minor
version release in the branch name release for The API on these branches are
stable and the only changes merged to it are bugfixes.
