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

# You can set those variables from the command line.
TARGET  =
ASVCMD  =
ASVOPTS =
SPHINXOPTS =

.PHONY: asv lint mypy style black spell copyright html clean_sphinx clean

asv:
	make -C $(TARGET) asv ASVCMD=$(ASVCMD) ASVOPTS="$(ASVOPTS)"

lint:
	python -m pylint -rn --ignore=.asv finance machine_learning nature optimization tools
	python tools/verify_headers.py finance machine_learning nature optimization tools
	python tools/check_version.py finance machine_learning nature optimization

mypy:
	python -m mypy finance machine_learning nature optimization tools

style:
	python -m black --check finance machine_learning nature optimization tools docs

black:
	python -m black finance machine_learning nature optimization tools docs

spell:
	python -m pylint -rn --disable=all --enable=spelling --spelling-dict=en_US --spelling-private-dict-file=.pylintdict --ignore=.asv finance machine_learning nature optimization tools
	make -C docs spell SPHINXOPTS=$(SPHINXOPTS)

copyright:
	python tools/check_copyright.py

html:
	make -C docs html SPHINXOPTS=$(SPHINXOPTS)

clean_sphinx:
	make -C docs clean

clean: clean_sphinx
