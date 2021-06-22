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

# You can set this variable from the command line.
TARGET  =
ASVOPTS =

.PHONY: machine dev run publish preview lint mypy style black spell copyright

machine:
	make -C $(TARGET) machine ASVOPTS=$(ASVOPTS)

dev:
	make -C $(TARGET) dev ASVOPTS=$(ASVOPTS)

run:
	make -C $(TARGET) run ASVOPTS=$(ASVOPTS)

publish:
	make -C $(TARGET) publish ASVOPTS=$(ASVOPTS)

preview:
	make -C $(TARGET) preview ASVOPTS=$(ASVOPTS)

lint:
	pylint -rn finance machine_learning nature optimization tools

mypy:
	mypy finance machine_learning nature optimization tools

style:
	black --check finance machine_learning nature optimization tools

black:
	black finance machine_learning nature optimization tools

spell:
	pylint -rn --disable=all --enable=spelling --spelling-dict=en_US --spelling-private-dict-file=.pylintdict finance machine_learning nature optimization tools

copyright:
	python tools/check_copyright.py
