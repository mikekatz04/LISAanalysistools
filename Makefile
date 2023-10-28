#########################################################################
#																		#
#                     Project management utilities                     	#
#		      		  ----------------------------						#
#																		#
#########################################################################

define PROJECT_HELP_MSG

Usage:\n
	\n
    make help\t\t\t             show this message\n
	\n
	-------------------------------------------------------------------------\n
	\t\tInstallation and start/stop the server\n
	-------------------------------------------------------------------------\n
	make\t\t\t\t                Install lisacattools in the system (root)\n
	make user\t\t\t 			Install lisacattools for non-root usage\n
	\n
	-------------------------------------------------------------------------\n
	\t\tDevelopment\n
	-------------------------------------------------------------------------\n
	make prepare-dev\t\t 		Prepare Development environment\n
	make install-dev\t\t 		Install COTS\n
	make data\t\t\t				Download data\n
	make test\t\t\t             Run units and integration tests\n
	make quality\t\t\t 			Run quality tests\n
	make tox\t\t\t 			Tests in several environments\n

	\n
	make demo\t\t\t				Play the demo\n
	make doc\t\t\t 				Generate the documentation\n
	make doc-pdf\t\t\t 			Generate the documentation as PDF\n
	make visu-doc-pdf\t\t 		View the generated PDF\n
	make visu-doc\t\t\t			View the generated documentation\n
	\n
	make release\t\t\t 			Release the package as tar.gz\n
	make conda\t\t\t			Make conda package from Pypi\n
	make release-pypi\t\t   	Release the package for pypi\n
	make upload-test-pypi\t\t   Upload the pypi package on the test platform\n
	make upload-prod-pypi\t\t   Upload the pypi package on the prod platform\n
	\n
	-------------------------------------------------------------------------\n
	\t\tOthers\n
	-------------------------------------------------------------------------\n
	make licences\t\t\t	Display the list of licences



endef
export PROJECT_HELP_MSG

VENV = ".lisacattools-env"

#
# Sotware Installation in the system (need root access)
# -----------------------------------------------------
#
init:
	python3 setup.py install

#
# Sotware Installation for user
# -----------------------------
# This scheme is designed to be the most convenient solution for users
# that don’t have write permission to the global site-packages directory or
# don’t want to install into it.
#
user:
	python3 setup.py install --user

#Show help
#---------
help:
	echo $$PROJECT_HELP_MSG

#
# Development - prepare env
# ----------------------------------
#
prepare-dev:
	echo "python3 -m venv lisacattools-env && export PYTHONPATH=." > .lisacattools-env && echo "source \"`pwd`/lisacattools-env/bin/activate\"" >> .lisacattools-env && scripts/install-hooks.bash && echo "\nnow source this file: \033[31msource ${VENV}\033[0m"

install-dev:
	pip install -r requirements.txt && pip install -r requirements-dev.txt && pre-commit install && pre-commit autoupdate

data:
	pip install -r requirements-data.txt && python scripts/data_download.py

#
# Development - create doc and tests
# ----------------------------------
#
doc:
	make test && cp tests/results/*.html docs/source/_static/ && cp -r tests/results/coverage docs/source/_static/ && make html -C docs

doc-pdf:
	make doc && make latexpdf -C docs

visu-doc-pdf:
	acroread docs/build/latex/lisacattools.pdf

visu-doc:
	firefox docs/build/html/index.html

test:
	make data && scripts/run-tests.bash

quality:
	pre-commit run --all-files

tox:
	pyenv local 3.8 3.9 3.10 && tox

#
# Create distribution
# ----------------------------------
#
changelog:
	pip install -r requirements-release.txt && gitchangelog > CHANGELOG

clean:
	rm -rf dist/ build/ lisacattools.egg-info/ docs/source/examples_* && make clean -C docs && find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

release:
	make clean && make changelog && python3 setup.py sdist

conda:
	bash scripts/to_conda.bash

release-pypi:
	make clean && make changelog && python3 setup.py sdist bdist_wheel && lisacattools-env/bin/twine check dist/*

upload-test-pypi:
	lisacattools-env/bin/twine check dist/* && lisacattools-env/bin/twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload-prod-pypi:
	lisacattools-env/bin/twine check dist/* && lisacattools-env/bin/twine upload --repository-url https://pypi.org/legacy/ dist/*

demo:
	make data && pip install -r requirements-demo.txt && ./lisacattools-env/bin/jupyter-notebook tutorial/MBHdemo.ipynb

licences:
	pip-licenses

.PHONY: help user prepare-dev install-dev doc visu-doc test tox changelog clean release release-pypi upload-test-pypi upload-prod-pypi demo licences conda
