#!/usr/bin/env bash

# if any command inside script returns error, exit and return that error
set -e

# magic line to ensure that we're always inside the root of our application,
# no matter from which directory we'll run script
# thanks to it we can just enter `./scripts/to_conda.bash`
cd "${0%/*}/.."
export PATH=$PATH:$HOME/anaconda3/bin
export CONDA=build_conda_pkg
version=`python setup.py --version`
name=`python setup.py --name`

echo "Script for generating conda package"
echo "Remove the conda directory if it exists"
rm -rf $CONDA
echo "Activating conda environment"
eval "$(conda shell.bash hook)"
conda activate
echo "Create conda directory"
mkdir -p $CONDA
echo "Create the sketeton for conda from pypi"
conda skeleton pypi $name --output-dir $CONDA
sed -i 's/ table/ pytable/g; s/your-github-id-here/mikekatz04/g; s/doc_url:/doc_url: https:\/\/mikekatz04.github.io\/LISAanalysistools\/html\/index.html/g; s/license_file:/license_file: LICENSE/g' $CONDA/$name/meta.yaml
echo "Generate the conda package"
conda build $CONDA/$name
echo "Convert the conda package to other systems"
conda convert $HOME/anaconda3/conda-bld/linux-64/$name-$version-py39_0.tar.bz2 -p all -o dist
echo "Deactivating conda environment"
conda deactivate
