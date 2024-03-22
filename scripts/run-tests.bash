#!/usr/bin/env bash

# if any command inside script returns error, exit and return that error
set -e

# magic line to ensure that we're always inside the root of our application,
# no matter from which directory we'll run script
# thanks to it we can just enter `./scripts/run-tests.bash`
cd "${0%/*}/.."

echo " "
echo "1 - Quality code : Cyclomatic Complexity"
echo "----------------------------------------"
nb=`radon cc -n C -e "LISAanalysistools/catalog.py" LISAanalysistools | wc -l`
if [ $nb -ne "0" ]
then
    echo "Failed!" && exit 1
else
    echo "OK!"
fi

echo " "
echo " "
echo "2 - Quality code : Maintenability"
echo "----------------------------------"
nb=`radon mi -n C LISAanalysistools | wc -l`
if [ $nb -ne "0" ]
then
    echo "Failed!" && exit 1
else
    echo "OK!"
fi

echo " "
echo " "
echo "3 - Tests"
echo "----------------------------------"
coverage run -m robot --variable TMP_DIR:tests/results --outputdir tests/results/ tests/testsuites/

if [ $? -ne "0" ]
then
    echo "Failed!" && exit 1
else
    echo "OK!"
fi
coverage html --include=LISAanalysistools/* --directory=tests/results/coverage

echo " "
echo " "
echo "Information"
echo "-----------"
radon raw -s LISAanalysistools
pygount --format=summary LISAanalysistools
