#######################################################################################
## DO NOT EDIT THIS FILE! This file was automatically generated from the dockerfile. ##
## Run dynwrap:::.container_dockerfile_to_singularityrecipe() to update this file.   ##
#######################################################################################

Bootstrap: shub

From: dynverse/dynwrap:py3.6

%labels
    version 0.1.0

%post
    pip install Cython
    pip install git+https://github.com/dynverse/pywishbone --upgrade --upgrade-strategy only-if-needed

%files
    . /code

%runscript
    exec python /code/run.py
