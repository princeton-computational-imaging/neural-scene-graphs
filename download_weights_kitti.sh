#!/bin/bash

DATADIR="./example_weights/"
mkdir $DATADIR
wget -q -O tmp.zip "https://drive.google.com/uc?export=download&id=1o28o6gOGHrjQ3LA5Kazj6zdzXEVboS8g" && unzip tmp.zip -d $DATADIR && rm tmp.zip
