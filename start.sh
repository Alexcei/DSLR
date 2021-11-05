#!/bin/sh

python -m venv env
source env/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
