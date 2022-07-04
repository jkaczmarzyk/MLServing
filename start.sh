#!/bin/bash
VENV='./venv'

# checks if venv exists else installs for you
if [ ! -d "$VENV" ]; then
    python3 -m venv $VENV
    source $VENV/bin/activate
    python3 -m pip install -r ./requirements.txt
else
    echo "venv exists"
fi

# starts app
export FLASK_APP=mlserving.py
flask run --host=0.0.0.0