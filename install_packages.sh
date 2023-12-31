#!/bin/bash

# Workaround for SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

while read package; do
    pip install "$package"
done < requirements.txt