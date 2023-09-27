#!/bin/bash

while read package; do
    pip install "$package"
done < requirements.txt
