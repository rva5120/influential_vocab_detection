#!/bin/bash

echo "Extracting documents..."
python extract_docs.py

echo "Generating dataset..."
python generate_dataset_vars.py
