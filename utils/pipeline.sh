#!/bin/bash
PYTHONPATH=. python train.py
PYTHONPATH=. python evaluate.py
PYTHONPATH=. python convert.py
