#!/bin/bash

rsync -avz --progress \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude "results/" \
  --exclude "logs/" \
  --exclude ".idea/" \
  --exclude "copy_to_cluster.sh" \
  ~/projects/dager-gradient-inversion/ \
  rub1:/lustre/guptap69/projects/dager-gradient-inversion/