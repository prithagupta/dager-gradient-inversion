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

#rsync -avz --progress rub1:/lustre/guptap69/projects/dager-gradient-inversion/results/ ~/projects/dager-gradient-inversion/results/
# rsync -avz --progress rub1:/lustre/guptap69/projects/dager-gradient-inversion/logs/ ~/projects/dager-gradient-inversion/logs/

#rsync -avz --progress ~/projects/dager-gradient-inversion/results rub1:/lustre/guptap69/projects/dager-gradient-inversion/results/
#rsync -avz --progress ~/projects/dager-gradient-inversion/logs rub1:/lustre/guptap69/projects/dager-gradient-inversion/logs/
