#!/bin/bash


rsync -avz --progress --exclude "*.lock" rub1:/lustre/guptap69/projects/dager-gradient-inversion/results/ ~/projects/dager-gradient-inversion/results/
rsync -avz --progress --exclude "*.lock" rub1:/lustre/guptap69/projects/dager-gradient-inversion/logs/ ~/projects/dager-gradient-inversion/logs/
rsync -avz --progress rub1:/lustre/guptap69/logs/ ~/projects/logs/

rsync -avz --progress --exclude "*.lock" themis:/media/data/gupta/dager-gradient-inversion/results/ ~/projects/dager-gradient-inversion/results/
rsync -avz --progress --exclude "*.lock" themis:/media/data/gupta/dager-gradient-inversion/logs/ ~/projects/dager-gradient-inversion/logs/
rsync -avz --progress --exclude "*.lock" themis:/media/data/gupta/dager-gradient-inversion/themis_scripts/logs/ ~/projects/logs/

#rsync -avz --progress ~/projects/dager-gradient-inversion/results/ themis:/media/data/gupta/dager-gradient-inversion/results/
#rsync -avz --progress ~/projects/dager-gradient-inversion/logs/ themis:/media/data/gupta/dager-gradient-inversion/logs/
