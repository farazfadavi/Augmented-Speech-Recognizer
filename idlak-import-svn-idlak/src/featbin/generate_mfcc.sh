#!/bin/sh
# Generates bndap from a wav file. The kaldi continuous pitch tracker is run first, then
# the output is used to generate bndap files
inwav=$1
out=$2
order=$3
export PATH=/home/potard/aria/idlak/src/featbin:$PATH
srate=`sox --i $inwav | grep 'Sample Rate' | awk '{print $4}'`
common_opts="--snip-edges=false --frame-shift=5 --sample-frequency=$srate --frame-length=25"
mfcc_opts="--num-ceps=$order --num-mel-bins=$order --cepstral-lifter=0.0"
input="scp:echo a $inwav |"
output="ark,t:| awk -v out=$out '(NR > 1){if (\$NF == \"]\") NF=NF-1; for (i = 1; i <= NF; i++) printf \"%f \", \$i > out; printf \"\n\" > out}'"
./compute-mfcc-feats $common_opts $mfcc_opts "$input" "$output"
