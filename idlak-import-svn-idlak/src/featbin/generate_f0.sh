#!/bin/sh
# Generates f0 from a wav file. The kaldi continuous pitch tracker is used.

export PATH=/home/potard/aria/idlak/src/featbin:$PATH

help_message="Usage: ./generate_bndap.sh [options] <in.wav> <out.bndap>\n\tcf. top of file for list of options."

f0wlen=25
extra_opts=""

. parse_options.sh

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: generate_f0.sh [options] <in.wav> <out.f0>"
   echo " => will generate f0 using kaldi continuous pitch tracker"
   echo " e.g.: generate_f0.sh /tmp/vde_z0001_001_000.wav /tmp/vde_z0001_001_000.f0"
   exit 1;
fi

inwav=$1
out=$2
srate=`sox --i $inwav | grep 'Sample Rate' | awk '{print $4}'`
#delay=$(( $f0wlen / 5 ))
common_opts="--snip-edges=false --frame-shift=5 --sample-frequency=$srate --frame-length=$f0wlen"
input="scp:echo a $inwav |"
output="ark,t:| awk -v out=$out '(NR > 1){print \$2 > out}'"
compute-kaldi-pitch-feats $common_opts $extra_opts "$input" "$output"
