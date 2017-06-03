#!/bin/sh
# Generates bndap from a wav file. The kaldi continuous pitch tracker is run first, then
# the output is used to generate bndap files.
# Note that you can also use an externally generated F0,
# with the format: 
# <F0 Hz> <probability of voicing>
# A continous F0 is recommended but not mandatory. If the F0 is
# trustworthy, then the peak_quality parameter may be increased; 
# a value of 1.0 will force the use of F0 to generate the harmonic
# spectrogram. A peak_width of 0 will further reinforce it.


export PATH=/home/potard/aria/idlak/src/featbin:$PATH


help_message="Usage: ./generate_bndap.sh [options] <in.wav> <out.bndap>\n\tcf. top of file for list of options."

f0wlen=25
bapwlen=40
peak_width=0.1
peak_quality=0.4
num_bands=5
use_hts_bands=true
external_f0=""
debug_aperiodic=false

. parse_options.sh

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: generate_bndap.sh [options] <in.wav> <out.bndap>"
   echo " => will generate bndap using kaldi aperiodic analyser"
   echo " e.g.: generate_bndap.sh /tmp/vde_z0001_001_000.wav /tmp/vde_z0001_001_000.bndap"
   exit 1;
fi

inwav=$1
out=$2

srate=`sox --i $inwav | grep 'Sample Rate' | awk '{print $4}'`
common_opts="--snip-edges=false --frame-shift=5 --sample-frequency=$srate --frame-length=$f0wlen"
aperiodic_opts="--num-mel-bins=$num_bands --debug-aperiodic=$debug_aperiodic --use-hts-bands=$use_hts_bands --peak-quality=$peak_quality --peak-width=$peak_width --max-iters=1  --round-to-power-of-two=true --frame-length=$bapwlen"
input="scp:echo a $inwav |"
if [ -e "$external_f0" ]; then
    # Assuming output from get_f0
    f0_extract="ark:awk < $external_f0 'BEGIN{print \"a [\"}{if (NF >= 2) print \$2, \$1; else if (\$1 > 0) print 1.0, \$1; else print 0.0, \$1}END{print \"]\"}' |"

else
    # Default is to use kaldi continous pitch extractor
    f0_extract="ark:compute-kaldi-pitch-feats $common_opts \"$input\" ark:- |"
fi
output="ark,t:| awk -v out=$out '(NR > 1){if (\$NF == \"]\") NF=NF-1; for (i = 1; i <= NF; i++) if (\$i >= -0.5) printf \"0.0 \" > out; else printf \"%f \", 20 * (\$i+0.5) / log(10) > out; printf \"\n\" > out}'"
compute-aperiodic-feats --verbose=2 $common_opts $aperiodic_opts "$input" "$f0_extract" "$output"
