source cmd.sh
source path.sh
#SETSHELL grid
#SETSHELL sptk

# This recipe uses the arctic database as audio input.
# Please feel free to adapt it to other DB

#WSJCAM0=/idiap/resource/database/WSJCAM0/
#WSJAUDIO=$WSJCAM0/audio/data/primary_microphone/
#WSJLABEL=$WSJCAM0/dbase/data/primary_microphone/
#TMPDIR=/idiap/temp/$USER


#### General instructions for training a DNN-TTS system ####
# 1. Generate acoustic features with emime script
# 2. Generate full / mono labels from festival
# 3. Align these labels with audio: create fake dictionary where phones are mapped to phones,
#    replace text with the phones of mono labels. Use normal kaldi setup on the MFCC features
#    only (i.e. the first 40 scalar of acoustic vector)
# 4. Generate acoustic / label dataset from above data
# 5. Train the DNN :-)

####################################
##### Step 0: data preparation #####
####################################

# Arctic database usual sampling rate; although 32k is
# also available for some speakers.
#srate=16000
srate=48000
FRAMESHIFT=0.005
TMPDIR=/tmp
# Clean up
rm -rf data/train data/eval data/dev data/train_* data/eval_* data/dev_*
HTS_DEMO_LOCATION=$KALDI_ROOT/idlak-voice-build/utils/hts_demo
RAW_DIR=$HTS_DEMO_LOCATION/HTS-demo_CMU-ARCTIC-SLT/data/raw/

mkdir -p rawaudio
for i in $RAW_DIR/*.raw; do
    a=`basename $i .raw`.wav 
    sox -q -r 48000 -c 1 -s -b 16 $i rawaudio/${a##cmu_us_arctic_}
done

audio_dir=rawaudio
label_dir=`echo $KALDI_ROOT/idlak-voice-build/idlak-scratch/en/ga/slt/cex_def/*/output/htslab/`

mkdir -p data/{train,dev,eval}
# Speaker ID
spks="slt" # can add any of awb, bdl, clb, jmk, ksp, rms
for spk in $spks; do


    # Create train, dev, eval sets
    dev_pat='slt_a0001_58?_???'
    train_pat='slt_?0001_???_???'
    #dev_pat='arctic_a0??2'
    #dev_rgx='arctic_a0..2'
    #eval_pat='arctic_b0??2'
    #eval_rgx='arctic_b0..2'
    #train_pat='arctic_????[^2]'
    #train_rgx='arctic_....[^2]'
    makeid="xargs -n 1 -I {} -n 1 awk -v lst={} -v spk=$spk BEGIN{print(gensub(\".*/([^.]*)[.].*\",\"\\\\1\",\"g\",lst),lst)}"
    #makelab="awk -v spk=$spk '{u=\$2;\$1=\"\";\$2=\"\";\$NF=\"\";print(\"<fileid id=\\\"\" spk \"_\" u \"\\\">\",substr(\$0,4,length(\$0)-5),\"</fileid>\")}'"
    #mkdir -p data/{train,dev,eval}

    find $audio_dir -iname "$train_pat".wav  | sort | $makeid > data/train/wav.scp
    find $audio_dir -iname "$dev_pat".wav    | sort | $makeid > data/dev/wav.scp

    find $label_dir -iname "$train_pat".wrd  | sort | xargs -n 1 -I {} -n 1 awk -v lst={} -v spk=$spk '{u=gensub(".*/([^.]*)[.].*", "\\1","g",lst); print("<fileid id=\"" u "\">", $0, "</fileid>")}' {} > data/train/text.xml
    find $label_dir -iname "$dev_pat".wrd    | sort | xargs -n 1 -I {} -n 1 awk -v lst={} -v spk=$spk '{u=gensub(".*/([^.]*)[.].*", "\\1","g",lst); print("<fileid id=\"" u "\">", $0, "</fileid>")}' {} > data/dev/text.xml
    


    #grep "$train_rgx" $label_dir/cmuarctic.data | sort | eval "$makelab" >> data/train/text.xml
    #grep "$dev_rgx"   $label_dir/cmuarctic.data | sort | eval "$makelab" >> data/dev/text.xml
    #grep "$eval_rgx"  $label_dir/cmuarctic.data | sort | eval "$makelab" >> data/eval/text.xml

    # Generate utt2spk / spk2utt info
    for step in train dev; do
	    cat data/$step/wav.scp | awk -v spk=$spk '{print $1, spk}' > data/$step/utt2spk
	    utt2spk_to_spk2utt.pl < data/$step/utt2spk > data/$step/spk2utt
    done
done

mkdir -p data/full
# This will be used for aligning the labels only
for k in text.xml wav.scp utt2spk; do
    cat data/{train,dev}/$k | sort -u > data/full/$k
done
utt2spk_to_spk2utt.pl < data/full/utt2spk > data/full/spk2utt

# Turn transcription into valid xml files
for step in full train dev; do
    cat data/$step/text.xml | awk 'BEGIN{print "<document>"}{print}END{print "</document>"}' > data/$step/text2
    mv data/$step/text2 data/$step/text.xml
done

# Alice test set
mkdir -p data/eval
cp $KALDI_ROOT/idlak-data/en/testdata/alice.xml data/eval/text.xml

############################################
##### Step 1: acoustic data generation #####
############################################

export featdir=$TMPDIR/dnn_feats/idlarctic

# Use kaldi to generate MFCC features for alignment
for step in full; do
    steps/make_mfcc.sh data/$step exp/make_mfcc/$step $featdir
    steps/compute_cmvn_stats.sh data/$step exp/make_mfcc/$step $featdir
done

# Use Kaldi + SPTK tools to generate F0 / BNDAP / MCEP
# NB: respective configs are in conf/pitch.conf, conf/bndap.conf, conf/mcep.conf
for step in dev train; do
    rm -f data/$step/feats.scp
    # Generate f0 features
    steps/make_pitch.sh   data/$step    exp/make_pitch/$step   $featdir;
    cp data/$step/pitch_feats.scp data/$step/feats.scp
    # Compute CMVN on pitch features, to estimate min_f0 (set as mean_f0 - 2*std_F0)
    steps/compute_cmvn_stats.sh data/$step    exp/compute_cmvn_pitch/$step   $featdir;
    # For bndap / mcep extraction to be successful, the frame-length must be adjusted
    # in relation to the minimum pitch frequency.
    # We therefore do something speaker specific using the mean / std deviation from
    # the pitch for each speaker.
    for spk in $spks; do
	min_f0=`copy-feats scp:"awk -v spk=$spk '(\\$1 == spk){print}' data/$step/cmvn.scp |" ark,t:- \
	    | awk '(NR == 2){n = \$NF; m = \$2 / n}(NR == 3){std = sqrt(\$2/n - m * m)}END{print m - 2*std}'`
	echo $min_f0
	# Rule of thumb recipe; probably try with other window sizes?
	bndapflen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 4.6 * 1000.0 / f0 + 0.5}'`
	mcepflen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 2.3 * 1000.0 / f0 + 0.5}'`
	f0flen=`awk -v f0=$min_f0 'BEGIN{printf "%d", 2.3 * 1000.0 / f0 + 0.5}'`
	echo "using wsizes: $bndapflen $mcepflen"
	subset_data_dir.sh --spk $spk data/$step 100000 data/${step}_$spk
	#cp data/$step/pitch_feats.scp data/${step}_$spk/
	# Regenerate pitch with more appropriate window
	steps/make_pitch.sh --frame_length $f0flen    data/${step}_$spk exp/make_pitch/${step}_$spk  $featdir;
	# Generate Band Aperiodicity feature
	steps/make_bndap.sh --frame_length $bndapflen data/${step}_$spk exp/make_bndap/${step}_$spk  $featdir
	# Generate Mel Cepstral features
	steps/make_mcep.sh  --frame_length $mcepflen --sample_frequency $srate data/${step}_$spk exp/make_mcep/${step}_$spk   $featdir	
    done
    # Merge features
    cat data/${step}_*/bndap_feats.scp > data/$step/bndap_feats.scp
    cat data/${step}_*/mcep_feats.scp > data/$step/mcep_feats.scp
    # Have to set the length tolerance to 1, as mcep files are a bit longer than the others for some reason
    paste-feats --length-tolerance=1 scp:data/$step/pitch_feats.scp scp:data/$step/mcep_feats.scp scp:data/$step/bndap_feats.scp ark,scp:$featdir/${step}_cmp_feats.ark,data/$step/feats.scp
    # Compute CMVN on whole feature set
    steps/compute_cmvn_stats.sh data/$step exp/compute_cmvn/$step   data/$step
done

############################################
#####      Step 2: label creation      #####
############################################

# We are using the idlak front-end for processing the text
tpdb=$KALDI_ROOT/idlak-data/en/ga/
dict=data/local/dict
for step in train dev eval full; do
    # Normalise text and generate phoneme information
    idlaktxp --pretty --tpdb=$tpdb data/$step/text.xml data/$step/text_norm.xml
    # Generate full labels
    #idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_norm.xml data/$step/text_full.xml
done
# Generate language models for alignment
mkdir -p $dict
python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 0 data/full/text_norm.xml data/full $dict

#######################################
## 3a: create kaldi forced alignment ##
#######################################

lang=data/lang
rm -rf $dict/lexiconp.txt $lang
utils/prepare_lang.sh --share-silence-phones true $dict "<OOV>" data/local/lang_tmp $lang
utils/validate_lang.pl $lang

# Now running the normal kaldi recipe for forced alignment
expa=exp-align
train=data/full
#test=data/eval_mfcc

steps/train_mono.sh  --nj 1 --cmd "$train_cmd" \
  $train $lang $expa/mono
steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
$train $lang $expa/mono $expa/mono_ali
steps/train_deltas.sh --cmd "$train_cmd" \
     5000 50000 $train $lang $expa/mono_ali $expa/tri1

steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
$train data/lang $expa/tri1 $expa/tri1_ali
steps/train_deltas.sh --cmd "$train_cmd" \
     5000 50000 $train $lang $expa/tri1_ali $expa/tri2

# Create alignments
steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
    $train $lang $expa/tri2 $expa/tri2_ali_full

steps/train_deltas.sh --cmd "$train_cmd" \
    --context-opts "--context-width=5 --central-position=2" \
    5000 50000 $train $lang $expa/tri2_ali_full $expa/quin

# Create alignments
steps/align_si.sh  --nj 1 --cmd "$train_cmd" \
    $train $lang $expa/quin $expa/quin_ali_full



################################
## 3b. Align with full labels ##
################################

# Convert to phone-state alignement
for step in full; do
    ali=$expa/tri2_ali_$step
    # Extract phone alignment
    ali-to-phones --per-frame $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:- \
	| utils/int2sym.pl -f 2- $lang/phones.txt > $ali/phones.txt
    # Extract state alignment
    ali-to-hmmstate $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:$ali/states.tra
    # Extract word alignment
    linear-to-nbest ark:"gunzip -c $ali/ali.*.gz|" \
	ark:"utils/sym2int.pl --map-oov 1669 -f 2- $lang/words.txt < data/$step/text |" '' '' ark:- \
	| lattice-align-words $lang/phones/word_boundary.int $ali/final.mdl ark:- ark:- \
	| nbest-to-ctm --frame-shift=$FRAMESHIFT --precision=3 ark:- - \
	| utils/int2sym.pl -f 5 $lang/words.txt > $ali/wrdalign.dat
    
    # Regenerate text output from alignment
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 1 "2:0.03,3:0.2" "4" $ali/phones.txt $ali/wrdalign.dat data/$step/text_align.xml

    # Generate corresponding quinphone full labels
    idlaktxp --pretty --tpdb=$tpdb data/$step/text_align.xml data/$step/text_anorm.xml
    idlakcex --pretty --cex-arch=default --tpdb=$tpdb data/$step/text_anorm.xml data/$step/text_afull.xml
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 2 data/$step/text_afull.xml data/$step/cex.ark > data/$step/cex_output_dump
    
    # Merge alignment with output from idlak cex front-end => gives you a nice vector
    make-fullctx-ali-dnn --max-sil-phone=15 $ali/final.mdl ark:"gunzip -c $ali/ali.*.gz|" ark,t:data/$step/cex.ark ark,t:data/$step/ali


    # UGLY convert alignment to features
    cat data/$step/ali \
	| awk '{print $1, "["; $1=""; na = split($0, a, ";"); for (i = 1; i < na; i++) print a[i]; print "]"}' \
	| copy-feats ark:- ark,scp:$featdir/in_feats_$step.ark,$featdir/in_feats_$step.scp
done

# Generate and align test set.
for step in eval; do
    idlaktxp --pretty --tpdb=$tpdb data/$step/text.xml - \
     | idlakcex --pretty --cex-arch=default --tpdb=$tpdb - data/$step/text_full.xml
    python $KALDI_ROOT/idlak-voice-build/utils/idlak_make_lang.py --mode 2 -r "alice" \
        data/$step/text_full.xml data/full/cex.ark.freq data/$step/cex.ark > data/$step/cex_output_dump

    # HACK: recover duration from HTS demo samples
    # NB: you need to have run the idlak test recipe all the way through!
   HTS_DEMO_LOCATION=$KALDI_ROOT/idlak-voice-build/utils/hts_demo/
   (
       echo '#!MLF!#';
       for i in $HTS_DEMO_LOCATION/HTS-demo_CMU-ARCTIC-SLT/gen/qst001/ver1/hts_engine/*.slab; do
           cat $i | awk -v id=`basename $i .slab` 'BEGIN{print "\"" id ".lab\""}{p = gensub("[^+-]*-([^+-]+)[+][^+-]*:.*", "\\1","g",$3); if (p == "ax") p = "ah"; print $1, $2, p "_" $4 }'; 
       done
   ) > data/$step/hts_lab.mlf

   # Align lab and idlak full labels
   python utils/map_mono_to_full.py data/$step/text_full.xml data/$step/hts_lab.mlf data/$step/labels_afull.mlf

   # Same as make-fullctx-ali-dnn, but using mlf as input
   python utils/make_fullctx_mlf_dnn.py data/$step/labels_afull.mlf data/$step/cex.ark data/$step/feat.ark
   copy-feats ark:data/$step/feat.ark ark,scp:$featdir/in_feats_$step.ark,$featdir/in_feats_$step.scp
   #cp $featdir/in_feats_$step.scp 
done

# Split in train / dev / eval
for step in train dev; do
    dir=lbldata/$step
    mkdir -p $dir
    cp data/$step/{utt2spk,spk2utt} $dir
    utils/filter_scp.pl $dir/utt2spk $featdir/in_feats_full.scp > $dir/feats.scp
    steps/compute_cmvn_stats.sh $dir $dir $dir
done

# Create test set
for step in eval; do
    dir=lbldata/$step
    mkdir -p $dir
    cp $featdir/in_feats_$step.scp $dir/feats.scp
    cut -d ' ' -f 1 $dir/feats.scp | awk -v spk=$spk '{print $1, spk}' > $dir/utt2spk
    utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
    steps/compute_cmvn_stats.sh $dir $dir $dir
done

##############################
## 4. Train DNN
##############################

exp=exp_dnn
acdir=data
lbldir=lbldata
#ensure consistency in lists
#for dir in $lbldir $acdir; do
for class in train dev; do
    cp $lbldir/$class/feats.scp $lbldir/$class/feats_full.scp
    cp $acdir/$class/feats.scp $acdir/$class/feats_full.scp
    cat $acdir/$class/feats_full.scp | awk -v lst=$lbldir/$class/feats_full.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $acdir/$class/feats.scp
    cat $lbldir/$class/feats_full.scp | awk -v lst=$acdir/$class/feats_full.scp 'BEGIN{ while (getline < lst) n[$1] = 1}{if (n[$1]) print}' > $lbldir/$class/feats.scp
done


# Very basic one for testing
mkdir -p $exp
dir=$exp/tts_dnn_train_3e
$cuda_cmd $dir/_train_nnet.log steps/train_nnet_basic.sh --config conf/3-layer-nn.conf --learn_rate 0.2 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --bunch_size 50 --mlpOption " " --hid-dim 300 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dir

# A bit more sophisticated, with splicing and more units
dir=$exp/tts_dnn_train_3_splice10
$cuda_cmd $dir/_train_nnet.log steps/train_nnet_basic.sh --config conf/3-layer-nn-splice10.conf --learn_rate 0.2 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --bunch_size 10 --mlpOption " " --hid-dim 500 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dir

# A bit more sophisticated, with deltas and more units
dir=$exp/tts_dnn_train_3_deltasc
rm -rf $dir
$cuda_cmd $dir/_train_nnet.log steps/train_nnet_basic.sh --delta_order 2 --config conf/3-layer-nn-splice5.conf --learn_rate 0.04 --momentum 0.1 --halving-factor 0.5 --min_iters 15 --randomize true --cache_size 50000 --bunch_size 200 --mlpOption " " --hid-dim 700 $lbldir/train $lbldir/dev $acdir/train $acdir/dev $dir

##############################
## 5. Synthesis
##############################

# Original samples:
mkdir -p exp_dnn/orig2/cmp exp_dnn/orig2/wav
copy-feats scp:data/eval/feats.scp ark,t:- | awk -v dir=exp_dnn/orig2/cmp/ '($2 == "["){if (out) close(out); out=dir $1 ".cmp";}($2 != "["){if ($NF == "]") $NF=""; print $0 > out}'
for cmp in exp_dnn/orig2/cmp/*.cmp; do
    utils/mlsa_synthesis_63_mlpg.sh --alpha 0.42 --fftlen 512 $cmp exp_dnn/orig2/wav/`basename $cmp .cmp`.wav
done

# Forward pass with test => creates $dir/tst_forward/cmp/*
utils/make_forward_fmllr.sh $dir $lbldir/eval $dir/tst_forward/ ""
# Synthesis
mkdir -p $dir/tst_forward/wav
for cmp in $dir/tst_forward/cmp/*.cmp; do
    utils/mlsa_synthesis_63_mlpg.sh --alpha 0.42 --fftlen 512 $cmp $dir/tst_forward/wav/`basename $cmp .cmp`.wav
done

# Variant with mlpg: simply use mean / variance from coefficients
copy-feats scp:data/train/feats.scp ark:- \
    | add-deltas --delta-order=2 ark:- ark:- \
    | compute-cmvn-stats --binary=false ark:- - \
    | awk '
(NR==2){count=$NF; for (i=1; i < NF; i++) mean[i] = $i / count}
(NR==3){if ($NF == "]") NF -= 1; for (i=1; i < NF; i++) var[i] = $i / count - mean[i] * mean[i]; nv = NF-1}
END{for (i = 1; i <= nv; i++) print mean[i], var[i]}' \
    > data/train/var_cmp.txt

utils/make_forward_fmllr.sh $dir $lbldir/eval $dir/tst_forward/ ""

# Using alice test set:
utils/make_forward_fmllr.sh $dir $lbldir/eval $dir/tst_forward/ ""
mkdir -p $dir/tst_forward/wav_mlpg/; for cmp in $dir/tst_forward/cmp/*.cmp; do
    utils/mlsa_synthesis_63_mlpg.sh --alpha 0.42 --fftlen 2048 --srate 48000 --bndap_order 25 --mcep_order 60 --delta_order 2 $cmp $dir/tst_forward/wav_mlpg/`basename $cmp .cmp`.wav data/train/var_cmp.txt
done
