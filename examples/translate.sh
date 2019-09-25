HOME=/fs/baldur0/ataman

python=$HOME/anaconda3/bin/python
exp_dir=$HOME/experiments/tr-en
opennmt=$HOME/Char-NMT
src=tr
tgt=en

$python $opennmt/translate.py -model $exp_dir/model_acc_60.91_ppl_6.65_e92.pt -src_data_type text-trigram -tgt_data_type text-trigram -src $exp_dir/test.$src -output $exp_dir/test.output.$src -gpu 0
sed 's/ @@//g' $exp_dir/test.output.$src > $exp_dir/test.output.postprocessed.$src

$bleu/multi-bleu.perl $exp_dir/test.$tgt < $exp_dir/test.output.postprocessed.$src

