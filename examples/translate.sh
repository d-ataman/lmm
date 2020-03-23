HOME=/home/ataman

python=$HOME/anaconda3/bin/python
exp_dir=$HOME/experiments/en-tr
opennmt=$HOME/lmm
src=en
tgt=tr

$python $opennmt/translate.py -model $exp_dir/model_acc_60.91_ppl_6.65_e92.pt -src_data_type words -tgt_data_type characters -src $exp_dir/test.$src -output $exp_dir/test.output.$src -gpu 0


