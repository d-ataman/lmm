HOME=/home/ataman

python=$HOME/anaconda3/bin/python
exp_dir=$HOME/experiments/en-tr
opennmt=$HOME/lmm
src=en
tgt=tr

$python $opennmt/preprocess.py -train_src $exp_dir/train.$src -train_tgt $exp_dir/train.$tgt -valid_src $exp_dir/dev.$src -valid_tgt $exp_dir/dev.$tgt -save_data $exp_dir/iwslt -src_data_type words -tgt_data_type characters -src_vocab_size 40000 -tgt_vocab_size 40000 -src_seq_length 100 -tgt_seq_length 100
