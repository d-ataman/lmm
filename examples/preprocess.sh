HOME=/fs/baldur0/ataman

python=$HOME/anaconda3/bin/python
exp_dir=$HOME/experiments/tr-en
opennmt=$HOME/Char-NMT
src=tr
tgt=en

$python $opennmt/preprocess.py -train_src $exp_dir/train.$src -train_tgt $exp_dir/train.$tgt -valid_src $exp_dir/dev.$src -valid_tgt $exp_dir/dev.$tgt -save_data $exp_dir/iwslt -src_data_type text-trigram -tgt_data_type text-trigram -src_vocab_size 40000 -tgt_vocab_size 40000 -src_seq_length 100 -tgt_seq_length 100
