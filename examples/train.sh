HOME=/fs/baldur0/ataman

python=$HOME/anaconda3/bin/python
exp_dir=$HOME/experiments/tr-en
opennmt=$HOME/Char-NMT
src=tr
tgt=en

$python $opennmt/train.py -data $exp_dir/iwslt -epochs 50 -word_vec_size 512 -enc_layers 1 -dec_layers 2 -seed 1234 -rnn_size 512 -rnn_type GRU -encoder_type trigramrnn -decoder_type trigramrnn -model_type text-trigram -optim adam -learning_rate 0.0003 -learning_rate_decay 0.9 -dropout 0.2 -start_decay_at 20 -start_checkpoint_at 5 -save_model $exp_dir/model -gpu 1 -gpuid 1 -max_grad_norm 1 > log.out

