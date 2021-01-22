# A Latent Morphology Model for Open-Vocabulary Neural Machine Translation


This software implements the Neural Machine Translation model based on Hierchical Character-based Decoding using Variational Inference.

## Options

### Hiearchical Decoder with Compositional Word Embeddings and Character-level Generation with Variational Inference 

  To activate the character-level decoder, select

  ```-tgt_data_type characters``` in the settings of preprocess.py and translate.py 

  and

  ```-decoder_type charrnn``` and ```-tgt_data_type characters```  in train.py
  
  The feature dimensions are hardcoded to 100 for the lemma and 10 for inflectional feature vectors, you can change this depending on your language or data size.

## Further information

For information about how to install and use OpenNMT-py:
[Full Documentation](http://opennmt.net/OpenNMT-py/)


## Citation

If you use this software, please cite:

@article{lmm,
  author    = {Duygu Ataman and
               Wilker Aziz and
               Alexandra Birch},
  title     = {A Latent Morphology Model for Open-Vocabulary Neural Machine Translation},
  booktitle = ICLR},
  year      = {2020},
}
```
