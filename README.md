# Jointly learning to align and convert graphemes to phonemes with neural attention models 
Grapheme-to-Phoneme (G2P) conversion using attention based encoder-decoder models

# Dependencies
* Tensorflow == 1.0.0
* Bunch
* Editdistance

# Evaluation Datasets
We used the following datasets provided by Stanley Chen (stanchen@us.ibm.com):
* CMUDict 
* Pronlex
* NetTalk

*Note - For CMUDict, it might be a good idea to use the newer version from here - https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict*

# Steps
* Prepare data:
```
python data_utils.py -data_dir DATA_DIR [-{train,dev,test}_file] {TRAIN,DEV,TEST}_FILE
```
* Train/Eval models
```
python g2p.py -data_dir DATA_DIR -tb_dir BASE_MODEL_DIR [-eval]
```

# Reference
[*Jointly learning to align and convert graphemes to phonemes with neural attention models*](https://arxiv.org/pdf/1610.06540v1.pdf) by [Shubham Toshniwal](https://ttic.uchicago.edu/~shtoshni/) and [Karen Livescu](https://ttic.uchicago.edu/~klivescu/).

Here's the [[BIBTEX](https://ttic.uchicago.edu/~shtoshni/papers/g2p_slt.bib)] entry for citation ease. 
