# NTCIR-STC3-NDDQsubtask 
Works of NTCIR14 STC-3 Nugget Detection &amp; Dialogue Quality tasks [(link)](https://sakai-lab.github.io/stc3-dataset/) <br/>
Proposed paper: [arXiv: 1907.03070](https://arxiv.org/abs/1907.03070)

## Directories
+ /bert: Source code of BERT <br/>
+ /BertPretrainModel: Pretrained model of BERT <br/>
+ /PickleCorpus: NTCIR-STC3 corpus preprocessed into embeddings <br/>
+ /PickleBert: NTCIR-STC3 embeddings preprocessed into BERT format <br/>
+ /PickleResult: The result of ND&DQ subtask <br/>

## Files
+ bert.ipynb: Preprocess NTCIR-STC3 embeddings to BERT format <br/>
+ param.py: parameters for models <br/>
+ stcevaluation.py: Evaluation methods provided by NTCIR-14 <br/>
+ stctokenizer.py: Tokenizer for this task <br/>

## Models
There are 3 types of models
1. Using word embedding trained by NTCIR-STC3 corpus, and use softmax function as the final layer (to fit the evaluation of NTCIR-14)
    + You may download NTCIR-STC3 word embedding [here](https://drive.google.com/drive/folders/12kBvfxXrJoul3137c16DKwWikq1-jQ_j?usp=sharing)
2. Using BERT as sentence embedding, and use softmax function as the final layer 
3. Using BERT as sentence embedding, and use CRF the final layer (ND subtask only)

### Note of different models
1. NTCIR-STC3 word embedding + softmax (Input should be NTCIR-STC3 word embedding format)
    + Hierarchical_model.ipynb: Main function
    + datahelper.py: Data processing
    + nuggetdetection.py: Model and loss function for ND subtask
    + dialogquality.py: Model and loss function for DQ subtask
    + dialogquality_ndfeature.py: Model and loss function for DQ subtask with ND result as feature
    + stc_train.py: Tensorflow graph

2. BERT sentence embedding + softmax (Input should be BERT sentence embedding format)
    + Hierarchical_BERT_model.ipynb: Main function
    + datahelper.py: Data processing
    + nuggetdetectionBERT.py: Model and loss function for ND subtask
    + dialogquality_ndfeatureBERT.py: Model and loss function for DQ subtask with ND result as feature
    + dialogqualityBERT.py: Model and loss function for DQ subtask
    + stc_trainBERT.py: Tensorflow graph

3. BERT sentence embedding + CRF (Input should be BERT sentence embedding format)
    + Hierarchical_CRF_model.ipynb: Main function
    + datahelperCRF.py: Data processing for CRF
    + nuggetdetectionCRF.py: Model and loss function for ND subtask (Output: CRF)
    + stc_trainCRF.py: Tensorflow graph

## Evaluation
+ Dialogue Quality:
    + NMD: Normalised Match Distance.
    + RSNOD: Root Symmetric Normalised Order-aware Divergence
+ Nugget Detection:
    + RNSS: Root Normalised Sum of Squared errors
    + JSD: Jensen-Shannon divergence

[Proposed by sakai lab](https://sakai-lab.github.io/stc3-dataset/)

## Result
Please check out [here](https://arxiv.org/abs/1907.03070)