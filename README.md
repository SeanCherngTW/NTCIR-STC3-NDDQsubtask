# NTCIR-STC3-NDDQsubtask 
Works of NTCIR14 STC-3 Nugget Detection &amp; Dialogue Quality tasks [(link)](https://sakai-lab.github.io/stc3-dataset/)

## Directory
/bert: 存放bert的原始碼 <br/>
/BertPretrainModel: google訓練好的bert的pretrain model <br/>
/PickleBert: 把NTCIR的資料集處理成bert的形式後存下來的檔案 <br/>
/PickleCorpus: 把NTCIR的資料集處理成比較容易給model用的格式後存下的檔案 <br/>
/PickleResult: ND&DQ subtask測試的結果的pickle檔 <br/>
bert.ipynb: NTCIR格式前處理成bert格式 <br/>
<br/>
param.py: 訓練時用到的參數 <br/>
stcevaluation.py: 評估nugget detection和dialogue quality任務的結果 <br/>
stctokenizer.py: 分詞用的工具 <br/>
<br/>
## Experiments
測試Model時共分為三類 <br/>
1. 使用NTCIR-STC3 word embedding + softmax 的測試
2. 使用BERT word embedding + softmax 的測試
3. 使用BERT word embedding + CRF 的測試 (僅限nugget detection任務)

## Files
會用到的檔案分別如下
1. 使用NTCIR-STC3 word embedding + softmax 的測試 <br/>
Hierarchical_model.ipynb: 主程式 <br/>
datahelper.py: 資料前處理/後處理的class <br/>
dialogquality.py: dialogue quality用的模型、loss function等 <br/>
dialogquality_ndfeature.py: dialogue quality 用的模型、loss function等，但包含了nugget detection的結果作為特徵 <br/>
nuggetdetection.py: nugget detection用的模型、loss function等 <br/>
stc_train.py: tensorflow建立graph用、以及讀入資料進行訓練用

2. 使用BERT word embedding + softmax 的測試 <br/>
Hierarchical_BERT_model.ipynb: 主程式 <br/>
datahelper.py: 資料前處理/後處理的class <br/>
dialogquality_ndfeatureBERT.py: dialogue quality 用的模型、loss function等，包含了nugget detection的結果作為特徵(必須以BERT embedding作為輸入) <br/>
dialogqualityBERT.py: dialogue quality用的模型、loss function等(必須以BERT embedding作為輸入) <br/>
nuggetdetectionBERT.py: nugget detection用的模型、loss function等(必須以BERT embedding作為輸入) <br/>
stc_trainBERT.py: tensorflow建立graph用、以及讀入資料進行訓練用(必須以BERT embedding作為輸入) <br/>

3. 使用BERT word embedding + CRF 的測試 (僅限nugget detection任務) <br/>
Hierarchical_CRF_model.ipynb: 主程式 <br/>
datahelperCRF.py: 資料前處理/後處理的class，轉成CRF格式 <br/>
nuggetdetectionCRF.py: nugget detection用的模型、loss function等(必須以BERT embedding作為輸入，CRF為輸出) <br/>
stc_trainCRF.py: tensorflow建立graph用、以及讀入資料進行訓練用(必須以BERT embedding作為輸入，CRF為輸出) <br/>

## Result
