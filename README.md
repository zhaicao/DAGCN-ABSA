# DAGCN: Dual-channel and Aspect-aware Graph Convolutional Network for Aspect-based Sentiment Analysis in Computational Social Systems

This repository contains the code for the paper "[DAGCN: Dual-channel and Aspect-aware Graph Convolutional Network for Aspect-based Sentiment Analysis in Computational Social Systems](https://ieeexplore.ieee.org/document/10588956)", IEEE Transactions on Computational Social Systems.

## Requirements

- numpy==1.23.5
- pandas==2.0.3
- torch==2.1.2+cuda
- transformers==4.42.3

To install requirements, run `pip install -r requirements.txt`.

## Preparation

1. Prepare data 
   
   - Restaurants, Laptop, Twitter and MAMS datasets. (We provide the parsed data at directory `dataset`)

   - Downloading Glove embeddings ([glove.840B.300d.zip](https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.840B.300d.zip)), and put it into `glove` directory after unzipping the file.
   
   - Downloading pretrained BERT ([bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/main)) and put it into `bert-base-uncased` folder. 

2. Build vocabulary

   ```
   bash build_vocab.sh
   ```

## Training

   Go to Corresponding directory and run scripts:

   ``` 
   sh run.sh
   ```

   The saved model and training logs will be stored at directory `results` automatically.

## Results

### GloVe-based Model

|Database|  Acc  | F1  | Log | 
|  :----:  | :----:  |:---:|  :----:  |
| Res14   | 84.90 | 78.34 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcn-restaurant-2024-01-04_10-48-36.log) |
| Laptop  | 79.59 | 76.60 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcn-laptop-2023-12-15_08-05-49.log) |
| Twitter | 76.96 | 75.92 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcn-twitter-2024-01-04_20-15-11.log) | 
| MAMS    | 81.96 | 81.09 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcn-mams-2023-12-20_18-37-58.log) |

### BERT-based Model

|Database|  Acc  | F1  | Log |
|  :----:  | :----:  |:---:|  :----:  | 
| Res14  | 87.67 | 81.56 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcnbert-restaurant-2023-12-12_14-28-32.log) |
| Laptop | 81.49 | 78.45 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcnbert-laptop-2024-01-04_12-20-21.log) |
| Twitter| 78.14 | 77.61 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcnbert-twitter-2024-01-07_10-42-18.log) |
| MAMS   | 85.03 | 84.63 | [log](https://github.com/zhaicao/DAGCN-ABSA/blob/master/logs/dagcnbert-mams-2023-12-19_23-17-33.log) |


## References
If you find this work useful, please cite as following.
```
@ARTICLE{10588956,
  author={Wanneng Shu and Cao Zhai and Ke Yu},
  journal={IEEE Transactions on Computational Social Systems}, 
  title={DAGCN: Dual-Channel and Aspect-Aware Graph Convolutional Network for Aspect-Based Sentiment Analysis in Computational Social Systems}, 
  year={2024},
  doi={10.1109/TCSS.2024.3418472}}
```
### Credits
The code and datasets in this repository are based on [DualGCN_ABSA](https://github.com/CCChenhao997/DualGCN-ABSA) and [R-GAT](https://github.com/goodbai-nlp/RGAT-ABSA), and we thank them.

