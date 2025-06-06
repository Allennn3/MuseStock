# 	Dynamic Graph Generation with Sentiment-Attentive Multimodal Fusion: A GNN Approach for Stock Prediction
Code detail for this paper.

## Environment
Python 3.8 & Pytorch 2.5.1 and more information is in the requirements.txt.

## Data collect
All the row data should be stored in  ./data. 
All the dataset are open-source datasets. You can download by yourself. Here are the links:
ACL18: https://github.com/yumoxu/stocknet-dataset
CMIN: https://github.com/BigRoddy/CMIN-Dataset

## Data process
After down the raw dataset. You need to process the data. Then you can cd into the data_process  folder and follow the following steps:
```sh
$ python price_process.py
$ python news_process.py
$ python sentiment_proces.py
```

After all the process, you will get the pkl to store the data.
And we provide a series of tools in the util.py. You can use them to help you understand the process.
After dataprocess  you can run the main.py to train the model. You can adjust your configuration in my_parser.py.

## Run
```sh
$ python main.py
```
Make sure that the GPU is used to reproduce our experiments.

