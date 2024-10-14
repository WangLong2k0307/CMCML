# CMCML

The source code for our paper "Information Processing and Management Category-guided Multi-interest Collaborative Metric Learning with Representation Uniformity Constraints". 

## Dependencies
All the experimental results of our paper are carried on a Ubuntu 18.04.4 server equipped with NVIDIA RTX3090 GPUs, Intel(R) Xeon(R) Silver 4314 CPU and 128GB size memory. In addition, our experiments were implemented based on python's pytorch framework, with the following specific experimental requirements,
- python=3.9.0
- pytorch=1.8.2
- numpy=1.24.2
- pandas=1.5.3
- scikit-learn
- tqdm
- toolz

## Datasets
In our paper, we evaluate our method on three datasets, including: 
- [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm)

- [Epinions](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm)

- [Tafeng](https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset
  )
  

![image-20241014204555003](C:/Users/15998/AppData/Roaming/Typora/typora-user-images/image-20241014204555003.png)


## Usage
1. Install the dependencies by 
```
pip install -r requirements.txt
```
2. Download  and process the datasets using 5-core method. 
3. Train  model by
```
python train_best.py --data_path=data/ciao --model=CMCML --samping_strategy=hard --k=5 --per_user_k=5 --reg=0.01 --reg=0.1
```
4. Test the model and output the results to an Excel file by
```
python test_max_diversification.py --data_path=data/ciao --model=CMCML --samping_strategy=hard --k=5 --per_user_k=5 --reg=0.01 --reg=0.1
```

## Questions
If you have any questions, please send an email to Wang Long (wanglong1327@link.tyut.edu.cn). 

## Credit
This repository is based on [DPCML](https://github.com/statusrank/DPCML). 