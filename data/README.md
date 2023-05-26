### 数据说明

一共包含三个文件夹：

- Pitt，ADReSS_train,ADReSS_test三个数据集的文件夹
- 每个文件夹包含三个csv文件，分别是
  数据集名称.csv,     此文件是cha文件转换后的原始文件
  数据集名称+'_data.csv', _   此文件是将文件去除non-ascii字符后的文件，并且只剩余：序号，text_data，label三列数据
  数据集名称+'_data_label.csv'   此文件是最终训练使用文件，包含三列:序号，text_data,label
- 其中最终使用文件的label中0代表正常人，1代表病人

### Data description

A total of three folders are included:

- Pitt, ADReSS_train,ADReSS_test folders for three datasets
- Each folder contains three csv files, which are
  dataset name.csv, this file is the original file after cha file conversion
  dataset name + '_data.csv', _ this file is the file after removing non-ascii characters, and only three columns of data remain: serial number, text_data, label
  dataset name + '_data_label.csv' This file is the final training file, containing three columns: serial number, text_data, label
- where 0 represents normal people and 1 represents patients in the label of the end-use file

运行 kfold_split.py 得到交叉验证所用的训练集划分 10fold

```python
python kfold_split.py
```