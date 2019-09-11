# Kaggle-
这事关于对kaggle数据集的处理，使数据能够喂给网络。另外有关于数据处理的 https://pytorch.org/tutorials/beginner/data_loading_tutorial.html。

1. 训练集和测试集的数据读取
train=pd.read_csv(‘train.csv’,index_col=0)
test=pd.read_csv(‘test.csv’,index_col=0)
y_test=pd.read_csv(‘sample_submission.csv’,index_col=0)

2.数据合并

3.删除缺失样本数据

4.将数值型和字符型特征值分开

5.数据集进行onehot哑编码

6.数据分割(之前把训练集和测试集合并在一起了，现在把他们分开）
