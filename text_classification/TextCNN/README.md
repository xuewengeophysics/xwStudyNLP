# TextCNN

本文是参考gaussic大牛的“text-classification-cnn-rnn”后，基于同样的数据集，嵌入词级别所做的CNN文本分类实验结果，gaussic大牛是基于字符级的；<br><br>
进行了第二版的更新：1.加入不同的卷积核；2.加入正则化；3.词仅为中文或英文，删掉文本中数字、符号等类型的词；4.删除长度为1的词；<br>
<br>
训练结果较第一版有所提升，验证集准确率从96.5%达到97.8%，测试准备率从96.7%达到97.2%。<br>
<br>


本实验的主要目是为了探究基于Word2vec训练的词向量嵌入CNN后，对模型的影响，实验结果得到的模型在验证集达到97.8%的效果，gaussic大牛为94.12%；<br><br>
更多详细可以阅读gaussic大牛的博客：[text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)<br><br>

使用TextCNN进行中文文本分类

## 数据预处理流程
![image](https://github.com/xuewengeophysics/xwStudyNLP/blob/master/text_classification/TextCNN/images/%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86%E6%B5%81%E7%A8%8B.png)

## TextCNN模型网络结构
![image](https://github.com/xuewengeophysics/xwStudyNLP/blob/master/text_classification/TextCNN/images/TextCNN%E6%A8%A1%E5%9E%8B.png)

## 参考资料

1 [文献Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)<br>
2 [文献Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626)<br>
3 [大牛dennybritz的博客Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)<br>
4 [源码dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)<br>
5 [源码gaussic/text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)<br>
6 [源码cjymz886/text-cnn](https://github.com/cjymz886/text-cnn)<br>
7 [源码NLPxiaoxu/Easy_TextCnn_Rnn](https://github.com/NLPxiaoxu/Easy_TextCnn_Rnn)<br>
8 [源码YCG09/tf-text-classification](https://github.com/YCG09/tf-text-classification)<br>
9 [源码pengming617/text_classification](https://github.com/pengming617/text_classification)<br>
