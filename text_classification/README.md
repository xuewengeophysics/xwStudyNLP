# Text Classification


本文是参考gaussic大牛的“text-classification-cnn-rnn”后，基于同样的数据集，嵌入词级别所做的CNN文本分类实验结果，gaussic大牛是基于字符级的；<br><br>
进行了第二版的更新：1.加入不同的卷积核；2.加入正则化；3.词仅为中文或英文，删掉文本中数字、符号等类型的词；4.删除长度为1的词；<br>
<br>
训练结果较第一版有所提升，验证集准确率从96.5%达到97.1%，测试准备率从96.7%达到97.2%。<br>
<br>


本实验的主要目是为了探究基于Word2vec训练的词向量嵌入CNN后，对模型的影响，实验结果得到的模型在验证集达到97.1%的效果，gaussic大牛为94.12%；<br><br>
更多详细可以阅读gaussic大牛的博客：[text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn)<br><br>

0 参考资料
=
1 [徐亮brightmart/text_classification](https://github.com/brightmart/text_classification)<br>
2 [cjymz886/text-cnn](https://github.com/cjymz886/text-cnn)<br>
3 [文本分类实战--从TFIDF到深度学习CNN系列效果对比（附代码）](https://github.com/lc222/text_classification_AI100)<br>
4 [gaussic/CNN-RNN中文文本分类](https://github.com/gaussic/text-classification-cnn-rnn)<br>
5 [clayandgithub/基于cnn的中文文本分类算法](https://github.com/clayandgithub/zh_cnn_text_classify)<br>
6 [pengming617/text_classification](https://github.com/pengming617/text_classification)<br>

1 环境
=
python3<br>
tensorflow 1.8以上GPU环境下<br>
gensim<br>
jieba<br>
scipy<br>
numpy<br>
scikit-learn<br>

2 CNN卷积神经网络
=
模型CNN配置的参数在text_model.py中，具体为：<br><br>

模型CNN大致结构为：<br><br>

3 数据集
=
本实验同样是使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议；<br><br>
文本类别涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；<br><br>
cnews.train.txt: 训练集(5000*10)<br>

cnews.val.txt: 验证集(500*10)<br>

cnews.test.txt: 测试集(1000*10)<br><br>

训练所用的数据，以及训练好的词向量可以下载：链接: [https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA](https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA)，密码: up9d<br><br>


4 预处理
=
本实验主要对训练文本进行分词处理，一来要分词训练词向量，二来输入模型的以词向量的形式；<br><br>
另外，词仅为中文或英文，词的长度大于1；<br><br>
处理的程序都放在loader.py文件中；<br><br>


5 运行步骤
=
python train_word2vec.py，对训练数据进行分词，利用Word2vec训练词向量(vector_word.txt)<br><br>
python text_train.py，进行训练模型<br><br>
python text_test.py，对模型进行测试<br><br>
python text_predict.py，提供模型的预测<br><br>


6 训练结果
=
运行：python text_train.py<br><br>
本实验经过6轮的迭代，满足终止条件结束，在global_step=2000时在验证集得到最佳效果97.1%<br><br>


7 测试结果
=
运行：python text_test.py<br><br>


## TextCNN

## Transformer


