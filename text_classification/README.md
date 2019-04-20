# Text Classification

0 参考资料
=
### 1、论文
1 [[Kamran Kowsari et al. 2019]Text Classification Algorithms: A Survey](https://arxiv.org/abs/1904.08067)<br>

### 2、代码
1 [徐亮brightmart/text_classification](https://github.com/brightmart/text_classification)<br>
2 [cjymz886/text-cnn](https://github.com/cjymz886/text-cnn)<br>
3 [文本分类实战--从TFIDF到深度学习CNN系列效果对比（附代码）](https://github.com/lc222/text_classification_AI100)<br>
4 [gaussic/CNN-RNN中文文本分类](https://github.com/gaussic/text-classification-cnn-rnn)<br>
5 [clayandgithub/基于cnn的中文文本分类算法](https://github.com/clayandgithub/zh_cnn_text_classify)<br>
6 [pengming617/text_classification](https://github.com/pengming617/text_classification)<br>
7 [roomylee/rcnn-text-classification](https://github.com/roomylee/rcnn-text-classification)<br>
8 [jiangxinyang227/textClassifier](https://github.com/jiangxinyang227/textClassifier)<br>

1 数据集
=
本实验使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。<br>

文本类别涉及10个类别：categories = \['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']，每个分类6500条数据；<br>
cnews.train.txt: 训练集(5000*10)<br>

cnews.val.txt: 验证集(500*10)<br>

cnews.test.txt: 测试集(1000*10)<br>

数据下载：训练数据以及训练好的词向量：<br>
链接: [https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA](https://pan.baidu.com/s/1DOgxlY42roBpOKAMKPPKWA)<br>
密码: up9d<br><br>

2 预处理
=
本实验主要对训练文本进行分词处理，一来要分词训练词向量，二来输入模型的以词向量的形式；<br><br>
另外，词仅为中文或英文，词的长度大于1；<br><br>
处理的程序都放在loader.py文件中；<br><br>

3 运行步骤
=
python train_word2vec.py，对训练数据进行分词，利用Word2vec训练词向量(vector_word.txt)<br><br>
python text_train.py，进行训练模型<br><br>
python text_test.py，对模型进行测试<br><br>
python text_predict.py，提供模型的预测<br><br>


## TextCNN

## Transformer


