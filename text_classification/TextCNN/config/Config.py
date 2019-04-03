class TextcnnConfig(object):
    """Textcnn配置参数"""

    # 文本参数
    seq_length = 600  # max length of sentence
    num_classes = 10  # number of labels
    vocab_size = 8000  # number of vocabulary

    # 模型参数
    embedding_size = 100  # dimension of word embedding
    num_filters = 128  # number of convolution kernel
    kernel_size = [2, 3, 4]  # size of convolution kernel
    pre_trianing = None  # use vector_char trained by word2vec

    # 训练参数
    batch_size = 8  # 每批训练大小
    num_epochs = 5  # 总迭代轮次
    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率
    lr_decay = 0.9  # learning rate decay
    clip = 7.0   # gradient clipping threshold
    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
