#encoding:utf-8
import tensorflow as tf

class Textcnn(object):

    def __init__(self, config):
        config = config
        # placeholders for input output and dropout
        self.input_x = tf.placeholder(tf.int32, shape=[None, config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, shape=[None, config.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # global_step 代表全局步数，比如在多少步该进行什么操作，现在神经网络训练到多少轮等等，类似于一个钟表。
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.device('/gpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[config.vocab_size, config.embedding_size],
                                             initializer=tf.constant_initializer(config.pre_training),
                                             trainable=False) # 是否使用预训练词向量，静态(False)或动态(默认True)
            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # conv2d操作接收一个4维的张量，维度分别代表batch, width, height 和 channel，这里的embedding不包含channel维度
            # 因此要手动添加，得到形状为[None, sequence_length, embedding_size, 1]的embedding层。
            self.embedding_expand = tf.expand_dims(embedding_input, -1)

        # 添加卷积层、池化层
        pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CNN layer
                # conv2d的卷积核形状[filter_height, filter_width, in_channels, out_channels]
                filter_shape = [filter_size, config.embedding_size, 1, config.num_filters]  # 卷积核维度

                # 设定卷积核的参数 W b
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name= 'W')  # 图变量初始化，截断正态分布
                b = tf.Variable(tf.constant(0.1, shape=[config.num_filters]), name='b')  # 生成常量
                conv = tf.nn.conv2d(self.embedding_expand, W, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                # nonlinearity activate funtion
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # global max pooling layer
                # 每个filter的max-pool输出形状:[batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, config.seq_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool')
                pooled_outputs.append(pooled)

        # combine all the pooled features
        # 有三种卷积核尺寸， 每个卷积核有num_filters个channel，因此一共有 num_filters * len(filter_sizes)个
        num_filter_total = config.num_filters * len(config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)  # 在第3个维度进行张量拼接
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filter_total])

        # add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            W_fc = tf.get_variable("W_fc", shape=[num_filter_total, config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            b_fc = tf.Variable(tf.constant(0.1, shape=[config.num_classes]), name='b_fc')
            self.logits = tf.matmul(self.h_drop, W_fc) + b_fc
            self.pro = tf.nn.softmax(self.logits)
            self.predicitions = tf.argmax(self.pro, 1, name='predictions')

        # calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            # 损失函数，交叉熵
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(loss)  # 对交叉熵取均值非常有必要
            l2_loss += tf.nn.l2_loss(W_fc)
            l2_loss += tf.nn.l2_loss(b_fc)
            self.loss = tf.reduce_mean(loss) + config.l2_reg_lambda * l2_loss

        # 优化器
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, config.clip)
            #对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        # accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(self.input_y, 1), self.predicitions)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float32'), name='accuracy')
