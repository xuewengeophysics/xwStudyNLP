# coding: utf-8

import sys
from collections import Counter

import numpy as np
import tensorflow.contrib.keras as kr


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，使用字符级的表示"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)    # Counter（计数器）是对字典的补充，用于追踪值的出现次数
    count_pairs = counter.most_common(vocab_size - 1)    # 前n个出现频率最高的元素以及它们对于的次数，用元组('a', 3)表示
    words, _ = list(zip(*count_pairs))    # 将对象中对应的元素打包成一个个元组，返回由这些元组组成的列表word及出现次数
    words = ['<PAD>'] + list(words)    # 添加一个 <PAD> ，用于将所有文本pad为同一长度
    with open(vocab_dir, mode='w', encoding='utf-8') as fw:
        fw.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表，转换为{词: id}表示"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_dir,  'r', encoding='utf-8', errors='ignore') as fp:
        words = [_.strip() for _ in fp.readlines()]    # 删除末尾的"\n"
    word_to_id = dict(zip(words, range(len(words))))    # 将词表转换为词ID，返回字典{"word", "ID"}
    return words, word_to_id


def read_category():
    """读取分类目录，转换为{类别: id}表示"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))    # 将标签转换为标签ID，返回字典{"标签", "ID"}

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将数据集从文字转换为固定长度的id序列表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])    # 该文档内容的ID表示
        label_id.append(cat_to_id[labels[i]])    # 该文档所属类别的ID号

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """为神经网络准备经过shuffle的批次训练数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1    # 训练批次

    indices = np.random.permutation(np.arange(data_len))    # 打乱数据
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):    # 生成批次训练数据
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        # 类似return的关键字，返回生成器；调用此函数时，函数内部的代码并不立马执行，当使用for进行迭代时代码才会执行
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
        
