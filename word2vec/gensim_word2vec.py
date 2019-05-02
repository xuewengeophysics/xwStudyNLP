"""
主要参数介绍如下：
01) sentences：语料，可以是一个列表，或者从文件中遍历读出（word2vec.LineSentence(filename)）。
02) size：词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关；
    如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了；如果是超大的语料，建议增大维度。
03) window：即词向量上下文最大距离，window越大，则和某一词较远的词也会产生上下文关系。
    默认值为5，在实际使用中，可以根据实际的需求来动态调整这个window的大小。
　　如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5, 10]之间。
04) sg：如果是0则是CBOW模型；是1则是Skip-Gram模型；默认是0即CBOW模型。
05) hs：即word2vec的两个解法。如果是0，则是Negative Sampling；
    是1并且负采样个数negative大于0，则是Hierarchical Softmax。默认是0即Negative Sampling。
06) negative：即使用Negative Sampling时负采样的个数，默认是5。推荐在[3, 10]之间。
07) cbow_mean：仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。默认值也是1。
08) min_count：需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。
09) iter：随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
10) alpha：在随机梯度下降法中迭代的初始步长，默认是0.025。
11) min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步。
"""
gensim.models.word2vec.Word2Vec(
    sentences=None, 
    corpus_file=None, 
    size=100, 
    alpha=0.025, 
    window=5, 
    min_count=5, 
    max_vocab_size=None, 
    sample=0.001, 
    seed=1, 
    workers=3, 
    min_alpha=0.0001, 
    sg=0, 
    hs=0, 
    negative=5, 
    ns_exponent=0.75, 
    cbow_mean=1, 
    hashfxn=<built-in function hash>, 
    iter=5, 
    null_word=0, 
    trim_rule=None, 
    sorted_vocab=1, 
    batch_words=10000, 
    compute_loss=False, 
    callbacks=(), 
    max_final_vocab=None)
    
