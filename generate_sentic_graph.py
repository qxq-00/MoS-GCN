# -*- coding: utf-8 -*-

import numpy as np
#import spacy
import pickle
import logging
import seaborn as sns
import matplotlib.pyplot as plt

#nlp = spacy.load('en_core_web_sm')
# 配置日志
logging.basicConfig(
    filename='process_sgraph.log',
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(text, aspect, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0
        if word in aspect:
            sentic += 1.0
        for j in range(seq_len):
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1

    return matrix

def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.sentic', 'wb')
    for i in range(0, 3, 3):
        print('matrix processing rate:', i, '/', len(lines), end='\r')
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right, aspect, senticNet)
        idx2graph[i] = adj_matrix
        
        logging.info(f'Index: {i}')
        logging.info(f'Text Left: {text_left}')
        logging.info(f'Text Right: {text_right}')
        logging.info(f'Aspect: {aspect}')
        logging.info(f'Adjacency Matrix: {adj_matrix}')
        logging.info(f'SenticNet: {senticNet}')

        plt.figure(figsize=(10, 10))
        sns.heatmap(adj_matrix, annot=True, cmap='coolwarm', linewidths=.5)
        plt.title('Heatmap Example')
        plt.xlabel('X Axis Label')
        plt.ylabel('Y Axis Label')
        plt.savefig('see1see.png')
    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
    fout.close() 

process('./datasets/semeval16/restaurant_train.raw')
'''
if __name__ == '__main__':
    print('process 1/10')
    process('./datasets/acl-14-short-data/train.raw')
    print('process 2/10')
    process('./datasets/acl-14-short-data/test.raw')
    print('process 3/10')
    process('./datasets/semeval14/restaurant_train.raw')
    print('process 4/10')
    process('./datasets/semeval14/restaurant_test.raw')
    print('process 5/10')
    process('./datasets/semeval14/laptop_train.raw')
    print('process 6/10')
    process('./datasets/semeval14/laptop_test.raw')
    print('process 7/10')
    process('./datasets/semeval15/restaurant_train.raw')
    print('process 8/10')
    process('./datasets/semeval15/restaurant_test.raw')
    print('process 9/10')
    process('./datasets/semeval16/restaurant_train.raw')
    print('process 10/10')
    process('./datasets/semeval16/restaurant_test.raw')
'''

