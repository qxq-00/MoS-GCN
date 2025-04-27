# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import logging

nlp = spacy.load('en_core_web_sm')
# 配置日志
logging.basicConfig(
    filename='process.log',
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        print(token.children)
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1

    return matrix



def process(filename):

#    print('start reading')
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines() 
    fin.close()
#    print('finish reading')
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, 15, 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix

        logging.info(f'Index: {i}')
        logging.info(f'Text Left: {text_left}')
        logging.info(f'Text Right: {text_right}')
        logging.info(f'Aspect: {aspect}')
        logging.info(f'Adjacency Matrix: {adj_matrix}')
    pickle.dump(idx2graph, fout)        
    fout.close() 
#    print('finish fout')

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
