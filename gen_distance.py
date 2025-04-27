import spacy
import numpy as np
from spacy.tokens import Doc
from transformers import BertTokenizer
import pickle
from tqdm import tqdm
import benepar

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


# spaCy + Berkeley
nlp = spacy.load('en_core_web_md')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
# benepar.download('benepar_en3')
# nlp.add_pipe("benepar", config={"model": "benepar_en3"})
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})
# BERT
model_path = '../bertmodel'
tokenizer = BertTokenizer.from_pretrained(model_path)


def map_bert_2D(ori_adj, text):   # 词语切割，矩阵延展
    words = text.split()
    bert_tokens = []
    bert_map = []
    for src_i, word in enumerate(words):
        for subword in tokenizer.tokenize(word):
            bert_tokens.append(subword)  # * ['expand', '##able', 'highly', 'like', '##ing']
            bert_map.append(src_i)  # * [0, 0, 1, 2, 2]

    truncate_tok_len = len(bert_tokens)
    bert_adj = np.zeros((truncate_tok_len, truncate_tok_len), dtype='float32')
    for i in range(truncate_tok_len):
        for j in range(truncate_tok_len):
            bert_adj[i][j] = ori_adj[bert_map[i]][bert_map[j]]
    return bert_adj

def get_dis_graph(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for sent in tokens.sents:
        for cons in sent._.constituents:
            if len(cons) == 1:
                continue
            matrix[cons.start:cons.end, cons.start:cons.end] += np.ones([len(cons), len(cons)])

    hops_matrix = np.amax(matrix, axis=1, keepdims=True) - matrix  # hops
    dis_matrix = 2 - hops_matrix / (np.amax(hops_matrix, axis=1, keepdims=True) + 1)
    
    np.fill_diagonal(dis_matrix, 1)
    mask = (np.zeros_like(dis_matrix) != dis_matrix).astype('float32')
    dis_matrix = dis_matrix * mask
    dis_matrix = map_bert_2D(dis_matrix, text)

    return dis_matrix



def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()

    idx2graph_dis = {}
    fout  = open(filename + '.dis_graph', 'wb')


    for i in tqdm(range(0, len(lines), 3)):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        dis_matrix = get_dis_graph(text_left+' '+aspect+' '+text_right)
        idx2graph_dis[i] = dis_matrix


    pickle.dump(idx2graph_dis, fout)
    fout.close()

if __name__ == "__main__":
    process('datasets\semeval14\laptop_train.raw')
    process('datasets\semeval14\laptop_test.raw')