# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output






class RelationalGraphConvLayer(nn.Module):
    def __init__(self, num_rel, input_size, output_size, bias=True):
        super(RelationalGraphConvLayer, self).__init__()
        self.num_rel = num_rel
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.FloatTensor(self.num_rel, self.input_size, self.output_size))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_size))
        else:
            self.register_parameter("bias", None)

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = mx.sum(dim=2)  # Compute row sums along the last dimension
        r_inv = rowsum.pow(-1)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag_embed(r_inv)  # Create a batch of diagonal matrices
        mx = torch.matmul(r_mat_inv, mx)
        return mx

    def forward(self, text, adj):
        weights = self.weight.view(self.num_rel * self.input_size, self.output_size)  # r*input_size, output_size
        supports = []
        for i in range(self.num_rel):
            hidden = torch.bmm(self.normalize(adj[:, i]), text)
            supports.append(hidden)
        tmp = torch.cat(supports, dim=-1)
        output = torch.matmul(tmp.float(), weights)  # batch_size, seq_len, output_size)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



class SenticGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(SenticGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.bz = opt.batch_size
        self.pd = opt.polarities_dim
        self.simi_weight = nn.Parameter(torch.randn(1))
        
        # self.gc1 = RelationalGraphConvLayer(5, opt.bert_dim, opt.bert_dim)
        # self.gc2 = RelationalGraphConvLayer(5, opt.bert_dim, opt.bert_dim)
        # self.gc3 = RelationalGraphConvLayer(5, opt.bert_dim, opt.bert_dim)
        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

        self.ln1 = nn.Linear(opt.batch_size, opt.batch_size)
        self.ln2 = nn.Linear(opt.batch_size, opt.polarities_dim)

    def sinx(self, x):
        return torch.sin(x*torch.pi/2) ** 10
    
    def full_bz(self, x):
        return len(x) == self.bz

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        #batch_size = len(x)
        #seq_len = len(x[1])
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1,self.opt.max_seq_len)):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], min(aspect_double_idx[i,1]+1, self.opt.max_seq_len)):
                mask[i].append(1)
            for j in range(min(aspect_double_idx[i,1]+1, self.opt.max_seq_len), seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        # ['text_bert_indices', 'text_indices', 'aspect_indices', 'bert_segments_indices', 'left_indices', 'sdat_graph'],
        idx, text_bert_indices, text_indices, aspect_indices, bert_segments_ids, left_indices, adj, disgraph, simi = inputs

        '''
        tensor([2036,  258, 1503, 2148,  501, 1715, 1487, 1832, 1603, 1063, 1300, 1145,
         298, 1138, 1632, 1856])
        '''

        '''
        text_bert_indices:torch.Size([16, 85])
        text_indices:torch.Size([16, 85])
        aspect_indices:torch.Size([16, 85])
        bert_segments_ids:torch.Size([16, 85])
        left_indices:torch.Size([16, 85])
        adj:torch.Size([16, 85, 85])
        '''
        bz_simi = self.sinx(simi[:, idx])
        if self.full_bz(idx):
            bz_simi = self.ln1(bz_simi)
            bz_simi = self.ln2(bz_simi)

        # bz_simi = simi[:, idx]
        #print(bz_simi)
        #input("waiting")

        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        #text = self.embed(text_indices)
        #text = self.text_embed_dropout(text)
        #text_out, (_, _) = self.text_lstm(text, text_len)

        encoder_layer = self.bert(input_ids=text_bert_indices, token_type_ids=bert_segments_ids)
        text_out = encoder_layer.last_hidden_state
        # hidden = output.last_hidden_state

        adj = adj * disgraph
        
        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc1(text_out, adj))
        #x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc4(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc5(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc6(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc7(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))
        #x = F.relu(self.gc8(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))

        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        
        if self.full_bz(idx):
        #    output = torch.matmul(self.simi_weight*bz_simi, output)
            output = output + bz_simi
        # print(f"simi para:{self.simi_weight}")
        '''
        torch.Size([16, 3])
        tensor([[ 0.3404,  0.7875, -0.8333],
        [ 0.1361,  0.8480, -0.5480],
        [ 0.3900,  0.5184, -0.3744],
        [ 0.4076,  0.8030, -0.3527],
        [ 0.3156,  0.8554, -0.5494],
        [ 0.2404,  0.7887, -0.5691],
        [ 0.4774,  0.6426, -0.3528],
        [ 0.3617,  0.7314, -0.4891],
        [ 0.2651,  0.7080, -0.5309],
        [ 0.3329,  0.8258, -0.4727],
        [ 0.1603,  0.7533, -0.3999],
        [ 0.4538,  0.9664, -0.5118],
        [ 0.5126,  0.8757, -0.6500],
        [ 0.5636,  0.9068, -0.0464],
        [ 0.4341,  0.6120, -0.2815],
        [ 0.8755,  0.3630, -0.1973]], grad_fn=<AddmmBackward0>)
        
        '''
        return output
