from minimda import MINIMDA
from torch import optim,nn
from tqdm import trange
from utils import k_matrix
import dgl
import networkx as nx
import copy
import numpy as np
import torch as th

def train(data,args):
    model = MINIMDA(args)
    optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    cross_entropy = nn.BCELoss()
    epochs = trange(args.epochs, desc='train')
    miRNA=data['ms']
    disease=data['ds']

    for _ in epochs:
        model.train()
        optimizer.zero_grad()

        mm_matrix = k_matrix(data['ms'], args.neighbor)
        dd_matrix = k_matrix(data['ds'], args.neighbor)
        mm_nx=nx.from_numpy_matrix(mm_matrix)
        dd_nx=nx.from_numpy_matrix(dd_matrix)
        mm_graph = dgl.from_networkx(mm_nx)
        dd_graph = dgl.from_networkx(dd_nx)

        md_copy = copy.deepcopy(data['train_md'])
        md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number
        md_graph = dgl.graph(
            (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
            num_nodes=args.miRNA_number + args.disease_number)

        miRNA_th=th.Tensor(miRNA)
        disease_th=th.Tensor(disease)
        train_samples_th = th.Tensor(data['train_samples']).float()

        train_score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['train_samples'])

        train_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2])
        train_loss.backward()
        # print(train_loss)
        optimizer.step()
    model.eval()
    score = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, data['unsamples'])
    score=score.detach().numpy()
    print(np.concatenate((data['m_num'][data['unsamples']],score),axis=1))
    return score