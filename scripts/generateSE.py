import node2vec
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import pickle

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
window_size = 10
iter = 1000

Adj_file = '../data/METR-LA/adj_mx.pkl'
SE_file = '../data/METR-LA/SE.txt'
Edgelist_file = '../data/METR-LA/edgelist.txt'

# Adj_file = '../data/PEMS-BAY/adj_mx_bay.pkl'
# SE_file = '../data/PEMS-BAY/SE.txt'
# Edgelist_file = '../data/PEMS-BAY/edgelist.txt'

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def read_graph(edgelist):
    G = nx.read_edgelist(
        edgelist, nodetype=int, data=(('weight',float),),
        create_using=nx.DiGraph())
    return G

def learn_embeddings(walks, dimensions, output_file):
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks, size = dimensions, window = 10, min_count=0, sg=1,
        workers = 8, iter = iter)
    model.wv.save_word2vec_format(output_file)
    return

def generateEdgeListFile(Adj_file, Edgelist_file):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(Adj_file)
    num_nodes = adj_mx.shape[0]
    with open(Edgelist_file, 'w') as f:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx[i][j] > 0:
                    f.write(str(i) + ' ' + str(j) + ' ' + str(adj_mx[i][j]) + '\r\n')

generateEdgeListFile(Adj_file, Edgelist_file)
nx_G = read_graph(Edgelist_file)
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
