import argparse
from utils import get_data,data_processing
from train import train
import os

def result(args):
    data = get_data(args)
    args.miRNA_number = data['miRNA_number']
    args.disease_number = data['disease_number']
    data_processing(data,args)
    train(data,args)



os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train')
parser.add_argument('--fm', type=int, default=64, help='length of miRNA feature')
parser.add_argument('--fd', type=int, default=64, help='length of dataset feature')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument("--in_feats", type=int, default=64, help='Input layer dimensionalities.')
parser.add_argument("--hid_feats", type=int, default=64, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=64, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--gcn_bias", type=bool, default=True, help='gcn bias')
parser.add_argument("--gcn_batchnorm", type=bool, default=True, help='gcn batchnorm')
parser.add_argument("--gcn_activation", default='relu', help='gcn activation')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[64, 1], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
parser.add_argument('--dataset', default='HMDD v2.0', help='dataset')
parser.add_argument('--save_score', default='True', help='save_score')
parser.add_argument('--negative_rate', type=float,default=1.0, help='negative_rate')


args = parser.parse_args()
args.dd2=True
args.data_dir = 'data/' + args.dataset + '/'
args.result_dir = 'result/' + args.dataset + '/'
args.save_score = True if str(args.save_score) == 'True' else False


result(args)
