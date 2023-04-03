from neural_nets.RankNet import RankNet
from neural_nets.SimpleNet import SimpleNet
import torch.nn as nn
from rankers import simple_ranker, pairwise_ranker
from metrics import ndsg

EXPERIMENTS = {
    'RankNet': {
        'train_folder': '../data/train/PairwiseMQ2008',
        'experiment_name': 'rank_net',
        'ranker': pairwise_ranker,
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 10,
        'model_name': 'rank_net.pt',
        'plot_y_label_name': 'BCE Loss',
        'plot_name': 'rank_net.png',
        'model': RankNet(46, 5),
        'loss_fn': nn.BCELoss(),
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg
        }
    },

    'MseNet': {
        'train_folder': '../data/train/MQ2008',
        'experiment_name': 'mse_net',
        'ranker': simple_ranker,
        'lr': 1e-3,
        'batch_size': 64,
        'num_epochs': 100,
        'model_name': 'mse_net.pt',
        'plot_y_label_name': 'MSE Loss',
        'plot_name': 'mse_net.png',
        'model': SimpleNet(46, 5),
        'loss_fn': nn.MSELoss(),
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg
        }
    }
}
