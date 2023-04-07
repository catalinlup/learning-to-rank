from loss_functions import neuralNDCG, neuralNDCG_wrap, approxNDCGLoss
from neural_nets.RankNet import RankNet
from neural_nets.SimpleNet import SimpleNet
from neural_nets.DropoutNet import DropoutNet
from neural_nets.ApproxNdcgNet import ApproxNdcgNet
import torch.nn as nn
from rankers import simple_ranker, pairwise_ranker, approx_ndcg_ranker
from metrics import ndsg, precision_at_k, average_precision

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
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'MseNet': {
        'train_folder': '../data/train/MQ2008',
        'experiment_name': 'mse_net',
        'ranker': simple_ranker,
        'lr': 5e-5,
        'batch_size': 64,
        'num_epochs': 300,
        'model_name': 'mse_net.pt',
        'plot_y_label_name': 'MSE Loss',
        'plot_name': 'mse_net.png',
        'model': SimpleNet(46, 5),
        'loss_fn': nn.MSELoss(),
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'NeuralNDCG_QbQ': {
        'train_folder': '../data/train/GroupedQbQMQ2008',
        'experiment_name': 'neural_ndcg_net_qbq',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 100,
        'model_name': 'neural_ndcg_net_qbq.pt',
        'plot_y_label_name': 'Neural NDCG Loss',
        'plot_name': 'neural_ndcg_net_qbq.png',
        'model': ApproxNdcgNet(46, 5),
        'loss_fn': neuralNDCG,
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'NeuralNDCG_normal': {
        'train_folder': '../data/train/GroupedMQ2008',
        'experiment_name': 'neural_ndcg_net_normal',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'neural_ndcg_net_normal.pt',
        'plot_y_label_name': 'Neural NDCG Loss',
        'plot_name': 'neural_ndcg_net_normal.png',
        'model': ApproxNdcgNet(46, 5),
        'loss_fn': neuralNDCG,
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'NeuralNDCG_batched': {
        'train_folder': '../data/train/GroupedQbQMQ2008',
        'experiment_name': 'neural_ndcg_net_batched',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 100,
        'model_name': 'neural_ndcg_net_batched.pt',
        'plot_y_label_name': 'Neural NDCG Loss',
        'plot_name': 'neural_ndcg_net_batched.png',
        'model': ApproxNdcgNet(46, 5),
        'loss_fn': neuralNDCG,
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'MseNet_batched_10k': {
        'train_folder': '../data/train/GroupedQbQMSLR10K',
        'experiment_name': 'mse_net_batched_10k',
        'ranker': approx_ndcg_ranker,
        'lr': 5e-5,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'mse_net_batched_10k.pt',
        'plot_y_label_name': 'MSE Loss',
        'plot_name': 'mse_net_batched_10k.png',
        'model': ApproxNdcgNet(136, 5),
        'loss_fn': nn.MSELoss(),
        'layer_structure': [136, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'MseNet_QbQ_10k': {
        'train_folder': '../data/train/GroupedQbQMSLR10K',
        'experiment_name': 'mse_net_qbq_10k',
        'ranker': simple_ranker,
        'lr': 1e-3,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'mse_net_qbq_10k.pt',
        'plot_y_label_name': 'MSE Loss',
        'plot_name': 'mse_net_qbq_10k.png',
        'model': ApproxNdcgNet(136, 5),
        'loss_fn': nn.MSELoss(),
        'layer_structure': [136, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'MseNet_QbQ': {
        'train_folder': '../data/train/GroupedQbQMQ2008',
        'experiment_name': 'mse_net_qbq',
        'ranker': simple_ranker,
        'lr': 1e-3,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'mse_net_qbq.pt',
        'plot_y_label_name': 'MSE Loss',
        'plot_name': 'mse_net_qbq.png',
        'model': ApproxNdcgNet(46, 5),
        'loss_fn': nn.MSELoss(),
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'NeuralNDCG_batched_10k': {
        'train_folder': '../data/train/GroupedQbQMSLR10K',
        'experiment_name': 'neural_ndcg_net_batched_10K',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'neural_ndcg_net_batched_10K.pt',
        'plot_y_label_name': 'Neural NDCG Loss',
        'plot_name': 'neural_ndcg_net_batched_10K.png',
        'model': ApproxNdcgNet(136, 5),
        'loss_fn': neuralNDCG,
        'layer_structure': [136, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'ApproxNDCG_batched_10k': {
        'train_folder': '../data/train/GroupedQbQMSLR10K',
        'experiment_name': 'approx_ndcg_net_batched_10K',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'approx_ndcg_net_batched_10K.pt',
        'plot_y_label_name': 'Approx NDCG Loss',
        'plot_name': 'approx_ndcg_net_batched_10K.png',
        'model': ApproxNdcgNet(136, 5),
        'loss_fn': approxNDCGLoss,
        'layer_structure': [136, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'ApproxNDCG': {
        'train_folder': '../data/train/GroupedMQ2008',
        'experiment_name': 'approx_ndcg_net',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'num_epochs': 100,
        'model_name': 'approx_ndcg_net.pt',
        'plot_y_label_name': 'Approx NDCG Loss',
        'plot_name': 'approx_ndcg_net.png',
        'model': ApproxNdcgNet(46, 5),
        'loss_fn': approxNDCGLoss,
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'ApproxNDCG_QbQ': {
        'train_folder': '../data/train/GroupedQbQMQ2008',
        'experiment_name': 'approx_ndcg_net_qbq',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'num_epochs': 100,
        'model_name': 'approx_ndcg_net_qbq.pt',
        'plot_y_label_name': 'Approx NDCG Loss',
        'plot_name': 'approx_ndcg_net_qbq.png',
        'model': ApproxNdcgNet(46, 5),
        'loss_fn': approxNDCGLoss,
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },

    'ApproxNDCG_QbQ_MSLR10K': {
        'train_folder': '../data/train/GroupedQbQMSLR10K',
        'experiment_name': 'approx_ndcg_net_qbq_10k',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'approx_ndcg_net_qbq_10k.pt',
        'plot_y_label_name': 'Approx NDCG Loss',
        'plot_name': 'approx_ndcg_net_qbq_10k.png',
        'model': ApproxNdcgNet(136, 5),
        'loss_fn': approxNDCGLoss,
        'layer_structure': [136, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },


    'NeuralNDCG_QbQ_MSLR10K': {
        'train_folder': '../data/train/GroupedQbQMSLR10K',
        'experiment_name': 'neural_ndcg_net_qbq_10k',
        'ranker': approx_ndcg_ranker,
        'lr': 1e-4,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'neural_ndcg_net_qbq_10k.pt',
        'plot_y_label_name': 'Neural NDCG Loss',
        'plot_name': 'neural_ndcg_net_qbq_10k.png',
        'model': ApproxNdcgNet(136, 5),
        'loss_fn': neuralNDCG,
        'layer_structure': [136, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    },


    'NeuralNDCG_reg': {
        'train_folder': '../data/train/MQ2008',
        'experiment_name': 'neural_ndcg_net_reg',
        'ranker': simple_ranker,
        'lr': 0.001,
        'weight_decay': 1e-1,
        'batch_size': 64,
        'num_epochs': 30,
        'model_name': 'neural_ndcg_net_reg.pt',
        'plot_y_label_name': 'Neural NDCG Loss',
        'plot_name': 'neural_ndcg_net_reg.png',
        'model': DropoutNet(46, 5, [0.1, 0.1]),
        'loss_fn': neuralNDCG,
        'layer_structure': [46, 5],
        'metrics': {
            'ndsg': ndsg,
            'ndsg@2': lambda y_pred, y_true: ndsg(y_pred, y_true, k=2),
            'ndsg@4': lambda y_pred, y_true: ndsg(y_pred, y_true, k=4),
            'ndsg@6': lambda y_pred, y_true: ndsg(y_pred, y_true, k=6),
            'precision@2': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 2),
            'precision@4': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 4),
            'precision@6': lambda y_pred, y_true: precision_at_k(y_pred, y_true, 6),
            'average_precision': average_precision

        }
    }
}
