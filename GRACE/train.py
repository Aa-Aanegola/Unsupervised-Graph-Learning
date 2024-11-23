import sys
sys.path.append('../')
sys.path.append('../../')
import os
import copy
import torch
import numpy as np
from absl import app
from absl import flags
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_scatter.composite import scatter_softmax
from torch_geometric.nn import GCNConv
import pickle as pkl
from fairness.metrics import get_average_fairness_metrics
import datetime
import wandb
from common.utils import get_logger, set_random_seeds, node_clustering, similarity_search
from common.constants import device
from common.logistic_regression_eval import fit_logistic_regression
from common.data import get_dataset
from common.transforms import get_graph_drop_transform
from model import Encoder, Model, drop_feature

FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 20, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'amazon-computers',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'cora', 'citeseer', 'twitch-en', 'twitch-de'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', '../data', 'Where the dataset resides.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes in the dataset.')
flags.DEFINE_string('centrality_path', 'degree_centrality.pkl', 'Path to centrality file')

# Architecture. 
flags.DEFINE_integer('num_hidden', 128, 'Number of hidden units in the model.')
flags.DEFINE_integer('num_proj_hidden', 128, 'Number of hidden units in the projection head.')
flags.DEFINE_string('activation', 'relu', 'Activation function to use in the model.')
flags.DEFINE_string('base_model', 'GCNConv', 'Base GNN layer to use in the model.')
flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('tau', 1., 'Temperature in computing positiveness.')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')
flags.DEFINE_string('transform_type', 'drop_edge', 'Type of transformation to apply to the graph.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 100, 'Evaluate every eval_epochs.')

# Logging 
flags.DEFINE_string('comment', '', 'Comment to add to the log file.')

def main(argv):
    os.makedirs('./logs', exist_ok=True)
    logger = get_logger(f'./logs/{FLAGS.dataset}.log')
    params = {
        'num_hidden': FLAGS.num_hidden,
        'lrwd': (FLAGS.lr, FLAGS.weight_decay),
        'tau': FLAGS.tau,
        'base_model': FLAGS.base_model,
        'drop_rate': (FLAGS.drop_edge_p_1, FLAGS.drop_feat_p_1, FLAGS.drop_edge_p_2, FLAGS.drop_feat_p_2 )
    }
    logger.info(f'{datetime.datetime.now()} {FLAGS.comment}')
    logger.info(str(params))

    wandb.init(project='Unsup-GNN', config=FLAGS.flag_values_dict())
    wandb.run.name = datetime.datetime.now().strftime("%Y%m%d") + ' ' + FLAGS.dataset + ' ' + FLAGS.transform_type + ' ' + FLAGS.centrality_path.split('_')[0] + ' GRACE'
    
    # wandb class accuracy table
    columns = ["Epoch"]
    for cls in range(FLAGS.num_classes):
        columns.extend([f"Group 0 Class {cls} Accuracy", f"Group 1 Class {cls} Accuracy", f"Class {cls} Accuracy Difference"])
    class_accuracy_table = wandb.Table(columns=columns)

    # use CUDA_VISIBLE_DEVICES to select gpu
    print('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        print('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)


    dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset, FLAGS.centrality_path)
    num_eval_splits = FLAGS.num_eval_splits

    data = dataset[0]  # all dataset include one graph
    data.num_classes = FLAGS.num_classes
    print('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpu memory
    src, dst = data.edge_index  # node-neighbor pairs

    # prepare transforms
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1, drop_feat_p=FLAGS.drop_feat_p_1, FLAGS=FLAGS)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2, drop_feat_p=FLAGS.drop_feat_p_2, FLAGS=FLAGS)

    # build networks
    encoder = Encoder(data.x.size(1), FLAGS.num_hidden, F.relu, base_model=({'GCNConv': GCNConv})[FLAGS.base_model], k=FLAGS.num_layers).to(device)
    model = Model(encoder, FLAGS.num_hidden, FLAGS.num_proj_hidden, FLAGS.tau).to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    
    def train(step):
        model.train()
        optimizer.zero_grad()

        d1, d2 = transform_1(data), transform_2(data)
        z1, z2 = model(d1.x, d1.edge_index), model(d2.x, d2.edge_index)
        loss = model.loss(z1, z2, batch_size=0)
        
        logger.info(f'Epoch: {epoch}, Loss: {loss.item()}')
        wandb.log({'loss': loss.item()}, commit=False)

        loss.backward()

        # update online network
        optimizer.step()


    def eval(epoch):
        # make temporary copy of encoder
        model.eval()
        representations = model(dataset[0].x, dataset[0].edge_index)
        labels = data.y

        # node classification
        y_preds, y_tests, groups, avg_report = fit_logistic_regression(representations.cpu().detach().numpy(), labels.cpu().detach().numpy(),
                                                                    data.group.cpu().detach().numpy(),
                                            data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)
    
        # TODO: add imparity stuff
        fairness_metrics = get_average_fairness_metrics(y_tests, y_preds, groups, labels.cpu().numpy())

        logger.info(
                    "Epoch: {:04d} | Accuracy: {:.2f}+-{:.2f}".format(
                        epoch, avg_report['accuracy']*100, avg_report['accuracy_std']*100
                    )
        )
        logger.info(f'Performance Metrics {avg_report}')
        logger.info(f'Fairness {fairness_metrics}')
        wandb.log(avg_report, commit=False)
        wandb.log(fairness_metrics, commit=False)

        group_0_accuracies = fairness_metrics['class_accuracy'][0]  
        group_1_accuracies = fairness_metrics['class_accuracy'][1]
        accuracy_diffs = [g1 - g0 for g0, g1 in zip(group_0_accuracies, group_1_accuracies)]
        class_acc_row = [epoch]
        for cls in range(FLAGS.num_classes):
            class_acc_row.extend([group_0_accuracies[cls], group_1_accuracies[cls], accuracy_diffs[cls]])
        class_accuracy_table.add_data(*class_acc_row)

        # node clustering
        clusterings = node_clustering(representations, labels)
        logger.info(clusterings)
        wandb.log(clusterings, commit=False)

        # node similarity search
        similarities = similarity_search(representations, labels)
        logger.info(similarities)
        wandb.log(similarities, commit=False)

    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:
            eval(epoch-1)
        wandb.log({'epoch': epoch-1}, commit=True)
    # log table
    wandb.log({'class_accuracy_table': class_accuracy_table}, step=FLAGS.epochs-1)

if __name__ == "__main__":
    print('PyTorch version: %s' % torch.__version__)
    app.run(main)
    wandb.finish()