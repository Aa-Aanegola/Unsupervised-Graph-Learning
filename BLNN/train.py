import sys
sys.path.append('../')
sys.path.append('../../')
import os
import copy
import torch
import numpy as np
from bgrl import *
from absl import app
from absl import flags
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_scatter.composite import scatter_softmax
import pickle as pkl
from bgrl.experiments import *
from fairness.metrics import get_average_fairness_metrics
import datetime
import wandb
from common.utils import get_logger, set_random_seeds, node_clustering, similarity_search
from common.constants import device
from common.logistic_regression_eval import fit_logistic_regression
from common.data import get_dataset
from common.transforms import get_graph_drop_transform

FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 5, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'amazon-computers',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'cora', 'citeseer', 'twitch-en', 'twitch-de'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', '../data', 'Where the dataset resides.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes in the dataset.')
flags.DEFINE_string('centrality_path', 'degree_centrality.pkl', 'Path to centrality file')

# Architecture. 
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10000, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')
flags.DEFINE_float('tau', 1., 'Temperature in computing positiveness.')
flags.DEFINE_bool('structure_learning', False, 'Add an auxiliary task to learn centrality')
flags.DEFINE_float('alpha', 0.001, 'Weight ratio for structure learning loss')
flags.DEFINE_integer('centrality_hidden_size', 64, 'Hidden layer size for centrality predictor MLP head')
flags.DEFINE_bool('use_bgrl', True, 'Use BGRL loss function')
flags.DEFINE_bool('use_blnn', False, 'Use BLNN loss function')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')
flags.DEFINE_string('transform_type', 'drop_edge', 'Type of transformation to apply to the graph.')
flags.DEFINE_bool('sample_two_hop', False, 'Sample two hop edges')

# Evaluation
flags.DEFINE_integer('eval_epochs', 250, 'Evaluate every eval_epochs.')

# Logging 
flags.DEFINE_string('comment', '', 'Comment to add to the log file.')

def main(argv):
    os.makedirs('./logs', exist_ok=True)
    logger = get_logger(f'./logs/{FLAGS.dataset}.log')
    params = {
        'lr_warmup_epochs': FLAGS.lr_warmup_epochs,
        'predictor_hidden_size': FLAGS.predictor_hidden_size,
        'lrwd': (FLAGS.lr, FLAGS.weight_decay),
        'tau': FLAGS.tau,
        'graph_encoder_layer': FLAGS.graph_encoder_layer,
        'drop_rate': (FLAGS.drop_edge_p_1, FLAGS.drop_feat_p_1, FLAGS.drop_edge_p_2, FLAGS.drop_feat_p_2 )
    }
    logger.info(f'{datetime.datetime.now()} {FLAGS.comment}')
    logger.info(str(params))

    wandb.init(project='Unsup-GNN', config=FLAGS.flag_values_dict())
    wandb.run.name = datetime.datetime.now().strftime("%Y%m%d") + ' ' + FLAGS.dataset \
        + ' ' + FLAGS.transform_type+'_diff' + (' all' if not FLAGS.sample_two_hop and 'extended' in FLAGS.transform_type else '') \
        + ' ' + FLAGS.centrality_path.split('_')[0] + ' BGRL'
    
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

    # load data
    dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset, FLAGS.centrality_path, FLAGS.sample_two_hop)
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
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)   # 512, 256, 128
    predictor = MLP_Predictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size)
    centrality_predictor = MLP_Predictor(representation_size, 1, hidden_size=FLAGS.centrality_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, FLAGS.epochs)
    
    
    def train(step):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # update momentum
        mm = 1 - mm_scheduler.get(step)

        # forward
        optimizer.zero_grad()

        x1, x2 = transform_1(data), transform_2(data)

        q1, y2 = model(x1, x2)
        q2, y1 = model(x2, x1)
        q1 = F.normalize(q1, p=2, dim=1)
        q2 = F.normalize(q1, p=2, dim=1)
        y1 = F.normalize(y1.detach(), p=2, dim=1)
        y2 = F.normalize(y2.detach(), p=2, dim=1)
        
        loss, loss_self, loss_neig, loss_structure = torch.Tensor([0.]).to(device), torch.Tensor([0.]).to(device), torch.Tensor([0.]).to(device), torch.Tensor([0.]).to(device)
        if FLAGS.use_bgrl:
            loss_self = bgrl_loss(q1, q2, y1, y2, src, dst, FLAGS)
            loss += loss_self
        if FLAGS.use_blnn:
            loss_neig = blnn_loss(q1, q2, y1, y2, src, dst, FLAGS)
            loss += loss_neig
        if FLAGS.structure_learning:
            c = centrality_predictor(q1)
            loss_structure = FLAGS.alpha*F.mse_loss(c, centrality)
            loss += loss_structure

        logger.info(f'Epoch: {epoch}, Loss: {loss.item()} Loss Self: {loss_self.item()} Loss Neig: {loss_neig.item()} Loss Structure: {loss_structure.item()}')
        wandb.log({'loss': loss.item(), 'loss_self': loss_self.item(), 'loss_neig': loss_neig.item(), 'loss_structure': loss_structure.item()}, commit=False)

        loss.backward()

        # update online network
        optimizer.step()
        # update target network
        model.update_target_network(mm)


    def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)

        # node classification
        y_preds, y_tests, groups, avg_report = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                                                    data.group.cpu().numpy(),
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
