import sys
sys.path.append('../')
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
from experiments import *
from fairness.metrics import get_average_fairness_metrics
import datetime
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 20, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'amazon-computers',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs', 'cora', 'citeseer', 'twitch', 'ppi'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes in the dataset.')

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
flags.DEFINE_bool('use_blnn', True, 'Use BLNN loss function')

# Augmentations.
flags.DEFINE_float('drop_edge_p_1', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p_1', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('drop_edge_p_2', 0., 'Probability of edge dropout 2.')
flags.DEFINE_float('drop_feat_p_2', 0., 'Probability of node feature dropout 2.')
flags.DEFINE_bool('weighted_edge_drop', False, 'Use the weighted edge drop')

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
    wandb.run.name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ' ' + FLAGS.dataset + ' ' + FLAGS.comment

    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using {} for training.'.format(device))

    # set random seed
    if FLAGS.model_seed is not None:
        print('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)

    # load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]  # all dataset include one graph
    data.num_classes = FLAGS.num_classes
    print('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpu memory
    src, dst = data.edge_index  # node-neighbor pairs


    if FLAGS.structure_learning:
        with open(f'{FLAGS.dataset_dir}/{FLAGS.dataset}/betweenness_centrality.pkl', 'rb') as f:
            centrality = pkl.load(f)

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

        wandb.log({'loss': loss.item(), 'loss_self': loss_self.item(), 'loss_neig': loss_neig.item(), 'loss_structure': loss_structure.item()})

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
        if FLAGS.dataset != 'wiki-cs':
            scores, y_preds, y_tests, groups, avg_report = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                                                       data.group.cpu().numpy(),
                                             data_random_seed=FLAGS.data_seed, repeat=FLAGS.num_eval_splits)
        else:
            scores, y_pred_all, [] = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(), train_masks, val_masks, test_masks)

        # TODO: add imparity stuff
        fairness_metrics = get_average_fairness_metrics(y_tests, y_preds, groups, labels.cpu().numpy())

        logger.info(
                    "Epoch: {:04d} | Accuracy: {:.2f}+-{:.2f}".format(
                        epoch, np.mean(scores)*100, np.std(scores)*100
                    )
        )
        logger.info(f'Performance Metrics {avg_report}')
        logger.info(f'Fairness {fairness_metrics}')
        wandb.log(avg_report)
        wandb.log({'accuracy': np.mean(scores)*100, 'accuracy_std': np.std(scores)*100})
        class_accuracy = fairness_metrics['class_accuracy'].tolist()
        table = wandb.Table(data=class_accuracy, columns=[f"Class {i}" for i in range(len(class_accuracy[0]))])
        wandb.log(fairness_metrics)
        wandb.log({'class_accuracy': table})
        

        # node clustering
        clusterings = node_clustering(representations, labels)
        logger.info(clusterings)
        wandb.log(clusterings)

        # node similarity search
        similarities = similarity_search(representations, labels)
        logger.info(similarities)
        wandb.log(similarities)

    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch-1)
        if epoch % FLAGS.eval_epochs == 0:
            eval(epoch)

if __name__ == "__main__":
    print('PyTorch version: %s' % torch.__version__)
    app.run(main)
