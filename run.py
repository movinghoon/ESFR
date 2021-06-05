import os;os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset import EmbeddingFSLLoader
from reconstruction import ReconstructionModule, ESFR
from algorithm import NN, BDCSPN, CSPN
from utils import generate_exp_string


def set_seed(seed):
    np.random.seed(seed)
    random.seed(np.random.randint(2 ** 30))
    tf.random.set_seed(np.random.randint(2 ** 30))


def run(architecture='WidResNet',
        dataset='mini',
        shot=1,
        num_tasks=2000,
        shift=True,
        num_ensemble=5,
        drop_rate=0.5,
        lam=0.,
        way=5,
        seed=1,
        name='ESFR'):
    # EXP name
    exp_string = generate_exp_string(locals())

    # Set seed
    set_seed(seed)

    # Loader
    dset = EmbeddingFSLLoader(architecture=architecture,
                              dset=dataset,
                              data_type='test',
                              way=way,
                              shot=shot,
                              meta_batch_size=1,
                              query_samples_per_class=15)

    # Reconstruction Module
    net = ReconstructionModule(input_dim=dset.feat_dim, hidden=tuple([dset.feat_dim, ] * 4))
    model = ESFR(net=net,
                 drop_rate=drop_rate,
                 num_ensemble=num_ensemble,
                 centering=True,
                 lam=lam,
                 weight_decay=0.,
                 k=21,
                 period=2,
                 max_updates=100,
                 way=way)

    # Classifier
    # alg = NN(centering=False, l2_normalize=False)
    alg = BDCSPN(centering=False, l2_normalize=False, supp_logit_by_label=True, shift=True)

    def test(feats, lbs):
        return [alg(feats, lbs).numpy()]

    # Pre-processing
    def _pre_process(feats):
        feat_dim = feats.shape[-1]
        if shift:
            feats = tf.reshape(feats, (way, -1, feat_dim))
            supp_feats, qry_feats = feats[:, :-15], feats[:, -15:]
            qry_feats = qry_feats - tf.reduce_mean(tf.reduce_mean(qry_feats, axis=0), axis=0) + \
                        tf.reduce_mean(tf.reduce_mean(supp_feats, axis=0), axis=0)
            feats = tf.concat([supp_feats, qry_feats], axis=1)
        feats = tf.reshape(feats, (-1, feat_dim))
        feats = feats - tf.reduce_mean(feats, axis=0)
        return tf.math.l2_normalize(feats, axis=-1)

    # Test
    accs = []
    labels = dset.label
    for i in tqdm(range(num_tasks)):
        features, labels = dset.sample()
        supp_lbs = tf.reshape(tf.squeeze(labels)[:, :-15], (-1, 5))

        # Early-Stage Feature Reconstruction
        feats = _pre_process(features)
        feats = model.get_feats(feats, supp_lbs)

        # Test
        accs.append(test(feats, lbs=labels))
    mean = np.mean(accs, axis=0) * 100
    std = np.std(accs, axis=0) * 196 / np.sqrt(num_tasks)
    return mean, std, exp_string


def generate_acc_string(mean, std):
    text = ''
    for i in range(len(mean)):
        text += '{:5.4g}+-{:4.3g}'.format(mean[i], std[i])
        if i < len(mean) - 1:
            text += '| '
    return text


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    # few-shot classification
    parser.add_argument('--dataset', nargs='*', default=['mini'], help='dataset, [mini, tiered, cub]')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--num_tasks', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1)

    # ESFR related
    parser.add_argument('--architecture', nargs='*', default=['WidResNet'], help='backbone, [WidResNet, ResNet_18]')
    parser.add_argument('--num_ensemble', type=int, default=5)
    parser.add_argument('--shift', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True)
    parser.add_argument('--drop_rate', nargs='*', default=[0.5])
    parser.add_argument('--lam', type=float, default=0.)

    # Logging
    parser.add_argument('--log', type=str, default='./log.txt')
    args = parser.parse_args()

    drop_rate = float(args.drop_rate[0]) if len(args.drop_rate) == 1 else args.drop_rate
    for dataset in args.dataset:
        for architecture in args.architecture:
            # run
            mean, std, exp_string = run(architecture=architecture,
                                        dataset=dataset,
                                        shot=args.shot,
                                        num_tasks=args.num_tasks,
                                        shift=args.shift,
                                        num_ensemble=args.num_ensemble,
                                        drop_rate=drop_rate,
                                        lam=args.lam,
                                        seed=args.seed)
            acc_string = generate_acc_string(mean, std)

            # Print
            print(exp_string)
            print('Accuracy:' + acc_string)
            print('')

            # Log
            print('Logfile:', args.log)
            if args.log is not None:
                with open(args.log, 'a') as f:
                    print(exp_string, file=f)
                    print('Accuracy:' + acc_string, file=f)
                    print('', file=f)