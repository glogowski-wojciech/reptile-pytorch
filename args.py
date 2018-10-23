import argparse
from bunch import Bunch


def reproduce_1shot_5way_transductive_omniglot(args):
    # python -u run_omniglot.py
    # --shots 1
    # --inner-batch 10
    # --inner-iters 5
    # --meta-step 1
    # --meta-batch 5
    # --meta-iters 100000
    # --eval-batch 5
    # --eval-iters 50
    # --learning-rate 0.001
    # --meta-step-final 0
    # --train-shots 10
    # --checkpoint ckpt_o15t
    # --transductive
    args.shots = 1
    args.train_batch = 10
    args.train_iters = 5
    args.meta_lr = 1.0
    args.meta_batch = 5
    args.meta_iters = 100000
    args.test_batch = 5
    args.test_iters = 50
    args.lr = 0.001
    args.train_shots = 10
    args.transductive = True
    return args


def reproduce_5shot_5way_omniglot(args):
    # python -u run_omniglot.py
    # --train-shots 10
    # --inner-batch 10
    # --inner-iters 5
    # --learning-rate 0.001
    # --meta-step 1.0
    # --meta-step-final 0
    # --meta-batch 5
    # --meta-iters 100000
    # --eval-batch 5
    # --eval-iters 50
    # --checkpoint ckpt_o55
    args.train_shots = 10
    args.train_batch = 10
    args.train_iters = 5
    args.lr = 0.001
    args.meta_lr = 1.0
    args.meta_batch = 5
    args.meta_iters = 100000
    args.test_batch = 5
    args.test_iters = 50
    return args


def reproduce_1shot_5way_omniglot(args):
    # python -u run_omniglot.py
    # --shots 1
    # --inner-batch 10
    # --inner-iters 10
    # --meta-step 1
    # --meta-batch 5
    # --meta-iters 100000
    # --eval-batch 5
    # --eval-iters 50
    # --learning-rate 0.001
    # --meta-step-final 0
    # --train-shots 10
    # --checkpoint ckpt_o15
    args.shots = 1
    args.train_batch = 10
    args.train_iters = 10
    args.meta_lr = 1.0
    args.meta_batch = 5
    args.meta_iters = 100000
    args.test_batch = 5
    args.test_iters = 50
    args.lr = 0.001
    args.train_shots = 10
    return args


def get_default_args():
    args = Bunch()
    args.classes = 5
    args.shots = 5
    args.train_shots = 0
    args.start_meta_iteration = 0
    args.meta_iters = 400000
    args.train_iters = 20
    args.test_iters = 50
    args.meta_batch = 1
    args.train_batch = 5
    args.test_batch = 5
    args.meta_lr = 0.1
    args.lr = 1e-3
    args.transductive = False
    args.num_samples = 10000
    args.validate_every = 100
    args.input = '../../omniglot'
    args.cuda = 1
    args.config = ''
    args.debug = False
    return args


def parse_args():
    # Parsing
    parser = argparse.ArgumentParser('Train reptile on omniglot')

    # Mode
    parser.add_argument('logdir', help='Folder to store everything/load')

    # - Training params
    parser.add_argument('--classes', default=5, type=int, help='classes in base-task (N-way)')
    parser.add_argument('--shots', default=5, type=int, help='shots per class (K-shot)')
    parser.add_argument('--train-shots', default=0, type=int, help='train shots')
    parser.add_argument('--start-meta-iteration', default=0, type=int, help='start iteration')
    parser.add_argument('--meta-iters', default=400000, type=int, help='number of meta iterations')
    parser.add_argument('--train-iters', default=20, type=int, help='number of base iterations')
    parser.add_argument('--test-iters', default=50, type=int, help='number of base iterations')
    parser.add_argument('--meta-batch', default=1, type=int, help='batch size in meta training')
    parser.add_argument('--train-batch', default=5, type=int, help='minibatch size in base task in training')
    parser.add_argument('--test-batch', default=5, type=int, help='minibatch size in base task in test')
    parser.add_argument('--meta-lr', default=0.1, type=float, help='meta learning rate')
    parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate')
    parser.add_argument('--transductive', help='test all samples at once', action='store_true')
    parser.add_argument('--num-samples', default=10000, type=int, help='number of samples in final evaluation')

    # - General params
    parser.add_argument('--validation', default=0.1, type=float, help='Percentage of validation')
    parser.add_argument('--validate-every', default=100, type=int, help='Meta-evaluation every ... base-tasks')
    parser.add_argument('--input', default='omniglot', help='Path to omniglot dataset')
    parser.add_argument('--cuda', default=1, type=int, help='Use cuda')
    parser.add_argument('--check-every', default=1000, help='Checkpoint every')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint. This works only if starting fresh (i.e., no checkpoints in logdir)')
    parser.add_argument('--config', default='', type=str, help='Set one from default configs (o15, o15t, o55)')
    parser.add_argument('--debug', help='short run for debug', action='store_true')

    # Do some processing
    args = parser.parse_args()
    return args


def preprocess_args(ctx):
    args = get_default_args()
    for param in ctx.params:
        args[param] = ctx.params[param]
    args.train_shots = args.train_shots or args.shots
    if 'o15t' in args.config:
        reproduce_1shot_5way_transductive_omniglot(args)
    elif 'o15' in args.config:
        reproduce_1shot_5way_omniglot(args)
    elif 'o55' in args.config:
        reproduce_5shot_5way_omniglot(args)
    if args.debug:
        args.meta_iters = 1000
        args.validate_every = 10
        args.num_samples = 10
    return args