import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--block_size', type=int, default=128)
    model_arg.add_argument('--n_layer', type=int, default=4)
    model_arg.add_argument('--n_head', type=int, default=4)
    model_arg.add_argument('--n_embd', type=int, default=128)
    model_arg.add_argument('--embd_pdrop', type=float, default=0.1)
    model_arg.add_argument('--resid_pdrop', type=float, default=0.1)
    model_arg.add_argument('--attn_pdrop', type=float, default=0.1)

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--train_epochs', type=int, default=80,
                           help='Number of epochs for model training')
    train_arg.add_argument('--batch_size', type=int, default=64,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    train_arg.add_argument('--grad_norm_clip', type=float, default=1.0,
                           help='Gradient clipping value')
    train_arg.add_argument('--max_iters', type=int, default=None,
                           help='Maximum number of training iterations; None means train by epochs')
    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]
