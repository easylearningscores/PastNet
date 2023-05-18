import argparse
from exp_vq import PastNet_exp

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='DiscreteSTModel', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--load_model', default="", type=str)

    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj', 'caltech'])
    parser.add_argument('--num_workers', default=8, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int, nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj 
    parser.add_argument('--hid_T', default=256, type=int)
    parser.add_argument('--N_T', default=8, type=int)
    parser.add_argument('--groups', default=4, type=int)
    parser.add_argument('--res_units', default=32, type=int)
    parser.add_argument('--res_layers', default=4, type=int)
    parser.add_argument('--K', default=512, type=int)
    parser.add_argument('--D', default=64, type=int)

    # Training parameters
    parser.add_argument('--epochs', default=201, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--load_pred_train', default=0, type=int, help='Learning rate')
    parser.add_argument('--freeze_vqvae', default=0, type=int)
    parser.add_argument('--theta', default=1, type=float)
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = PastNet_exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)