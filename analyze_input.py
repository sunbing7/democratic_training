from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data, Normalizer
from utils.utils import get_model_path, get_uap_path
from utils.network import get_network
from utils.network import get_num_parameters, get_num_trainable_parameters
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg
from collections import OrderedDict
from utils.training import *

import warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis on Input')
    parser.add_argument('--option', default='analyze_entropy',
                        choices=['repair_ae', 'repair'],
                        help='Run options')
    parser.add_argument('--causal_type', default='act', choices=['act'],
                        help='Causality analysis type (default: act)')
    parser.add_argument('--dataset', default='imagenet',
                        choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Used dataset to generate UAP (default: imagenet)')
    parser.add_argument('--arch', default='resnet50',
                        choices=['googlenet', 'vgg19', 'resnet50', 'shufflenetv2', 'mobilenet', 'wideresnet', 'resnet110'])
    parser.add_argument('--model_name', type=str, default='vgg19.pth',
                        help='model name')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--split_layer', type=int, default=43,
                        help='causality analysis layer (default: 43)')
    parser.add_argument('--split_layers', type=int, nargs="*", default=[43])
    # Parameters regarding UAP
    parser.add_argument('--num_iterations', type=int, default=32,
                        help='Number of iterations (default: 32)')
    parser.add_argument('--num_batches', type=int, default=1500)
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')
    parser.add_argument('--targeted',  type=bool, default='',
                        help='Target a specific class (default: False)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    parser.add_argument('--analyze_clean', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--ae_iter', type=int, default=10)

    parser.add_argument('--is_nips', default=1, type=int,
                        help='Evaluation on NIPS data')

    parser.add_argument('--loss_function', default='ce', choices=['ce', 'neg_ce', 'logit', 'bounded_logit',
                                                                  'bounded_logit_fixed_ref', 'bounded_logit_neg'],
                        help='Used loss function for source classes: (default: bounded_logit_fixed_ref)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--epsilon', type=float, default=0.03922,
                        help='Norm restriction of UAP (default: 10/255)')
    parser.add_argument('--uap_name', type=str, default='uap.npy',
                        help='uap file name (default: uap.npy)')
    parser.add_argument('--en_weight', default=0.5, type=float,
                        help='control the weight of entropy in loss function')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
    print('use_cuda: {}'.format(args.use_cuda))
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    return args


def uap_repair(args):
    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    data_train, _ = get_data(args.dataset, args.dataset)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    uap = torch.from_numpy(uap)

    ####################################
    # Init model, criterion, and optimizer
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.arch,
                                 input_size=input_size,
                                 num_classes=num_classes,
                                 finetune=False)

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in target_network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            target_network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        if 'repaired' in args.model_name:
            adaptive = '_adaptive'
    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    elif args.dataset == "cifar10":
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
            adaptive = '_adaptive'
        else:
            if args.arch == 'resnet110':
                sd0 = torch.load(model_weights_path)['state_dict']
                target_network.load_state_dict(sd0, strict=True)
            else:
                target_network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    if args.arch == 'resnet110':
        target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
        # Normalization wrapper, so that we don't have to normalize adversarial perturbations
        normalize = Normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        target_network = nn.Sequential(normalize, target_network)

    #non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print("Target Network Trainable parameters: {}".format(trainale_params))
    print("Target Network Total # parameters: {}".format(total_params))

    target_network.train()

    if args.loss_function == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == "neg_ce":
        criterion = NegativeCrossEntropy()
    elif args.loss_function == "logit":
        criterion = LogitLoss(num_classes=num_classes, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit":
        criterion = BoundedLogitLoss(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_fixed_ref":
        criterion = BoundedLogitLossFixedRef(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_neg":
        criterion = BoundedLogitLoss_neg(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    else:
        raise ValueError

    print('Criteria: {}'.format(criterion))

    if args.use_cuda:
        target_network.cuda()
        criterion.cuda()

    optimizer = torch.optim.SGD(target_network.parameters(), lr=args.learning_rate, momentum=0.9)
    #'''
    # Measure the time needed for the UAP generation
    start = time.time()

    if 'ae' in args.option:
        repaired_network = adv_train(data_train_loader,
                                     target_network,
                                     args.arch,
                                     criterion,
                                     optimizer,
                                     args.num_iterations,
                                     args.split_layers,
                                     uap=uap,
                                     num_batches=args.num_batches,
                                     alpha=args.alpha,
                                     use_cuda=args.use_cuda,
                                     adv_itr=args.ae_iter,
                                     eps=args.epsilon,
                                     mean=mean,
                                     std=std)
        post_fix = 'ae'
    else:
        #fine tune with clean sample only
        repaired_network = train_repair(data_loader=data_train_loader,
                                        model=target_network,
                                        arch=args.arch,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        num_iterations=args.num_iterations,
                                        num_batches=args.num_batches,
                                        print_freq=args.print_freq,
                                        use_cuda=args.use_cuda)
        post_fix = 'finetuned'

    end = time.time()
    print("Time needed for UAP repair: {}".format(end - start))

    #eval
    if args.use_cuda:
        uap = uap.cuda()
    metrics_evaluate_test(data_loader=data_test_loader,
                          target_model=repaired_network,
                          uap=uap,
                          targeted=args.targeted,
                          target_class=args.target_class,
                          log=None,
                          use_cuda=args.use_cuda)

    model_repaired_path = os.path.join(model_path, args.arch + '_' + args.dataset + '_' + post_fix + '_repaired.pth')

    torch.save(repaired_network, model_repaired_path)
    print('repaired model saved to {}'.format(model_repaired_path))


if __name__ == '__main__':
    args = parse_arguments()
    state = {k: v for k, v in args._get_kwargs()}
    start = time.time()
    if 'repair' in args.option:
        uap_repair(args)
    end = time.time()
    #print('Process time: {}'.format(end - start))

