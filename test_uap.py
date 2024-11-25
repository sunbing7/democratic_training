from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict

from utils.data import get_data_specs, get_data, Normalizer
from utils.utils import *
from utils.network import *
from utils.training import *


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis')
    # dataset used to train UAP
    parser.add_argument('--dataset', default='imagenet',
                        choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Used dataset to generate UAP (default: imagenet)')
    # dataset used to train UAP model
    parser.add_argument('--pretrained_dataset', default='imagenet',
                        choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Used dataset to train the initial model (default: imagenet)')
    # model used to train UAP
    parser.add_argument('--model_name', type=str, default='vgg19.pth',
                        help='model name (default: vgg19.pth)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--uap_model', type=str, default='checkpoint.pth.tar',
                        help='uap model name (default: checkpoint.pth.tar)')
    parser.add_argument('--uap_name', type=str, default='uap.npy',
                        help='uap file name (default: uap.npy)')

    # model to test
    parser.add_argument('--test_dataset', default='imagenet',
                        choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Test model training set (default: cifar10)')
    parser.add_argument('--test_arch', default='vgg19',
                        choices=['googlenet', 'vgg19', 'resnet50', 'shufflenetv2', 'mobilenet', 'wideresnet', 'resnet110'],
                        help='Test model architecture: (default: vgg19)')

    # Parameters regarding UAP
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
    parser.add_argument('--is_nips', action='store_true', default=True, help='Evaluation on NIPS data')
    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)
    return args


def main():
    args = parse_arguments()

    random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.pretrained_seed)
    cudnn.benchmark = True

    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.dataset,
                                network_arch=args.test_arch,
                                random_seed=args.pretrained_seed,
                                result_subfolder=args.result_subfolder,
                                postfix=args.postfix)

    # Init logger
    log_file_name = os.path.join(result_path, 'log.txt')
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('save path : {}'.format(result_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.pretrained_seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    _, data_test = get_data(args.test_dataset, args.test_dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.test_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.test_dataset,
                                network_arch=args.test_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.test_arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    # Set the target model into evaluation mode
    target_network.eval()

    if args.pretrained_dataset == "caltech" or args.pretrained_dataset == 'asl':
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
    elif args.pretrained_dataset == 'eurosat':
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        if 'repaired' in args.model_name:
            adaptive = '_adaptive'
    elif args.pretrained_dataset == "imagenet" and 'repaired' in args.model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    elif args.pretrained_dataset == "cifar10":
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            if args.test_arch == 'resnet110':
                sd0 = torch.load(model_weights_path)['state_dict']
                target_network.load_state_dict(sd0, strict=True)
            else:
                target_network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))

    if args.test_arch == 'resnet110':
        # Normalization wrapper, so that we don't have to normalize adversarial perturbations
        normalize = Normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        target_network = nn.Sequential(normalize, target_network)

    total_params = get_num_parameters(target_network)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    if args.use_cuda:
        target_network = target_network.cuda()

    #load uap
    mask = None
    print('Evaluate uap stamp')
    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.pretrained_dataset,
                            network_arch=args.test_arch,
                            random_seed=args.pretrained_seed)

    if args.targeted:
        target_name = str(args.target_class)
    else:
        target_name = 'nontarget'

    if 'spgd' in args.uap_name:
        uap_fn = os.path.join(uap_path, 'spgd/uap_' + target_name + '.pth')
        tstd = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1))
        tuap = torch.load(uap_fn) / tstd
    elif 'lavan' in args.uap_name:
        uap_fn = os.path.join(uap_path, 'lavan/uap_' + target_name + '.pth')
        tuap = torch.load(uap_fn)
        _, mask = init_patch_square((1, 3, 224, 224), 176, 224, 176, 224)
        if args.use_cuda:
            mask = mask.cuda()
    elif 'sga' in args.uap_name:
        uap_fn = os.path.join(uap_path, 'sga/uap_' + target_name + '.pth')
        tstd = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1))
        tuap = torch.load(uap_fn) / tstd
    elif 'gap' in args.uap_name:
        uap_fn = os.path.join(uap_path, 'gap/uap_' + target_name + '.pth')
        U_loaded = torch.load(uap_fn)
        U_loaded = U_loaded.expand(1, U_loaded.size(1), U_loaded.size(2), U_loaded.size(3))
        tuap = gap_normalize_and_scale(U_loaded, 1)
    else:
        if 'adaptive' in args.uap_name:
            uap_fn = os.path.join(uap_path, 'uap_' + target_name + '_adaptive.npy')
        else:
            uap_fn = os.path.join(uap_path, 'uap_' + target_name + '.npy')
        uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
        tuap = torch.from_numpy(uap)

    if args.use_cuda:
        tuap = tuap.cuda()
    metrics_evaluate_test(data_loader=data_test_loader,
                          target_model=target_network,
                          uap=tuap,
                          targeted=args.targeted,
                          target_class=args.target_class,
                          mask=mask,
                          log=log,
                          use_cuda=args.use_cuda)

    log.close()


if __name__ == '__main__':
    main()
