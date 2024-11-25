from __future__ import division
import numpy as np
import os, shutil, time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils.utils import time_string, print_log
from torch.distributions import Categorical
from torch.autograd import Variable
from torchsummary import summary
from matplotlib import pyplot as plt


DEBUG = True


def train(data_loader,
          model,
          criterion,
          optimizer,
          epsilon,
          num_iterations,
          targeted,
          target_class,
          log,
          print_freq=200,
          use_cuda=True):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()

    end = time.time()

    data_iterator = iter(data_loader)

    iteration=0
    while iteration < num_iterations:
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)

        if targeted:
            target = torch.ones(input.shape[0], dtype=torch.int64) * target_class
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        if model.module._get_name() == "Inception3":
            output, aux_output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            output = model(input)
            loss = criterion(output, target)

        if not targeted:
            loss = -loss

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projection
        model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        iteration, num_iterations, batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

        iteration+=1
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)


def train_advanced(data_loader,
                   model,
                   arch,
                   criterion,
                   optimizer,
                   epsilon,
                   num_iterations,
                   split_layers,
                   targeted,
                   target_class,
                   log,
                   print_freq=200,
                   use_cuda=True,
                   en_weight=0.5,
                   adjust=5.0):
    # train function (forward, backward, update)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.module.generator.train()
    model.module.target_model.eval()

    end = time.time()

    data_iterator = iter(data_loader)

    en_loss = Variable(torch.tensor(.0), requires_grad=True)
    en_cri = hloss(use_cuda)

    iteration=0
    while iteration < num_iterations:
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)

        pmodel1, pmodel2 = split_perturbed_model(model, arch, split_layer=split_layers[0])

        if targeted:
            target = torch.ones(input.shape[0], dtype=torch.int64) * target_class
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            target = target.cuda()
            input = input.cuda()

        # compute output
        if model.module._get_name() == "Inception3":
            output, aux_output = model(input)
            loss1 = criterion(output, target)
            loss2 = criterion(aux_output, target)
            loss = loss1 + 0.4*loss2
        else:
            #output_ori = model(input)
            poutput = pmodel1(input)
            output = pmodel2(poutput)
            en_loss = torch.mean(en_cri(poutput.view(len(input), -1)))
            lcce = criterion(output, target)
            #print('[DEBUG] lcce: {}, en_loss: {}'.format(lcce, en_loss))
            loss = (1.0 - en_weight) * lcce - en_weight * adjust * en_loss

        if not targeted:
            loss = -loss

        # measure accuracy and record loss
        if len(target.shape) > 1:
            target = torch.argmax(target, dim=-1)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projection
        model.module.generator.uap.data = torch.clamp(model.module.generator.uap.data, -epsilon, epsilon)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iteration % print_freq == 0:
            print_log('  Iteration: [{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        iteration, num_iterations, batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

        iteration+=1
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                    top5=top5,
                                                                                                    error1=100-top1.avg), log)


def train_repair(data_loader,
                 model,
                 arch,
                 criterion,
                 optimizer,
                 num_iterations,
                 num_batches=1000,
                 print_freq=200,
                 use_cuda=True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()
                loss = criterion(output, target)

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/1563]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                    num_batch, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
              'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
               iteration, num_iterations, batch_time=batch_time,
               data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model

def adv_train(data_loader,
              model,
              arch,
              criterion,
              optimizer,
              num_iterations,
              split_layers,
              uap=None,
              num_batches=1000,
              alpha=0.1,
              use_cuda=True,
              adv_itr=10,
              eps=0.0392,
              mean=[0, 0, 0],
              std=[1, 1, 1]):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if DEBUG:
        adv_top1 = AverageMeter()
        adv_top5 = AverageMeter()
        delta_top1 = AverageMeter()
        delta_top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break
            p_models = []
            for split_layer in split_layers:
                pmodel, _ = split_model(model, arch, split_layer=split_layer)
                p_models = p_models + [pmodel]

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()

            # generate AEs
            delta = ae_training_individual(p_models,
                                           input,
                                           adv_itr,
                                           eps,
                                           True,
                                           mean,
                                           std,
                                           use_cuda)

            x_adv = input + delta

            uap_x = (input + uap).float()

             # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                if DEBUG:
                    delta_output = model(x_adv)
                    adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            losses.update(loss.item(), input.size(0))

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if DEBUG:
                adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
                adv_top1.update(adv_prec1.item(), input.size(0))
                adv_top5.update(adv_prec5.item(), input.size(0))

                delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
                delta_top1.update(delta_prec1.item(), input.size(0))
                delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            for pmodel in p_models:
                del pmodel

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/{}]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch, len(data_loader),
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def pgd_train(data_loader,
              model,
              target_class,
              criterion,
              optimizer,
              num_iterations,
              uap=None,
              num_batches=1000,
              alpha=0.1,
              use_cuda=True,
              adv_itr=10,
              eps=0.0392,
              mean=[0, 0, 0],
              std=[1, 1, 1]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()
    delta_top1 = AverageMeter()
    delta_top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()
            #targeted
            tgt_class = torch.ones_like(target) * target_class

            delta = ae_training_pgd_tgt(model,
                                        input,
                                        tgt_class,
                                        criterion,
                                        adv_itr,
                                        eps,
                                        True,
                                        mean,
                                        std,
                                        use_cuda)
            #'''
            x_adv = input + delta

            uap_x = (input + uap).float()

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                delta_output = model(x_adv)
                adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
            delta_top1.update(delta_prec1.item(), input.size(0))
            delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/{}]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch, len(data_loader),
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def pgd_train_untgt(data_loader,
                    model,
                    criterion,
                    optimizer,
                    num_iterations,
                    uap=None,
                    num_batches=1000,
                    alpha=0.1,
                    use_cuda=True,
                    adv_itr=10,
                    eps=0.0392,
                    mean=[0, 0, 0],
                    std=[1, 1, 1]):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()
    delta_top1 = AverageMeter()
    delta_top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            if num_batch > num_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                target = target.cuda()
                input = input.cuda()
                uap = uap.cuda()

            #untarget
            delta = ae_training_pgd(model,
                                    input,
                                    target,
                                    criterion,
                                    adv_itr,
                                    eps,
                                    True,
                                    mean,
                                    std,
                                    use_cuda)

            x_adv = input + delta

            uap_x = (input + uap).float()

            # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                delta_output = model(x_adv)
                adv_output = model(uap_x)

                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                poutput = model(x_adv)
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(adv_output.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            delta_prec1, delta_prec5 = accuracy(delta_output.data, target_, topk=(1, 5))
            delta_top1.update(delta_prec1.item(), input.size(0))
            delta_top5.update(delta_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/{}]   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      'deltaPrec@1 {delta_top1.val:.3f} ({delta_top1.avg:.3f})   '
                      'deltaPrec@5 {delta_top5.val:.3f} ({delta_top5.avg:.3f})   '
                      .format(
                    num_batch, len(data_loader),
                    loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5,
                    delta_top1=delta_top1, delta_top5=delta_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '.format(
            iteration, num_iterations,
            loss=losses, top1=top1, top5=top5, adv_top1=adv_top1, adv_top5=adv_top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def ae_training_individual(pmodels,
                           x,
                           attack_iters=10,
                           eps=0.0392,
                           rs=True,
                           mean=[0, 0, 0],
                           std=[1, 1, 1],
                           use_cuda=True):
    delta = torch.zeros_like(x)

    # preprocess mean and std
    if type(mean) is not torch.Tensor:
        mean = torch.from_numpy(np.array([0, 0, 0]).reshape(1, 3, 1, 1)).float()
        std = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1)).float()

    if use_cuda:
        delta = delta.cuda()
        mean = mean.cuda()
        std = std.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    en_loss = Variable(torch.tensor(.0), requires_grad=True)
    en_cri = hloss(use_cuda)
    for i in range(attack_iters):
        delta.data = standardize_delta(clamp(delta.data, -eps, eps, use_cuda), mean, std)
        ae_x = (x + delta)
        delta.data = destandardize_delta(delta.data, mean, std)

        for pmodel in pmodels:
            poutput = pmodel(ae_x).view(len(x), -1)
            en_loss = torch.mean(en_cri(poutput))

        loss = -en_loss
        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta + (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps, use_cuda)
        delta.grad.zero_()

    return standardize_delta(delta.detach(), mean, std)


def ae_training_pgd(model,
                    x,
                    y,
                    criterion,
                    attack_iters=10,
                    eps=0.0392,
                    rs=True,
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    use_cuda=True):
    '''
    adversarial training

    '''
    delta = torch.zeros_like(x)

    # preprocess mean and std
    if type(mean) is not torch.Tensor:
        mean = torch.from_numpy(np.array([0, 0, 0]).reshape(1, 3, 1, 1)).float()
        std = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1)).float()

    if use_cuda:
        delta = delta.cuda()
        mean = mean.cuda()
        std = std.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    for i in range(attack_iters):
        delta.data = standardize_delta(clamp(delta.data, -eps, eps, use_cuda), mean, std)
        ae_x = (x + delta)
        delta.data = destandardize_delta(delta.data, mean, std)

        output = model(ae_x)
        loss = criterion(output, y)

        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta + (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps, use_cuda)
        delta.grad.zero_()

    return standardize_delta(delta.detach(), mean, std)


def ae_training_pgd_tgt(model,
                        x,
                        target_class,
                        criterion,
                        attack_iters=10,
                        eps=0.0392,
                        rs=True,
                        mean=[0, 0, 0],
                        std=[1, 1, 1],
                        use_cuda=True):
    '''
    adversarial training

    '''
    delta = torch.zeros_like(x)

    # preprocess mean and std
    if type(mean) is not torch.Tensor:
        mean = torch.from_numpy(np.array([0, 0, 0]).reshape(1, 3, 1, 1)).float()
        std = torch.from_numpy(np.array(std).reshape(1, 3, 1, 1)).float()


    if use_cuda:
        delta = delta.cuda()
        mean = mean.cuda()
        std = std.cuda()
        target_class = target_class.cuda()

    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True

    for i in range(attack_iters):
        delta.data = standardize_delta(clamp(delta.data, -eps, eps, use_cuda), mean, std)
        ae_x = (x + delta)
        delta.data = destandardize_delta(delta.data, mean, std)

        output = model(ae_x)
        loss = criterion(output, target_class)
        #print('Iteration {} loss {}'.format(i, loss))

        loss.backward()
        grad = delta.grad.detach()

        grad_sign = sign(grad)
        delta.data = (delta - (eps / 4) * grad_sign)
        delta.data = clamp(delta.data, -eps, eps, use_cuda)
        delta.grad.zero_()

    return standardize_delta(delta.detach(), mean, std)


def known_uap_train(data_loader,
                    model,
                    arch,
                    criterion,
                    optimizer,
                    num_iterations,
                    split_layers,
                    uaps,
                    alpha=0.1,
                    use_cuda=True):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    adv_top1 = AverageMeter()
    adv_top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    print('[DEBUG] dataloader length: {}'.format(len(data_loader)))

    iteration = 0
    while (iteration < num_iterations):
        num_batch = 0
        for input, target in data_loader:
            # measure data loading time
            data_time.update(time.time() - end)

            # select a uap
            indices = torch.randperm(len(uaps))[:len(input)]
            uap = uaps[indices]
            pert_input = input
            for i, uap_i in enumerate(uap):
                if torch.randint(low=0, high=2, size=(1,1)):
                    pert_input[i] = input[i] + uap_i

            if use_cuda:
                input = input.cuda()
                target = target.cuda()
                pert_input = pert_input.cuda()

             # compute output
            if model._get_name() == "Inception3":
                output, aux_output = model(input)
                loss1 = criterion(output, target)
                loss2 = criterion(aux_output, target)
                loss = loss1 + 0.4 * loss2
            else:
                output = model(input)
                if output.shape != target.shape:
                    target = nn.functional.one_hot(target, len(output[0])).float()

                # cce loss only
                poutput = model((pert_input).float())
                pce_loss = criterion(poutput, target)

                ce_loss = criterion(output, target)
                loss = (1 - alpha) * ce_loss + alpha * pce_loss

            # measure accuracy and record loss
            if len(target.shape) > 1:
                target_ = torch.argmax(target, dim=-1)
            if use_cuda:
                target_ = target_.cuda()

            prec1, prec5 = accuracy(output.data, target_, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            adv_prec1, adv_prec5 = accuracy(poutput.data, target_, topk=(1, 5))
            adv_top1.update(adv_prec1.item(), input.size(0))
            adv_top5.update(adv_prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if num_batch % 100 == 0:
                print('  Batch: [{:03d}/1563]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
                      'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
                      'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
                      .format(
                    num_batch, batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5,
                    adv_top1=adv_top1, adv_top5=adv_top5) + time_string())
            num_batch = num_batch + 1

        print('  Iteration: [{:03d}/{:03d}]   '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
              'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
              'Loss {loss.val:.4f} ({loss.avg:.4f})   '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '
              'advPrec@1 {adv_top1.val:.3f} ({adv_top1.avg:.3f})   '
              'advPrec@5 {adv_top5.val:.3f} ({adv_top5.avg:.3f})   '
              .format(
               iteration, num_iterations, batch_time=batch_time,
               adv_top1=adv_top1, adv_top5=adv_top5, data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string())

        iteration += 1
    print('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1,
                                                                                                top5=top5,
                                                                                                error1=100 - top1.avg))
    return model


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def standardize_delta(delta, mean, std):
    return (delta - mean) / std


def destandardize_delta(delta, mean, std):
    return (delta * std + mean)


def metrics_evaluate(data_loader, target_model, perturbed_model, targeted, target_class, log=None, use_cuda=True):
    # switch to evaluate mode
    target_model.eval()
    perturbed_model.eval()
    perturbed_model.module.generator.eval()
    perturbed_model.module.target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            pert_output = perturbed_model(input)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(pert_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        pert_out_class = torch.argmax(pert_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == pert_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == pert_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg != 0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(pert_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), pert_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = pert_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    if log:
        print_log('\n\t#######################', log)
        print_log('\tClean model accuracy: {:.3f}'.format(clean_acc.avg), log)
        print_log('\tPerturbed model accuracy: {:.3f}'.format(perturbed_acc.avg), log)
        print_log('\tAbsolute Accuracy Drop: {:.3f}'.format(aad_source), log)
        print_log('\tRelative Accuracy Drop: {:.3f}'.format(rad_source), log)
        print_log('\tAttack Success Rate: {:.3f}'.format(100-attack_success_rate.avg), log)
        print_log('\tFooling Ratio: {:.3f}'.format(fooling_ratio), log)
        if targeted:
            print_log('\tAll --> Target Class {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate.avg), log)
            print_log('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate_filtered.avg), log)



def metrics_evaluate_test(data_loader, target_model, uap, targeted, target_class, mask=None, log=None, use_cuda=True):
    # switch to evaluate mode
    target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            if mask is None:
                adv_x = (input + uap).float()
            else:
                adv_x = torch.mul((1 - mask), input) + torch.mul(mask, uap).float()
            attack_output = target_model(adv_x)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(attack_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        uap_out_class = torch.argmax(attack_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == uap_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == uap_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = target_model(adv_x[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg != 0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(attack_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), attack_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = attack_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    print('\n\t#######################')
    print('\tClean model accuracy: {:.3f}'.format(clean_acc.avg))
    print('\tPerturbed model accuracy: {:.3f}'.format(perturbed_acc.avg))
    print('\tAbsolute Accuracy Drop: {:.3f}'.format(aad_source))
    print('\tRelative Accuracy Drop: {:.3f}'.format(rad_source))
    print('\tAttack Success Rate: {:.3f}'.format(100-attack_success_rate.avg))
    print('\tFooling Ratio: {:.3f}'.format(fooling_ratio))
    if targeted:
        print('\tAll --> Target Class {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate.avg))
        print('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate_filtered.avg))



def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def split_model(ori_model, model_name, split_layer=43, flat=False):
    '''
    split given model from the dense layer before logits
    Args:
        ori_model:
        model_name: model name
    Returns:
        splitted models
    '''
    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer < 9:
            modules = list(ori_model.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:9]
            module3 = modules[9:]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 9:
            modules = list(ori_model.children())
            module1 = modules[:9]
            module2 = modules[9]

            model_1st = nn.Sequential(*module1, Flatten())
            model_2nd = nn.Sequential(*[module2])

        else:
            return None, None
    elif model_name == 'googlenet':
        if split_layer < 17:
            modules = list(ori_model.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:17]
            module3 = modules[17:]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 17:
            modules = list(ori_model.children())
            module1 = modules[:17]
            module2 = modules[17:]

            model_1st = nn.Sequential(*module1, Flatten())
            model_2nd = nn.Sequential(*module2)

        elif split_layer == 19:  # googlenet caffe
            modules = list(ori_model.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:22]
            module3 = modules[22:]

            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 22: # googlenet caffe
            modules = list(ori_model.children())
            module1 = modules[:22]
            module2 = modules[22:]

            model_1st = nn.Sequential(*module1, Flatten())
            model_2nd = nn.Sequential(*module2)
        else:
            return None, None
    elif model_name == 'vgg19':
        if flat:
            layers = list(ori_model.children())
            module1 = layers[:split_layer]
            module2 = layers[split_layer:]
            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*module2)
        else:
            if split_layer < 38:
                modules = list(ori_model.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:split_layer]
                module2 = layers[split_layer:38]
                module3 = layers[38:]
                model_1st = nn.Sequential(*module1)
                model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])
            else:
                modules = list(ori_model.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:38]
                moduel2 = layers[38:split_layer]
                module3 = layers[split_layer:]
                model_1st = nn.Sequential(*[*module1, Flatten(), *moduel2])
                model_2nd = nn.Sequential(*module3)
    elif model_name == 'alexnet':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[0]
            module2 = [modules[1]]
            module_ = list(modules[2])
            module3 = module_[:5]
            module4 = module_[5:]

            model_1st = nn.Sequential(*[*module1, Flatten(), *module2, *module3])
            model_2nd = nn.Sequential(*module4)
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(ori_model.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = modules[1:6]
            module2 = [sub_modules[0]]
            module3 = [sub_modules[1]]

            model_1st = nn.Sequential(*[*module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2])
            model_2nd = nn.Sequential(*module3)
        elif split_layer == 1:
            modules = list(ori_model.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = modules[2:6]
            module3 = sub_modules

            model_1st = nn.Sequential(*[*module0, *module1])
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'mobilenet':
        if split_layer == 3:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = [modules[3]]

            model_1st = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(ori_model.children())
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = [modules[2]]
            module3 = [modules[3]]
            model_1st = nn.Sequential(*module0)
            model_2nd = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'wideresnet':
        if split_layer == 6:
            modules = list(ori_model.children())
            module1 = modules[:2]
            module2 = modules[3:7]
            module3 = [modules[-1]]

            model_1st = nn.Sequential(*[*module1, *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(ori_model.children())
            module1 = modules[:2]

            module2 = modules[3:7]
            module3 = [modules[-1]]
            model_1st = nn.Sequential(*module1)
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    else:
        return None, None

    #summary(ori_model, (3, 224, 224))
    #summary(model_1st, (3, 224, 224))

    return model_1st, model_2nd


def split_perturbed_model(ori_model, model_name, split_layer=43, flat=False):
    modules = list(ori_model.module.children())
    generator = modules[0]
    target_network = modules[1]

    if model_name == 'resnet18' or model_name == 'resnet50':
        if split_layer < 9:
            modules = list(target_network.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:9]
            module3 = modules[9:]

            model_1st = nn.Sequential(*[*[generator], *module1])
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 9:
            modules = list(target_network.children())
            module1 = modules[:9]
            module2 = modules[9]

            model_1st = nn.Sequential(*[generator], *module1, Flatten())
            model_2nd = nn.Sequential(*[module2])

        else:
            return None, None
    elif model_name == 'googlenet':
        if split_layer < 17:
            modules = list(target_network.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:17]
            module3 = modules[17:]

            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 17:
            modules = list(target_network.children())
            module1 = modules[:17]
            module2 = modules[17:]

            model_1st = nn.Sequential(*[generator], *module1, Flatten())
            model_2nd = nn.Sequential(*module2)

        elif split_layer == 19:  # googlenet caffe
            modules = list(target_network.children())
            module1 = modules[:split_layer]
            module2 = modules[split_layer:22]
            module3 = modules[22:]

            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])

        elif split_layer == 22: # googlenet caffe
            modules = list(target_network.children())
            module1 = modules[:22]
            module2 = modules[22:]

            model_1st = nn.Sequential(*[generator], *module1, Flatten())
            model_2nd = nn.Sequential(*module2)
        else:
            return None, None
    elif model_name == 'vgg19':
        if flat:
            layers = list(target_network.children())
            module1 = layers[:split_layer]
            module2 = layers[split_layer:]
            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*module2)
        else:
            if split_layer < 38:
                modules = list(target_network.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:split_layer]
                module2 = layers[split_layer:38]
                module3 = layers[38:]
                model_1st = nn.Sequential(*[generator], *module1)
                model_2nd = nn.Sequential(*[*module2, Flatten(), *module3])
            else:
                modules = list(target_network.children())
                layers = list(modules[0]) + [modules[1]] + list(modules[2])
                module1 = layers[:38]
                moduel2 = layers[38:split_layer]
                module3 = layers[split_layer:]
                model_1st = nn.Sequential(*[*[generator], *module1, Flatten(), *moduel2])
                model_2nd = nn.Sequential(*module3)
    elif model_name == 'alexnet':
        if split_layer == 6:
            modules = list(target_network.children())
            module1 = modules[0]
            module2 = [modules[1]]
            module_ = list(modules[2])
            module3 = module_[:5]
            module4 = module_[5:]

            model_1st = nn.Sequential(*[*[generator], *module1, Flatten(), *module2, *module3])
            model_2nd = nn.Sequential(*module4)
    elif model_name == 'shufflenetv2':
        if split_layer == 6:
            modules = list(target_network.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = modules[1:6]
            module2 = [sub_modules[0]]
            module3 = [sub_modules[1]]

            model_1st = nn.Sequential(*[*[generator], *module0, *module1, Avgpool2d_n(poolsize=7), Flatten(), *module2])
            model_2nd = nn.Sequential(*module3)
        elif split_layer == 1:
            modules = list(target_network.children())
            sub_modules = list(modules[-1])
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = modules[2:6]
            module3 = sub_modules

            model_1st = nn.Sequential(*[*[generator], *module0, *module1])
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'mobilenet':
        if split_layer == 3:
            modules = list(target_network.children())
            module1 = modules[:2]
            module2 = [modules[2]]
            module3 = [modules[3]]

            model_1st = nn.Sequential(*[*[generator], *module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(target_network.children())
            module0 = [modules[0]]
            module1 = [modules[1]]
            module2 = [modules[2]]
            module3 = [modules[3]]
            model_1st = nn.Sequential(*[generator], *module0)
            model_2nd = nn.Sequential(*[*module1, Relu(), *module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    elif model_name == 'wideresnet':
        if split_layer == 6:
            modules = list(target_network.children())
            module1 = modules[:2]
            module2 = modules[3:7]
            module3 = [modules[-1]]

            model_1st = nn.Sequential(*[*[generator], *module1, *module2, Avgpool2d_n(poolsize=7), Flatten()])
            model_2nd = nn.Sequential(*module3)
        if split_layer == 1:
            modules = list(target_network.children())
            module1 = modules[:2]

            module2 = modules[3:7]
            module3 = [modules[-1]]
            model_1st = nn.Sequential(*[generator], *module1)
            model_2nd = nn.Sequential(*[*module2, Avgpool2d_n(poolsize=7), Flatten(), *module3])
    else:
        return None, None

    #summary(target_network, (3, 224, 224))
    #summary(model_1st, (3, 224, 224))

    return model_1st, model_2nd


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1, p=2)[:,None,None]


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx >= 0 and idx < self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    return self.max_accuracy(False) == val_acc

  def max_accuracy(self, istrain):
    if self.current_epoch <= 0: return 0
    if istrain: return self.epoch_accuracy[:self.current_epoch, 0].max()
    else:       return self.epoch_accuracy[:self.current_epoch, 1].max()

  def plot_curve(self, save_path):
    title = 'the accuracy/loss curve of train/val'
    dpi = 80
    width, height = 1200, 800
    legend_fontsize = 10
    scale_distance = 48.8
    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    x_axis = np.array([i for i in range(self.total_epoch)]) # epochs
    y_axis = np.zeros(self.total_epoch)

    plt.xlim(0, self.total_epoch)
    plt.ylim(0, 100)
    interval_y = 5
    interval_x = 5
    plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
    plt.yticks(np.arange(0, 100 + interval_y, interval_y))
    plt.grid()
    plt.title(title, fontsize=20)
    plt.xlabel('the training epoch', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)

    y_axis[:] = self.epoch_accuracy[:, 0]
    plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_accuracy[:, 1]
    plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)


    y_axis[:] = self.epoch_losses[:, 0]
    plt.plot(x_axis, y_axis*50, color='g', linestyle=':', label='train-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    y_axis[:] = self.epoch_losses[:, 1]
    plt.plot(x_axis, y_axis*50, color='y', linestyle=':', label='valid-loss-x50', lw=2)
    plt.legend(loc=4, fontsize=legend_fontsize)

    if save_path is not None:
      fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
      print ('---- save figure {} into {}'.format(title, save_path))
    plt.close(fig)


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        x = F.relu(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Avgpool2d(nn.Module):
    def __init__(self):
        super(Avgpool2d, self).__init__()

    def forward(self, x):
        x = F.avg_pool2d(x, 4)
        return x


class Avgpool2d_n(nn.Module):
    def __init__(self, poolsize=2):
        super(Avgpool2d_n, self).__init__()
        self.poolsize = poolsize
    def forward(self, x):
        x = F.avg_pool2d(x, self.poolsize)
        return x


class MyAvgPool2D(nn.Module):
    '''
    tested model:
    - vgg19
    '''
    def __init__(self, output_size, replace=False):
        super(MyAvgPool2D, self).__init__()
        self.output_size = output_size
        self.kernel_size = 4
        self.stride = 4

    def forward(self, x):
        self.stride = x.size(2) // self.output_size[0]
        self.kernel_size = int(x.size(2) - (self.output_size[0] - 1) * self.stride)
        #print('[DEBUG] kernel_size {}'.format(self.kernel_size))
        #print('[DEBUG] stride {}'.format(self.stride))
        return self.get_pools(x)

    def get_pools(self, x):
        pooled = []
        for i in torch.arange(start=0, end=x.size(2), step=self.stride):
            for j in torch.arange(start=0, end=x.size(2), step=self.stride):
                #get a single pool
                fmap = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                pooled.append(torch.mean(fmap, dim=2))
        return torch.reshape(torch.stack(pooled, dim=2),
                             (pooled[0].size(0), pooled[0].size(1), self.output_size[0], self.output_size[1]))


class MyMaxPool2D(nn.Module):
    '''
    tested model:
    - vgg19
    '''
    def __init__(self, kernel_size, stride, replace=False):
        super(MyMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = (7, 7)
        self.replace = replace

    def forward(self, x):
        self.output_size = (x.size(2) // self.stride, x.size(2) // self.stride)
        #print('[DEBUG] output_size {}'.format(self.output_size))
        if self.replace:
            return self.get_entropy_pools(self.get_pools(x))
        else:
            return self.get_pools(x)

    def get_pools(self, x):
        pooled = []
        for i in torch.arange(start=0, end=x.size(2), step=self.stride):
            for j in torch.arange(start=0, end=x.size(2), step=self.stride):
                #get a single pool
                fmap = x[:, :, i:i+self.kernel_size, j:j+self.kernel_size]
                pooled.append(torch.max(torch.max(fmap,dim=2).values,dim=2,keepdim=True).values)
        return torch.reshape(torch.stack(pooled, dim=2),
                             (pooled[0].size(0), pooled[0].size(1), self.output_size[0], self.output_size[1]))

    def get_entropy_pools(self, pooled):
        global_pool_entropy = F.softmax(pooled, dim=1) * F.log_softmax(pooled, dim=1)
        global_pool_entropy = -1.0 * global_pool_entropy.sum(dim=1)
        max_entropy = torch.unsqueeze(torch.max(
                                   torch.max(global_pool_entropy, dim=1).values, dim=1, keepdim=True).values, dim=2)
        #mask = (global_pool_entropy > max_entropy * 0.8)
        global_entropy_weight = (1 - torch.div(global_pool_entropy, max_entropy))
        weighted_pooled = pooled * torch.unsqueeze(global_entropy_weight, dim=1)
        #weighted_pooled = pooled * torch.unsqueeze(mask, dim=1)
        return weighted_pooled


class Mask(nn.Module):
    def __init__(self, mask):
        super(Mask, self).__init__()
        self.mask = mask.to(torch.float)

    def forward(self, x):
        x = x * self.mask
        return x


class hloss(nn.Module):
    def __init__(self, cuda=True):
        super(hloss, self).__init__()
        self.cuda = cuda

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=1)
        return b