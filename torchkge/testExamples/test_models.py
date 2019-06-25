#!/usr/bin/env python
# coding: utf-8

import torch.cuda as cuda
import sys
import pickle

from time import time
from torchkge.testExamples.utils import get_model, get_optimizer, get_criterion, get_dataset
from torchkge.testExamples.utils import save_current, get_path, load_parameters

from torch.utils.data import DataLoader

from torchkge.data.utils import corrupt_batch

if __name__ == "__main__":
    ################################################################################################
    # Parsing Arguments
    ################################################################################################
    args = sys.argv[1:]
    params = load_parameters(args)
    path = get_path(params['model'], params['dataset'])

    ################################################################################################
    # Presentation of script
    ################################################################################################
    print('\n')
    print('_____________________')
    print('Test of {} model'.format(params['model']))
    print('_____________________')
    print('lr : {}, nb_epochs : {}, batch size : {}'.format(params['lr'],
                                                            params['nb_epochs'],
                                                            params['b_size']))

    ################################################################################################
    # Data pre processing
    ################################################################################################
    print('______________________')
    print('Loading Data')
    kg_train, kg_test = get_dataset(params)

    ################################################################################################
    # Model definition
    ################################################################################################
    model = get_model(params, kg_train.n_ent, kg_test.n_rel)
    optimizer = get_optimizer(model, params)
    criterion = get_criterion(params)

    dicts = {}, {}, {}, {}, {}, {}, {}, {}

    ################################################################################################
    # CUDA
    ################################################################################################
    if params['cuda'] and cuda.is_available():
        cuda.set_device(params['device'])
        print('Use cuda on device {} ({})'.format(params['device'],
                                                  cuda.get_device_name(params['device'])))
        model.cuda()
        criterion.cuda()
        print('Variables moved to CUDA')
    else:
        print("Don't use CUDA")

    cuda.empty_cache()

    ################################################################################################
    # Begin of training
    ################################################################################################
    dataloader = DataLoader(kg_train, batch_size=params['b_size'], shuffle=False,
                            pin_memory=params['cuda'])
    output_file = open(path + 'file.txt', 'w')
    print('___________________')
    print('Start of training :')
    epoch = 0
    for epoch in range(params['nb_epochs']):

        epoch_time = time()
        first_loss, current_loss = 0, 0
        for i, batch in enumerate(dataloader):
            # get the input
            heads, tails, rels = batch[0], batch[1], batch[2]
            if heads.is_pinned():
                heads, tails, rels = heads.cuda(), tails.cuda(), rels.cuda()

            # Create Negative Samples
            neg_heads, neg_tails = corrupt_batch(heads, tails, n_ent=kg_train.n_ent)

            # zero model gradient
            model.zero_grad()

            # forward + backward + optimize
            output = model(heads, tails, neg_heads, neg_tails, rels)
            loss = criterion(output)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if i == 0:
                first_loss = current_loss

            output_file.write('[%d, %5d] loss: %.3f \n' % (epoch + 1, i + 1, current_loss))

        print('Epoch {} loss : {} to {} (duration : {}s)'.format(epoch + 1,
                                                                 first_loss, current_loss,
                                                                 time() - epoch_time))

        if (epoch + 1) % params['test_epochs'] == 0:
            print('Saving at epoch : {}'.format(epoch + 1))
            t = time()
            dicts = save_current(model, kg_train, kg_test,
                                 params['b_size_eval'], dicts, epoch, path)
            output_file.write('[%d], Training Hit@10 : %.3f, Training MeanRank : %.3f, '
                              'Testing Hit@10 : %.3f, Testing MeanRank : %.3f' % (
                                  epoch + 1, dicts[0][epoch + 1], dicts[4][epoch + 1],
                                  dicts[2][epoch + 1], dicts[6][epoch + 1]))
            output_file.write('[%d], Filt. Training Hit@10 : %.3f, Filt. Training MeanRank : %.3f, '
                              'Filt. Testing Hit@10 : %.3f, Filt. Testing MeanRank : %.3f' % (
                                  epoch + 1, dicts[1][epoch + 1], dicts[5][epoch + 1],
                                  dicts[3][epoch + 1], dicts[7][epoch + 1]))
            print('Saved in {}s'.format(time() - t))

    dicts = save_current(model, kg_train, kg_test, params['b_size_eval'], dicts, epoch + 1, path)

    ################################################################################################
    # Saving final results
    ################################################################################################
    output_file.close()

    with open(path + 'performance_values.pkl', 'wb') as file_:
        try:
            pickle.dump(dicts, file_)
        except OverflowError:
            pickle.dump(dicts, file_, protocol=4)
