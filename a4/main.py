import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
spacy.load('en')

torch.manual_seed(0)
import random
random.seed(0)


import argparse
import os


from models import *


def load_model(lr, vector, mod, embed_dim = None, hidden_dim = None, n_filters = None, filter_sizes = None):

    loss_fnc = torch.nn.BCELoss()
    if mod == 'baseline':
        model = Baseline(embed_dim, vector)
    elif mod == 'rnn':
        model = RNN(embed_dim, vector, hidden_dim)
    elif mod == 'cnn':
        model = CNN(embed_dim,vector, n_filters=n_filters, filter_sizes=filter_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    return model, loss_fnc, optimizer

def evaluate(model, val_loader, loss_fnc):
    total_corr = 0

    ######

    # 3.6 YOUR CODE HERE
    accum_loss = 0
    for i, vbatch in enumerate(val_loader):
        feats = vbatch.text[0].transpose(0, 1)
        label = vbatch.label
        prediction = model(feats, vbatch.text[1])
        batch_loss = loss_fnc(input=prediction, target=label.float())
        accum_loss += batch_loss
        #print(prediction.shape, label.shape)
        corr = (prediction>0.5).squeeze().float() == label.float()
        total_corr += int(corr.sum())
    ######
    #print(len(val_loader.dataset), float(total_corr))
    return float(total_corr)/len(val_loader.dataset), accum_loss

def main(args):
    ######

    # 3.2 Processing of the data

    ######
    text = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    label = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(path='./data/', train='train.tsv',validation='validation.tsv', test='test.tsv', format='tsv',fields=[('', None), ('text', text), ('label', label)], skip_header=True)
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(datasets=(train, val, test), batch_sizes = (args.batch_size,args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False, shuffle = True)
    '''train_iterator, val_iterator, test_iterator = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.text),sort_within_batch = False,
        batch_sizes=(args.batch_size, args.batch_size, args.batch_size), device=-1)'''

    text.build_vocab(train)
    vocab = text.vocab
    vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim='100'))

    model, loss_fnc, optimizer = load_model(args.lr, text.vocab, mod=args.model, embed_dim=args.emb_dim, hidden_dim=args.rnn_hidden_dim, n_filters= args.num_filt, filter_sizes=[2,4])
    ######

    # 5 Training and Evaluation
    t = 0
    max_val = 0
    max_epoch = 0
    for epoch in range(args.epochs):
        accum_loss = 0
        tot_corr = 0
        num_feats = 0

        for i,batch in enumerate(train_iterator):
            feats = batch.text[0].transpose(0,1)
            label = batch.label
            optimizer.zero_grad()

            predictions = model(feats,batch.text[1])
            #print('The predictions are',predictions.shape)
            #print('The labels are',label.shape)
            batch_loss = loss_fnc(input=predictions, target=label.float())

            accum_loss += batch_loss

            batch_loss.backward()
            optimizer.step()

            num_feats += feats.shape[0]

            corr = (predictions>0.5).squeeze().float() == label.float()
            tot_corr += int(corr.sum())

            '''if ((t + 1) % 5 == 0):
                valid_acc = evaluate(model, val_iterator)
                training_acc = float(tot_corr / num_feats)

                accum_loss = 0
                tot_corr = 0
                num_feats = 0

                print("Epoch: {}, Step: {}, Training Accuracy: {}, Validation Accuracy: {}".format(epoch + 1, t + 1,
                                                                                     training_acc,
                                                                                                   valid_acc))
            t = t + 1'''
        valid_acc, valid_loss = evaluate(model, val_iterator, loss_fnc)
        training_acc = float(tot_corr / num_feats)
        print("Epoch: {} | Training Accuracy {} | Validation Accuracy {} | Training Loss {} | Validation Loss {}".format(epoch+1,training_acc, valid_acc,accum_loss,valid_loss))
        if (valid_acc > max_val):
            max_val = valid_acc
            max_epoch = epoch + 1
    print("Max Validation: {} max_epoch {}".format(max_val, max_epoch))
    train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(datasets=(train, val, test), batch_sizes = (args.batch_size, args.batch_size, args.batch_size), sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False, shuffle = True)
    '''train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.text),
        batch_sizes=(len(train), len(val), len(test)), device=-1)'''
    train_acc, train_loss = evaluate(model, train_iterator, loss_fnc)
    valid_acc, valid_loss = evaluate(model, val_iterator, loss_fnc)
    test_acc, test_loss = evaluate(model,test_iterator, loss_fnc)
    print('Train Accuracy: {}| Train Loss {}'.format(train_acc,train_loss))
    print('Validation Accuracy: {}| Valid Loss {}'.format(valid_acc,valid_loss))
    print('Test Accuracy: {}| Test Loss {}'.format(test_acc,test_loss))
    torch.save(model,'model_rnn_noppsequence_wBucketIterator.pt')
    #92.125 so far for RNN bs 64 lr 0.01 epochs 25
    ######


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)