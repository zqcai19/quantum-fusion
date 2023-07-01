import random
import pickle
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from datasets import IEMOCAPDataLoaders
from FusionModel import QNet, translator
from Arguments import Arguments


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    print("Test accuracy: {}".format(metrics['acc']))
    print("Test f1 score: {}".format(metrics['f1']))


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for data_a, data_v, data_t, target in data_loader:
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()
        output = model(data_a, data_v, data_t)
        loss = criterion(output, torch.max(target, 1)[1])
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_a, data_v, data_t, target in data_loader:
            data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
            target = target.to(args.device)
            output = model(data_a, data_v, data_t)
            instant_loss = criterion(output, torch.max(target, 1)[1]).item()
            total_loss += instant_loss
    total_loss /= len(data_loader.dataset)
    return total_loss


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_a, data_v, data_t, target = next(iter(data_loader))
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        output = model(data_a, data_v, data_t)
    output = output.cpu().numpy()
    target = target.numpy()
    true_label = np.argmax(target, axis=1)
    pred_label = np.argmax(output, axis=1)
    metrics['acc'] = accuracy_score(true_label, pred_label).item()
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()
    return metrics


def Scheme(design):
    args = Arguments()
    if torch.cuda.is_available() and args.device == 'cuda':
        print("using cuda device")
    else:
        print("using cpu device")
    train_loader, val_loader, test_loader = IEMOCAPDataLoaders(args)
    model = QNet(args, design).to(args.device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam([
        {'params': model.ClassicalLayer_a.parameters()},
        {'params': model.ClassicalLayer_v.parameters()},
        {'params': model.ClassicalLayer_t.parameters()},
        {'params': model.ProjLayer_t.parameters()},
        {'params': model.QuantumLayer.parameters(), 'lr': args.qlr},
        {'params': model.Classifier.parameters()}
        ], lr=args.clr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 10000
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, args)
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(epoch, train_loss, val_loss, 'saving model')
            best_model = copy.deepcopy(model)
        else:
            print(epoch, train_loss, val_loss)
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'metrics': metrics}
    return best_model, report


if __name__ == '__main__':
    with open('search_space', 'rb') as file:
        search_space = pickle.load(file)
    net = random.choice(search_space)
    design = translator(net)
    best_model, report = Scheme(design)
