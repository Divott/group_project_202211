import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import random
import numpy as np
import dataloader as dataloader
from sklearn.mixture import GaussianMixture
from config import get_variables
import multiprocessing as mp
import plotter
import os
from tqdm import tqdm


data = get_variables()
num_class = data['num_class']
train_set_path = data['train_set_path']
validation_set_path = data['validation_set_path']
image_path_prefix = data['image_path_prefix']
train_number = data['train_number']
validation_number = data['validation_number']
resize = data['resize']
crop = data['crop']
dataset_mean = data['dataset_mean']
dataset_std = data['dataset_std']
seed = data['seed']
threshold_ini = data['threshold']
batch_size = data['batch_size']
num_workers = data['num_workers']
lr = data['lr']
num_epochs = data['num_epochs']
model_ini_path = data['model_ini_path']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# In all of the cases, the data has to be mapped to the device.

# If X and y are the data:

# X.to(device)
# y.to(device)

# Training


def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, model_code):
    pass


def warm_up(net, optimizer, scaler, dataloader, epoch, model_code):
    net.train()
    loss_plot = np.zeros(train_number//(batch_size*2) + 1)
    with torch.cuda.amp.autocast():
        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                pbar.update(len(inputs))
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = CEloss(outputs, labels)
                penalty = conf_penalty(outputs)
                L = loss + penalty
                loss_plot[batch_idx] = L
                scaler.scale(L).backward()
                scaler.step(optimizer)
                scaler.update()
    plotter.plot_train_loss(loss_plot, epoch, model_code)


def test(net1, net2, test_loader, epoch):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    losses = torch.zeros(validation_number)
    n = 0
    confusion_matrix = torch.zeros((num_class, num_class))
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with tqdm(total=len(test_loader)) as pbar:
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    pbar.update(len(inputs))
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs1 = net1(inputs)
                    outputs2 = net2(inputs)
                    outputs = (outputs1 + outputs2) / 2
                    loss = CE(outputs, targets)
                    for b in range(inputs.size(0)):
                        losses[n] = loss[b]
                        n += 1
                    _, predicted = torch.max(outputs, 1)
                    for t, p in zip(targets.view(-1), predicted.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                    total += targets.size(0)
                    correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Acc: %.2f%%\n" % (acc))
    np_arr = confusion_matrix.detach().cpu().numpy().astype(int)
    plotter.plot_confusion_matrix(np_arr, epoch)
    return acc, torch.sum(losses)/validation_number


def eval(epoch, model, eval_loader, model_code):
    model.eval()
    losses = torch.zeros(train_number)
    n = 0
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with tqdm(total=len(eval_loader)) as pbar:
                for batch_idx, (inputs, targets) in enumerate(eval_loader):
                    pbar.update(len(inputs))
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = CE(outputs, targets)
                    for b in range(inputs.size(0)):
                        losses[n] = loss[b]
                        n += 1
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    losses = losses.reshape(-1, 1)
    loss_plot = np.zeros(101)
    for i in range(len(losses)):
        loss_plot[(int)(losses[i]*100)] += 1
    plotter.plot_loss_num(loss_plot, epoch, 8, model_code)
    gmm = GaussianMixture(n_components=2, max_iter=10,
                          reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)
    prob = gmm.predict_proba(losses)
    prob = prob[:, gmm.means_.argmin()]
    prob_plot = np.zeros(101)
    for i in range(len(prob)):
        prob_plot[(int)(prob[i]*100)] += 1
    plotter.plot_prob_num(prob_plot, epoch, 8, model_code)
    return prob


cudnn.benchmark = True


# define models here


def create_model(path=model_ini_path):
    model = models.resnet50()
    model.load_state_dict(torch.load(path))
    # resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, num_class)
    model = nn.DataParallel(model)
    model.to(device)
    return model


print('| Building net')
net1 = create_model()
net2 = create_model()


# define losses here


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()


# define optimizers here


optimizer1 = optim.SGD(net1.parameters(), lr=lr,
                       momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=lr,
                       momentum=0.9, weight_decay=1e-3)

scaler1 = torch.cuda.amp.GradScaler()
scaler2 = torch.cuda.amp.GradScaler()


# some other modules(e.g. ada_cm) todo here


loader = dataloader.dataloader(num_class=num_class, train_set_path=train_set_path, validation_set_path=validation_set_path,
                               image_path_prefix=image_path_prefix,
                               train_number=train_number, validation_number=validation_number, resize=resize, crop=crop,
                               dataset_mean=dataset_mean, dataset_std=dataset_std, batch_size=batch_size, num_workers=num_workers)


# the train module


loss_paint = np.zeros(num_epochs+1)
for epoch in range(num_epochs + 1):
    print('start epoch' + str(epoch))
    if not os.path.exists("/home/tangb_lab/cse30013027/zmj/checkpoint/images/%s" % str(epoch)):
        os.mkdir("/home/tangb_lab/cse30013027/zmj/checkpoint/images/%s" %
                 str(epoch))

    if epoch >= 40:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    if epoch < 5:  # warm up
        train_loader = loader.run('warm_up')
        print('Warmup Net1')
        warm_up(net1, optimizer1, scaler1, train_loader, epoch, 1)
        train_loader = loader.run('warm_up')
        print('\nWarmup Net2')
        warm_up(net2, optimizer2, scaler2, train_loader, epoch, 2)
    else:
        # pred1 = (prob1 > threshold_ini)  # divide dataset
        # pred2 = (prob2 > threshold_ini)
        pred1 = (prob1 > 0.8)
        pred2 = (prob2 > 0.8)

        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run(
            'train', pred2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader,
              unlabeled_trainloader, 1)  # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(
            'train', pred1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader,
              unlabeled_trainloader, 2)  # train net2

    # evaluate training data loss for next epoch
    print('\n==== net 1 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval')
    prob1 = eval(epoch, net1, eval_loader, 1)
    print('\n==== net 2 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval')
    prob2 = eval(epoch, net2, eval_loader, 2)

    test_loader = loader.run('test')
    acc, loss = test(net1, net2, test_loader, epoch)
    loss_paint[epoch] = loss

    torch.save(net1.state_dict(),
               '/home/tangb_lab/cse30013027/zmj/checkpoint/model_1(epoch %s).pth' % str(epoch))
    torch.save(net2.state_dict(),
               '/home/tangb_lab/cse30013027/zmj/checkpoint/model_2(epoch %s).pth' % str(epoch))
plotter.plot_test_loss(loss_paint)
