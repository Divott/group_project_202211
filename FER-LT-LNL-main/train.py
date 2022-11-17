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
num_class, train_set_path, validation_set_path, image_path_prefix, train_number, validation_number, resize, crop, dataset_mean, dataset_std, seed, threshold_ini, batch_size, num_workers = get_variables(
    train_number=5000)

torch.cuda.set_device(0)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    pass


def warm_up(net, optimizer, dataloader):
    pass


def test(net1, net2, test_loader):
    pass


def eval(epoch, model):
    pass


def create_model():
    model = models.resnet50(wrights=None)
    model.load_state_dict(torch.load(
        '/home/tangb_lab/cse30013027/zmj/checkpoint/model_initial.pth'))
    model.fc = nn.Linear(2048, num_class)
    model = model.cuda()
    return model


loader = dataloader.clothing_dataloader(root=args.data_path, batch_size=args.batch_size, num_workers=0,
                                        num_batches=args.num_batches)

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

optimizer1 = optim.SGD(net1.parameters(), lr=args.lr,
                       momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr,
                       momentum=0.9, weight_decay=1e-3)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()

best_acc = [0, 0]
for epoch in range(args.num_epochs + 1):
    lr = args.lr
    if epoch >= 40:
        lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr

    if epoch < 1:  # warm up
        train_loader = loader.run('warmup')
        print('Warmup Net1')
        warmup(net1, optimizer1, train_loader)
        train_loader = loader.run('warmup')
        print('\nWarmup Net2')
        warmup(net2, optimizer2, train_loader)
    else:
        pred1 = (prob1 > args.p_threshold)  # divide dataset
        pred2 = (prob2 > args.p_threshold)

        print('\n\nTrain Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run(
            'train', pred2, prob2, paths=paths2)  # co-divide
        train(epoch, net1, net2, optimizer1, labeled_trainloader,
              unlabeled_trainloader)  # train net1
        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run(
            'train', pred1, prob1, paths=paths1)  # co-divide
        train(epoch, net2, net1, optimizer2, labeled_trainloader,
              unlabeled_trainloader)  # train net2

    print('\n==== net 1 evaluate next epoch training data loss ====')
    # evaluate training data loss for next epoch
    eval_loader = loader.run('eval_train')
    prob1, paths1 = eval_train(epoch, net1)
    print('\n==== net 2 evaluate next epoch training data loss ====')
    eval_loader = loader.run('eval_train')
    prob2, paths2 = eval_train(epoch, net2)

test_loader = loader.run('test')
net1.load_state_dict(torch.load('./checkpoint/%s_net1.pth.tar' % args.id))
net2.load_state_dict(torch.load('./checkpoint/%s_net2.pth.tar' % args.id))
acc = test(net1, net2, test_loader)

log.write('Test Accuracy:%.2f\n' % (acc))
log.flush()
