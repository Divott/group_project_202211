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
import plotter
import os
from tqdm import tqdm
import argparse

use_Aff = False


# define the hyperparameters
parser = argparse.ArgumentParser(description='PyTorch FER Training')

parser.add_argument('--num_class', default=7, type=int)

if use_Aff:
    # this is for AffectNet
    parser.add_argument('--train_set_path',
                        default='/home/tangb_lab/cse30013027/Data/AffectNet/aligned_affectnet_train.csv', type=str)
    parser.add_argument('--validation_set_path',
                        default='/home/tangb_lab/cse30013027/Data/AffectNet/aligned_affectnet_test.csv', type=str)
    parser.add_argument('--image_path_prefix',
                        default='/home/tangb_lab/cse30013027/Data/AffectNet/Manually_Annotated_Images_AffectNet', type=str)
else:
    # this is for RAF-DB
    parser.add_argument('--train_set_path',
                        default='/home/tangb_lab/cse30013027/Data/RAF-DB/list_patition_label.txt', type=str)
    parser.add_argument('--validation_set_path',
                        default='/home/tangb_lab/cse30013027/Data/RAF-DB/list_patition_label.txt', type=str)
    parser.add_argument('--image_path_prefix',
                        default='/home/tangb_lab/cse30013027/Data/RAF-DB/aligned', type=str)

if use_Aff:
    # this is for AffectNet
    parser.add_argument('--train_number', default=282406, type=int)
    parser.add_argument('--neutral_number', default=74495, type=int)
    parser.add_argument('--happy_number', default=133756, type=int)
    parser.add_argument('--sad_number', default=25309, type=int)
    parser.add_argument('--surprise_number', default=14016, type=int)
    parser.add_argument('--fear_number', default=6322, type=int)
    parser.add_argument('--disgust_number', default=3783, type=int)
    parser.add_argument('--anger_number', default=24725, type=int)
    # parser.add_argument('--contempt_number', default=3734, type=int)
    parser.add_argument('--validation_number', default=3498, type=int)

    parser.add_argument(
        '--prior', default=[0.2638, 0.4736, 0.0896, 0.0496, 0.0224, 0.014, 0.0875], type=list)
else:
    # this is for RAF-DB
    parser.add_argument('--train_number', default=12271, type=int)
    parser.add_argument('--surprise_number', default=1290, type=int)
    parser.add_argument('--fear_number', default=281, type=int)
    parser.add_argument('--disgust_number', default=717, type=int)
    parser.add_argument('--happy_number', default=4772, type=int)
    parser.add_argument('--sad_number', default=1982, type=int)
    parser.add_argument('--anger_number', default=705, type=int)
    parser.add_argument('--neutral_number', default=2524, type=int)
    parser.add_argument('--validation_number', default=3068, type=int)

if use_Aff:
    # this is for AffectNet
    parser.add_argument(
        '--prior', default=[0.2638, 0.4736, 0.0896, 0.0496, 0.0224, 0.014, 0.0875], type=list)
else:
    # this is for RAF-DB
    parser.add_argument(
        '--prior', default=[0.1051, 0.0229, 0.0584, 0.3889, 0.1615, 0.0575, 0.2057], type=list)

parser.add_argument('--resize', default=256, type=int)
parser.add_argument('--crop', default=224, type=int)

parser.add_argument(
    '--dataset_mean', default=(0.5863, 0.4595, 0.4030), type=tuple)
parser.add_argument(
    '--dataset_std', default=(0.2715, 0.2424, 0.2366), type=tuple)

parser.add_argument('--seed', default=123, type=int)
parser.add_argument(
    '--threshold_ini', default=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], type=list)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=0, type=int)

parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--num_epochs', default=25, type=int)

if use_Aff:
    # this is for AffectNet
    parser.add_argument(
        '--model_ini_path', default='/home/tangb_lab/cse30013027/zmj/checkpoint/model_initial.pth', type=str)
else:
    # this is for RAF-DB
    parser.add_argument(
        '--model_ini_path', default='/home/tangb_lab/cse30013027/zmj/checkpoint/resnet18_msceleb.pth', type=str)

parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--warm_up', default=5, type=bool)
parser.add_argument('--lambda_u', default=0, type=float)
args = parser.parse_args()

# set cuda and seed
device = torch.device("cuda")
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if use_Aff:
    # this is for AffectNet
    class_count = [
        args.neutral_number,
        args.happy_number,
        args.sad_number,
        args.surprise_number,
        args.fear_number,
        args.disgust_number,
        args.anger_number,
    ]
else:
    # this is for RAF-DB
    class_count = [
        args.surprise_number,
        args.fear_number,
        args.disgust_number,
        args.happy_number,
        args.sad_number,
        args.anger_number,
        args.neutral_number
    ]

# In all of the cases, the data has to be mapped to the device.

# If X and y are the data:

# X.cuda()
# y.cuda()

# Training


def train(epoch, net, net2, optimizer, scaler, labeled_trainloader, unlabeled_trainloader, model_code):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    loss_plot = np.zeros(num_iter)  # for plot

    with torch.cuda.amp.autocast():
        with tqdm(total=len(labeled_trainloader)) as pbar:
            for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
                pbar.update(1)
                try:
                    inputs_u, inputs_u2 = next(unlabeled_train_iter)
                except:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, inputs_u2 = next(unlabeled_train_iter)
                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                labels_x = torch.zeros(batch_size, args.num_class).scatter_(
                    1, labels_x.view(-1, 1), 1)
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)

                inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(
                ), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
                inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    outputs_u11 = net(inputs_u)
                    outputs_u12 = net(inputs_u2)
                    outputs_u21 = net2(inputs_u)
                    outputs_u22 = net2(inputs_u2)

                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                                dim=1) + torch.softmax(
                        outputs_u22, dim=1)) / 4  # co-guessing
                    ptu = pu ** (1 / args.T)  # temparature sharpening

                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()

                    # label refinement of labeled samples
                    outputs_x = net(inputs_x)
                    outputs_x2 = net(inputs_x2)

                    px = (torch.softmax(outputs_x, dim=1) +
                          torch.softmax(outputs_x2, dim=1)) / 2  # co-refinement
                    px = w_x * labels_x + (1 - w_x) * px
                    ptx = px ** (1 / args.T)  # temparature sharpening

                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                    # mixmatch
                l = np.random.beta(args.alpha, args.alpha)
                l = max(l, 1 - l)

                all_inputs = torch.cat(
                    [inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                all_targets = torch.cat(
                    [targets_x, targets_x, targets_u, targets_u], dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a[:batch_size * 2] + \
                    (1 - l) * input_b[:batch_size * 2]
                mixed_target = l * target_a[:batch_size *
                                            2] + (1 - l) * target_b[:batch_size * 2]

                logits = net(mixed_input)
                logits_x = logits[:batch_size*2]
                logits_u = logits[batch_size*2:]

                Lx, Lu, lamb = criterion(
                    logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+batch_idx/num_iter, args.warm_up)

                # regularization
                prior = torch.tensor(args.prior)
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                loss = Lx + lamb * Lu+penalty

                loss_plot[batch_idx] = loss

                # compute gradient and do SGD step
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    plotter.plot_train_loss(loss_plot, epoch, model_code)  # plot


def warm_up(net, optimizer, scaler, dataloader, epoch, model_code):
    net.train()
    loss_plot = np.zeros(args.train_number//(args.batch_size*2) + 1)
    with torch.cuda.amp.autocast():
        with tqdm(total=len(dataloader)) as pbar:
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                pbar.update(1)
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = CEloss(outputs, labels)
                # penalty = conf_penalty(outputs)
                L = loss  # + penalty
                loss_plot[batch_idx] = L
                scaler.scale(L).backward()
                scaler.step(optimizer)
                scaler.update()
    plotter.plot_train_loss(loss_plot, epoch, model_code)


def test(net1, net2, test_loader, epoch, model_code):
    if net1 is not None:
        net1.eval()
    if net2 is not None:
        net2.eval()
    correct = 0
    total = 0
    losses = torch.zeros(args.validation_number)
    n = 0
    confusion_matrix = torch.zeros((args.num_class, args.num_class))
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with tqdm(total=len(test_loader)) as pbar:
                for inputs, targets in test_loader:
                    pbar.update(1)
                    inputs, targets = inputs.cuda(), targets.cuda()
                    if net1 is None:
                        outputs = net2(inputs)
                    elif net2 is None:
                        outputs = net1(inputs)
                    else:
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
    plotter.plot_confusion_matrix(np_arr, epoch, model_code, use_Aff)
    return acc, torch.sum(losses)/args.validation_number


def eval(epoch, model, eval_loader, model_code):
    model.eval()

    loss_flat = []  # the loss for all samples

    # the loss for samples of each class
    class_loss = [[] for _ in range(args.num_class)]
    # the idx for samples of each class
    index = [[] for _ in range(args.num_class)]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with tqdm(total=len(eval_loader)) as pbar:
                for inputs, targets in eval_loader:

                    pbar.update(1)
                    inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    loss = CE(outputs, targets).cpu().numpy()

                    targets = np.rint(targets.cpu().numpy()).astype(int)
                    for i in range(inputs.size(0)):
                        loss_flat.append(float(loss[i].item()))
                        idx = len(loss_flat)-1
                        class_loss[targets[i]].append(loss[i])
                        index[targets[i]].append(idx)

    for i in range(len(class_count)):
        assert class_count[i] == len(index[i])
        assert class_count[i] == len(class_loss[i])

    prob_flat = torch.zeros(args.train_number)  # the probability to return
    loss_flat = np.asarray(loss_flat)
    for i in range(args.num_class):
        loss = np.asarray(class_loss[i])
        idx = np.asarray(index[i])

        # for plot and compare with other classes
        loss_p = (loss - loss_flat.min()) / (loss_flat.max() - loss_flat.min())
        loss_p = loss_p.reshape(-1, 1)
        # for gmm fit
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        loss = loss.reshape(-1, 1)

        prob = np.ones(len(loss))

        # all classes
        gmm = GaussianMixture(n_components=2, max_iter=10,
                              tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss)
        prob = gmm.predict_proba(loss)
        prob = prob[:, gmm.means_.argmin()]

        # if args.prior[i] > 0.1:  # neutral, happy
        #     gmm = GaussianMixture(n_components=2, max_iter=10,
        #                           tol=1e-2, reg_covar=5e-4)
        #     gmm.fit(loss)
        #     prob = gmm.predict_proba(loss)
        #     prob = prob[:, gmm.means_.argmin()]

        loss_plot = np.zeros(101)
        prob_plot = np.zeros(101)
        for j in range(len(prob)):
            loss_plot[(int)(loss_p[j]*100)] += 1
            prob_plot[(int)(prob[j]*100)] += 1
            prob_flat[idx[j]] = prob[j]  # write the prob back

        plotter.plot_loss_num(loss_plot, epoch, i, model_code, use_Aff)
        plotter.plot_prob_num(prob_plot, epoch, i, model_code, use_Aff)

    loss_flat = (loss_flat - loss_flat.min()) / \
        (loss_flat.max() - loss_flat.min())  # plot all samples
    loss_flat = loss_flat.reshape(-1, 1)
    loss_plot = np.zeros(101)
    for i in range(len(loss_flat)):
        loss_plot[(int)(loss_flat[i]*100)] += 1
    plotter.plot_loss_num(
        loss_plot, epoch, args.num_class, model_code, use_Aff)

    return prob_flat.detach().cpu().numpy()


cudnn.benchmark = True


# define models here


def create_model(path=args.model_ini_path):
    if use_Aff:
        # this is for AffectNet
        model = models.resnet50()
    else:
        # this is for RAF-DB
        model = models.resnet18()

    model.load_state_dict(torch.load(path))

    if use_Aff:
        # this is for AffectNet
        model.fc = nn.Linear(2048, args.num_class)
    else:
        # this is for RAF-DB
        model.fc = nn.Linear(512, args.num_class)

    # model = nn.DataParallel(model)
    model = model.to(device)
    model.cuda()
    return model


print('| Building net')
net1 = create_model()

net2 = create_model()

if args.resume:
    net1.load_state_dict(torch.load(
        '/home/tangb_lab/cse30013027/zmj/checkpoint/models/model_1(epoch 0).pth'))
    net2.load_state_dict(torch.load(
        '/home/tangb_lab/cse30013027/zmj/checkpoint/models/model_2(epoch 0).pth'))


# define losses here
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x,
                                                 dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        pr = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(pr.log() * pr, dim=1))


CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
conf_penalty = NegEntropy()
criterion = SemiLoss()


# define optimizers here


optimizer1 = optim.SGD(net1.parameters(), lr=args.lr,
                       momentum=0.9, weight_decay=1e-3)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr,
                       momentum=0.9, weight_decay=1e-3)

scaler1 = torch.cuda.amp.GradScaler()
scaler2 = torch.cuda.amp.GradScaler()


# some other modules(e.g. ada_cm) todo here


loader = dataloader.dataloader(num_class=args.num_class, train_set_path=args.train_set_path, validation_set_path=args.validation_set_path,
                               image_path_prefix=args.image_path_prefix,
                               train_number=args.train_number, validation_number=args.validation_number, resize=args.resize, crop=args.crop,
                               dataset_mean=args.dataset_mean, dataset_std=args.dataset_std, batch_size=args.batch_size, num_workers=args.num_workers)


# the train module


loss_paint = np.zeros(args.num_epochs+1)
acc_paint = np.zeros((args.num_epochs+1, 3))
if args.resume:
    loss_paint = torch.load('test_loss.pth')
    acc_paint = torch.load('test_acc.pth')

for epoch in range(0, args.num_epochs + 1):
    print('start epoch' + str(epoch))
    if not os.path.exists("/home/tangb_lab/cse30013027/zmj/checkpoint/images/%s" % str(epoch)):
        os.mkdir("/home/tangb_lab/cse30013027/zmj/checkpoint/images/%s" %
                 str(epoch))

    if epoch >= 40:
        args.lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = args.lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = args.lr

    if epoch > 0:
        if epoch < 5:  # warm up
            train_loader = loader.run('warm_up', use_Aff)
            print('Warmup Net1')
            warm_up(net1, optimizer1, scaler1, train_loader, epoch, 1)
            train_loader = loader.run('warm_up', use_Aff)
            print('\nWarmup Net2')
            warm_up(net2, optimizer2, scaler2, train_loader, epoch, 2)
        else:
            # evaluate training data loss for next epoch
            print('\n==== net 1 evaluate next epoch training data loss ====')
            eval_loader = loader.run('eval', use_Aff)
            prob1 = eval(epoch, net1, eval_loader, 1)
            print('\n==== net 2 evaluate next epoch training data loss ====')
            eval_loader = loader.run('eval', use_Aff)
            prob2 = eval(epoch, net2, eval_loader, 2)

            print('\n\nTrain Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run(
                'train', use_Aff, prob2, args.threshold_ini)  # co-divide
            train(epoch, net1, net2, optimizer1, scaler1, labeled_trainloader,
                  unlabeled_trainloader, 1)  # train net1
            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run(
                'train', use_Aff, prob1, args.threshold_ini)  # co-divide
            train(epoch, net2, net1, optimizer2, scaler2, labeled_trainloader,
                  unlabeled_trainloader, 2)  # train net2

    test_loader = loader.run('test', use_Aff)
    acc, loss = test(net1, net2, test_loader, epoch, 0)
    acc_paint[epoch, 0] = acc
    test_loader = loader.run('test', use_Aff)
    acc, loss = test(net1, None, test_loader, epoch, 1)
    acc_paint[epoch, 1] = acc
    test_loader = loader.run('test', use_Aff)
    acc, loss = test(None, net2, test_loader, epoch, 2)
    acc_paint[epoch, 2] = acc

    loss_paint[epoch] = loss

    torch.save(net1.state_dict(),
               '/home/tangb_lab/cse30013027/zmj/checkpoint/models/model_1(epoch %s).pth' % str(epoch))
    torch.save(net2.state_dict(),
               '/home/tangb_lab/cse30013027/zmj/checkpoint/models/model_2(epoch %s).pth' % str(epoch))
    torch.save(loss_paint, 'test_loss.pth')
    torch.save(acc_paint, 'test_acc.pth')
plotter.plot_test_loss(loss_paint)
plotter.plot_test_acc(acc_paint)
