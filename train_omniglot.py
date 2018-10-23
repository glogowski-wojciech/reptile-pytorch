import os
from neptune import ChannelType
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from args import parse_args, preprocess_args
from models import OmniglotModel
from omniglot import MetaOmniglotFolder, split_omniglot, ImageCache, transform_image, transform_label
from specs.neptune_utils import get_configuration


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


ctx, exp_dir_path = get_configuration()
os.environ['MRUNNER_UNDER_NEPTUNE'] = '1'
args = preprocess_args(ctx)
print('args:')
print(args)


def Variable_(tensor, *args_, **kwargs):
    '''
    Make variable cuda depending on the arguments
    '''
    # Unroll list or tuple
    if type(tensor) in (list, tuple):
        return [Variable_(t, *args_, **kwargs) for t in tensor]
    # Unroll dictionary
    if isinstance(tensor, dict):
        return {key: Variable_(v, *args_, **kwargs) for key, v in tensor.items()}
    # Normal tensor
    variable = Variable(tensor, *args_, **kwargs)
    if args.cuda:
        variable = variable.cuda()
    return variable


train_meta_loss_channel = ctx.create_channel('train loss', channel_type=ChannelType.NUMERIC)
train_meta_accuracy_channel = ctx.create_channel('train meta accuracy', channel_type=ChannelType.NUMERIC)
val_meta_loss_channel = ctx.create_channel('val meta loss', channel_type=ChannelType.NUMERIC)
val_meta_accuracy_channel = ctx.create_channel('val meta accuracy', channel_type=ChannelType.NUMERIC)
meta_lr_channel = ctx.create_channel('meta lr', channel_type=ChannelType.NUMERIC)

final_channel = ctx.create_channel('final', channel_type=ChannelType.TEXT)

# Load data
# Resize is done by the MetaDataset because the result can be easily cached
omniglot = MetaOmniglotFolder(args.input, size=(28, 28), cache=ImageCache(),
                              transform_image=transform_image,
                              transform_label=transform_label)
meta_train, meta_test = split_omniglot(omniglot)

print('Meta-Train characters', len(meta_train))
print('Meta-Test characters', len(meta_test))


# Loss
cross_entropy = nn.NLLLoss()
def get_loss(prediction, labels):
    return cross_entropy(prediction, labels)


def do_learning(net, optimizer, train_iter, iterations):
    net.train()
    for iteration in range(iterations):
        # Sample minibatch
        data, labels = Variable_(next(train_iter))

        # Forward pass
        prediction = net(data)

        # Get loss
        loss = get_loss(prediction, labels)

        # Backward pass - Update fast net
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def do_evaluation(net, train, test):
    net.eval()
    if args.transductive:
        test_batch = list(DataLoader(test, args.classes, shuffle=True))[0]
        data, labels = Variable_(test_batch)

        # Forward pass
        prediction = net(data)
        loss = get_loss(prediction, labels)

        # Get accuracy
        argmax = net.predict(prediction)
        accuracy = (argmax == labels).float().mean()

        return loss.item(), accuracy.item()
    else:
        losses = []
        accuracies = []
        for test_input, test_label in test:
            test_input = test_input[None, :, :, :]
            test_label = torch.tensor(test_label)[None]

            data, labels = list(DataLoader(train, args.classes, shuffle=True))[0]
            data = torch.cat([data, test_input])
            labels = torch.cat([labels, test_label])
            data, labels = Variable_(data), Variable_(labels)

            prediction = net(data)[-1:]
            label = labels[-1:]

            loss = get_loss(prediction, label)
            argmax = net.predict(prediction)
            accuracy = (argmax == label).float()

            losses.append(loss.item())
            accuracies.append(accuracy.item())
        return np.mean(losses), np.mean(accuracies)


def get_optimizer(net, state=None):
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0, 0.999))
    if state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Build model, optimizer, and set states
meta_net = OmniglotModel(args.classes)
if args.cuda:
    meta_net.cuda()
meta_optimizer = torch.optim.SGD(meta_net.parameters(), lr=args.meta_lr)
state = None

# Main loop
for meta_iteration in range(args.start_meta_iteration, args.meta_iters):

    # Update learning rate

    meta_lr = args.meta_lr * (1. - meta_iteration/float(args.meta_iters))
    set_learning_rate(meta_optimizer, meta_lr)

    mean_net = meta_net.clone()
    batch_nets = []
    for _ in range(args.meta_batch):
        # Clone model
        net = meta_net.clone()
        optimizer = get_optimizer(net, state)
        # load state of base optimizer?

        # Sample base task from Meta-Train
        train = meta_train.get_random_task(args.classes, args.train_shots)
        train_iter = make_infinite(DataLoader(train, args.train_batch, shuffle=True))

        # Update fast net
        loss = do_learning(net, optimizer, train_iter, args.train_iters)
        state = optimizer.state_dict()  # save optimizer state
        batch_nets.append(net)

    batch_params = [list(net.parameters()) for net in batch_nets]
    batch_params = zip(*batch_params)
    for mean_param, batch_param in zip(mean_net.parameters(), batch_params):
        np_mean_param = np.mean([param.data.cpu().numpy() for param in batch_param], axis=0)
        mean_param.data = Variable_(torch.FloatTensor(np_mean_param))

    # Update slow net
    meta_net.point_grad_to(mean_net)
    meta_optimizer.step()

    # Meta-Evaluation
    if meta_iteration % args.validate_every == 0:

        for (meta_dataset, mode) in [(meta_train, 'train'), (meta_test, 'val')]:

            net = meta_net.clone()
            optimizer = get_optimizer(net, state)  # do not save state of optimizer
            train, test = meta_dataset.get_random_task_split(args.classes, train_K=args.shots, test_K=1)

            train_iter = make_infinite(DataLoader(train, args.test_batch, shuffle=True))
            loss = do_learning(net, optimizer, train_iter, args.test_iters)

            meta_loss, meta_accuracy = do_evaluation(net, train, test)

            # (Logging)
            if mode == 'train':
                train_meta_loss_channel.send(x=meta_iteration, y=meta_loss)
                train_meta_accuracy_channel.send(x=meta_iteration, y=meta_accuracy)
            elif mode == 'val':
                val_meta_loss_channel.send(x=meta_iteration, y=meta_loss)
                val_meta_accuracy_channel.send(x=meta_iteration, y=meta_accuracy)
        meta_lr_channel.send(x=meta_iteration, y=meta_lr)

# evaluate
for (meta_dataset, mode) in [(meta_train, 'train'), (meta_test, 'val')]:
    eval_losses = []
    eval_accuracies = []
    for i in range(args.num_samples):
        net = meta_net.clone()
        optimizer = get_optimizer(net, state)  # do not save state of optimizer
        train, test = meta_dataset.get_random_task_split(args.classes, train_K=args.shots, test_K=1)

        train_iter = make_infinite(DataLoader(train, args.test_batch, shuffle=True))
        loss = do_learning(net, optimizer, train_iter, args.test_iters)

        meta_loss, meta_accuracy = do_evaluation(net, train, test)
        eval_losses.append(meta_loss)
        eval_accuracies.append(meta_accuracy)
    eval_loss = np.mean(eval_losses)
    eval_accuracy = np.mean(eval_accuracies)
    final_channel.send('{}:'.format(mode))
    final_channel.send('final metaloss: {}'.format(eval_loss))
    final_channel.send('final accuracy: {}'.format(eval_accuracy))
