import argparse
import os
import time

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import datetime
from torch import nn

from utils_clom.Dataloader import get_dataset
from utils_clom.model_pool import get_network
from utils_clom.trainer import train, validate
from utils_clom.utils import set_random_seed, set_gpu, get_logger, get_pretrained_model_root


def train_model(args):
    args.random_seed = int(time.time() * 1000) % 100000 if args.random_seed is None else args.random_seed
    set_random_seed(args.random_seed)

    if args.data_path is None:
        args.data_path = os.path.join('data', args.dataset)
    print(os.path.join('data', args.dataset))

    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path, exist_ok=True)

    save_root = get_pretrained_model_root(args.dataset, args.model)
    if not os.path.isdir(save_root):
        os.makedirs(save_root, exist_ok=True)

    print("data path:", args.data_path)

    index = 0
    while os.path.exists(os.path.join(save_root, f"{args.dataset}_{args.model}_original_{index}.pt")):
        index = index + 1
    save_name = f"{args.dataset}_{args.model}_original_{index}.pt"
    logger_name = f"{args.dataset}_{args.model}_original_{index}_log.log"

    # log information
    logger, file_handler, stream_handler = get_logger(os.path.join(save_root,logger_name))
    logger.info("Time: "+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logger.info("Dataset: "+args.dataset)
    logger.info("model: "+args.model)
    logger.info("random seed: "+str(args.random_seed))
    logger.info("epochs: "+str(args.epochs))
    logger.info("batch size: "+str(args.batch_size))
    logger.info("lr: "+str(args.lr))
    logger.info("momentum: "+str(args.momentum))
    logger.info("weight decay: "+str(args.weight_decay))
    logger.info("lr decay step: "+ args.lr_decay_step)
    logger.info("normalize data: "+str(args.normalize_data))

    # load dataset
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args)

    # get train loader
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # get model
    model = get_network(args.model, channel, num_classes, im_size)
    model = set_gpu(args, model)

    # train settings
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_decay_step = list(map(int, args.lr_decay_step.split(',')))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_step, gamma=0.1)

    best_acc1 = 0.0
    # start train
    logger.info("start training")
    for epoch in range(args.epochs):
        train_acc1, _,  loss, = train(trainloader, model, criterion, optimizer, epoch, args, aug=False)
        scheduler.step()
        if (epoch + 1) % args.save_every == 0:
            acc1, _ = validate(testloader, model, criterion, args)
            logger.info(f"epoch:{epoch}, train acc:{train_acc1} loss: {loss} test acc:{acc1}")
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                torch.save(model.state_dict(), os.path.join(save_root, save_name))

    logger.info(f"dataset: {args.dataset}, model: {args.model}")
    logger.info(f"best acc: {best_acc1}")
    logger.info(f"save path: {os.path.join(save_root, save_name)}")
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)


def main(args):
    for i in range(args.num):
        train_model(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Testing", epilog="End of Parameters")
    parser.add_argument("--dataset", help="name of dataset", type=str, default='FashionMNIST',
                        choices=["MNIST", "FashionMNIST", "SVHN", "CIFAR10", "CIFAR100", "tinyimagenet"])
    parser.add_argument("--model", metavar="ARCH", default='ConvNet', help="model architecture")

    parser.add_argument("--random_seed", default=None, type=int, help="random seed")
    parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")],
                        help="Which GPUs to use for multigpu training")
    parser.add_argument("--gpu", default=0, type=int, help="Which GPU to use for training")
    parser.add_argument("--num", default=1, type=int, help="How many models need to be trained")

    parser.add_argument("--epochs", default=150, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--batch_size", default=256, type=int, help="batch_size")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum(defalt: 0.9)")
    parser.add_argument('--weight_decay', type=float, default=0.005, help="SGD weight decay")
    parser.add_argument("--save_every", default=1, type=int, help="how many epochs to save")
    parser.add_argument('--lr_decay_step', default='50,100', type=str, help='learning rate')
    parser.add_argument('--normalize_data', action="store_true", default=False,
                        help='whether normalize dataset')

    parser.add_argument('--data_path', type=str, default=None, help='dataset path')
    args = parser.parse_args()
    main(args)