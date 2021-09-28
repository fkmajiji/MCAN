import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from lapsrn import Net, L1_Charbonnier_loss
from tqdm import tqdm
import pandas as pd
from dataset_msimaker.data import get_training_set_opt, get_test_set_opt
from torch.utils.data import DataLoader
from math import log10

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=7500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-3, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=2000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default  ="checkpoint/De_happy_model_epoch_.pth", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument('--msfa_size', '-uf',  type=int, default=4, help="the size of square msfa")
parser.add_argument("--train_dir", default="G:\dataset\CAVE2\\new_train", type=str, help="path to train dataset")
parser.add_argument("--val_dir", default="CAVE_dataset/new_val", type=str, help="path to validation dataset")


def main() -> object:

    global opt, model
    opt = parser.parse_args()
    print(opt)
    cuda = True
    opt.cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    opt.norm_flag = False
    opt.augment_flag = False
    train_set = get_training_set_opt(opt.train_dir, opt.msfa_size,opt.norm_flag,opt.augment_flag)
    test_set = get_test_set_opt(opt.val_dir, opt.msfa_size,opt.norm_flag)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=24, shuffle=False)

    print("===> Building model")
    model = Net()
    criterion = L1_Charbonnier_loss()
    criterion1 = nn.MSELoss()
    print(model)
    print("===> Setting GPU")
    if cuda:
        device_flag = torch.device('cuda' )
        model = model.cuda()
        criterion = criterion.cuda()
        criterion1 = criterion1.cuda()
    else:
        device_flag = torch.device('cpu')
        model = model.cpu()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    save_opt(opt)
    print("===> Training")
    print('# parameters:', sum(param.numel() for param in model.parameters()))  # 输出模型参数数量
    results = {'im_loss': [], 're_loss': [], 'all_loss': [], 'psnr': []}
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        running_results = train(training_data_loader, optimizer, model, criterion, criterion1, epoch, opt.nEpochs)
        results['im_loss'].append(running_results['im_loss'] / running_results['batch_sizes'])
        results['re_loss'].append(running_results['re_loss'] / running_results['batch_sizes'])
        results['all_loss'].append(running_results['all_loss'] / running_results['batch_sizes'])
        test_results = test(testing_data_loader, optimizer, model, criterion, criterion1, epoch, opt.nEpochs)
        results['psnr'].append(test_results['psnr'])
        if epoch%250==0:
            save_checkpoint(model, epoch)
        if epoch!=0:
            save_statistics(opt, results, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, criterion1, epoch, num_epochs):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'im_loss': 0, 're_loss': 0, 'all_loss': 0}
    model.train()

    for batch in train_bar:
        input_raw, input, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
        N, C, H, W = batch[0].size()
        running_results['batch_sizes'] += N

        scale_coord_map = input_matrix_wpn(H, W)
        if opt.cuda:
            input = input.cuda()
            input_raw = input_raw.cuda()
            label_x4 = label_x4.cuda()
            scale_coord_map = scale_coord_map.cuda()

        HR_4x = model([input, input_raw], scale_coord_map)

        loss_x4 = 0.125 * criterion(HR_4x, label_x4)
        loss_raw = 0
        loss = loss_x4
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_results['im_loss'] += loss_x4.item()
        # running_results['re_loss'] += loss_raw.item()
        running_results['all_loss'] += loss.item()
        train_bar.set_description(desc='[%d/%d] Loss_im: %.4f Loss_re: %.1f Loss_all: %.1f' % (
            epoch, num_epochs, running_results['im_loss'] / running_results['batch_sizes'],
            running_results['re_loss'] / running_results['batch_sizes'],
            running_results['all_loss'] / running_results['batch_sizes']))
    return running_results

def test(testing_data_loader, optimizer, model, criterion, criterion1, epoch, num_epochs):
    test_bar = tqdm(testing_data_loader)
    test_results = {'batch_sizes': 0, 'psnr': 0, 'mse': 0}
    model.eval()

    with torch.no_grad():
        # for batch in training_data_loader:
        for batch in test_bar:
        # for batch_num, (input_raw, input, target) in enumerate(training_data_loader):

            input_raw, input, label_x4 = Variable(batch[0]), Variable(batch[1]), Variable(batch[2], requires_grad=False)
            # batch_size = batch[0].shape[0]
            # running_results['batch_sizes'] += batch_size
            N, C, H, W = batch[0].size()
            # input_raw, input, label_x4 = input_raw.to(device_flag),input.to(device_flag), target.to(device_flag)
            # N, C, H, W  = target.shape
            test_results['batch_sizes'] += N

            scale_coord_map = input_matrix_wpn(H, W)
            if opt.cuda:
                input = input.cuda()
                input_raw = input_raw.cuda()
                label_x4 = label_x4.cuda()
                scale_coord_map = scale_coord_map.cuda()

            HR_4x = model([input, input_raw], scale_coord_map)

            batch_mse = ((HR_4x - label_x4) ** 2).data.mean()
            test_results['mse'] += batch_mse * N
            test_results['psnr'] = 10 * log10(1 / (test_results['mse'] / test_results['batch_sizes']))

            test_bar.set_description(desc='[%d/%d] psnr: %.4f ' % (
                epoch, num_epochs, test_results['psnr']))
    return test_results

def input_matrix_wpn(inH, inW, add_id_channel=False):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''

    outH, outW = inH, inW
    # h_offset = torch.ones(inH, 1, 1)
    # w_offset = torch.ones(1, inW, 1)
    h_offset_coord = torch.zeros(inH, inW, 1)
    w_offset_coord = torch.zeros(inH, inW, 1)
    h_offset_coord[0::4, :, 0] = 0.25
    h_offset_coord[1::4, :, 0] = 0.5
    h_offset_coord[2::4, :, 0] = 0.75
    h_offset_coord[3::4, :, 0] = 1.0

    w_offset_coord[:, 0::4, 0] = 0.25
    w_offset_coord[:, 1::4, 0] = 0.5
    w_offset_coord[:, 2::4, 0] = 0.75
    w_offset_coord[:, 3::4, 0] = 1.0

    pos_mat = torch.cat((h_offset_coord, w_offset_coord), 2)
    pos_mat = pos_mat.contiguous().view(1, -1,2)

    return pos_mat ##outH*outW*2 outH=scale_int*inH , outW = scale_int *inW


def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "De_happy_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_opt(opt):
    statistics_folder = "checkpoint/"
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=vars(opt), index=range(1, 2))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_opt.csv', index_label='Epoch')
    print("save--opt")

def save_statistics(opt, results, epoch):
    statistics_folder = "checkpoint/"
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=results,index=range(opt.start_epoch, epoch + 1))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_train_results.csv', index_label='Epoch')
if __name__ == "__main__":
    main()
