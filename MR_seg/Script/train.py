import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import nibabel as nib
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


from networks.vnet import VNet
from utils.losses import dice_loss
from dataloaders.Loader import (MR_seg,  RandomCrop, CenterCrop, RandomRotFlip, ToTensor,
                                TwoStreamBatchSampler, RandomNoise, NormThresholding, Padding,
                                )
from sklearn.model_selection import KFold, train_test_split
from monai.metrics import DiceMetric
from utils.util import save_image, show_slice_img, val_show_slice_img
from monai.transforms import RandAffined


def load_data(img_path, label_path):
    img_lst = []
    label_lst = []

    for img_file, label_file in zip(os.listdir(img_path),
                                    os.listdir(label_path)):
        img_lst.append(os.path.join(img_path, img_file))
        label_lst.append(os.path.join(label_path, label_file))

    return np.array(img_lst), np.array(label_lst)

img_path = 'Data/mr'
label_path = 'Data/mr_mask'

img_data, label_data = load_data(img_path, label_path)

img_train, img_val, label_train, label_val = train_test_split(img_data, label_data, test_size=0.2, random_state=419)

train_files = [{"image": img, "label": label}
               for img, label in
               zip(img_train, label_train)]
val_files = [{"image": img, "label": label}
             for img, label in
             zip(img_val, label_val)]


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='..', help='Name of Experiment')
# parser.add_argument('--exp', type=str,  default='roi_localization', help='model_name')
parser.add_argument('--exp', type=str,  default='..', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=30001, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
# snapshot_path = "../model_save/" + args.exp + "/"
# snapshot_path = "../model_save/"
snapshot_path = "../model_save_temp/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (128, 128, 128)
# patch_size = (256, 256, 256)
num_classes = 2

if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()
    #save_mode_path = os.path.join('/hpc/data/home/bme/v-cuizm/project/NC/model/binary_seg_ROI_HZ_02_(142data_256size)/iter_10000.pth')
    #net.load_state_dict(torch.load(save_mode_path))
    net.load_state_dict(torch.load(r''))

    db_train = MR_seg(data=train_files, transform=transforms.Compose([
                            NormThresholding(up=75, low=20),
                            Padding(),
                            RandAffined(keys=['image', 'label'],
                                        mode=['bilinear', 'nearest'],
                                        prob=1.0,
                                        rotate_range=(np.pi/24, np.pi/24, 0),
                                        translate_range=(10, 10, 0)
                                        ),
                            # RandomCrop(patch_size),
                            RandomNoise(),
                            # RandomRotFlip(),
                            ToTensor(),
                            ]))
    db_test = MR_seg(data=val_files, transform=transforms.Compose([
                            NormThresholding(up=75, low=20),
                            Padding(),
                            # RandAffined(keys=['image', 'label'],
                            #             mode=['bilinear', 'nearest'],
                            #             prob=1.0,
                            #             rotate_range=(np.pi/24, np.pi/24, 0),
                            #             translate_range=(10, 10, 0)
                            #             ),
                            ToTensor()
                       ]))


    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    Dice = DiceMetric(include_background=False)

    '''vis'''
    with torch.no_grad():
        for i, val_data in enumerate(testloader):
            val_img = val_data["image"].cuda()
            print(val_img.shape)
            val_label = val_data["label"].cuda()
            # print("*" * 5)
            # print(val_img.shape)
            # print(val_label.shape)

            # show_slice_img(val_img, val_label, 32)
            outputs = net(val_img)
            outputs_soft = F.softmax(outputs, dim=1)
            val_show_slice_img(val_img, val_label, outputs_soft, 32)


    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = net(volume_batch)

            loss_seg = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            loss = 0.5*(loss_seg+loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f , dice_acc : %f' % (iter_num, loss.item(), 1-loss_seg_dice.item()))

            ## change lr
            if iter_num % 5000 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_dir = snapshot_path + "pth/"
                save_mode_path = os.path.join(save_dir, 'iter_' + str(iter_num) + '.pth')
                # save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            # if iter_num % 10000 == 0:
            #     save_image()



            if iter_num % 200 == 0:
                net.eval()
                test_loss = 0
                iter_test = 0
                DSC = []
                for i_batch, sampled_batch in enumerate(testloader):
                    volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    with torch.no_grad():
                        outputs = net(volume_batch)

                    loss_seg = F.cross_entropy(outputs, label_batch)
                    outputs_soft = F.softmax(outputs, dim=1)

                    '''vis'''
                    if iter_num % 30000 == 0:
                        val_show_slice_img(volume_batch, label_batch, outputs_soft, 32)

                    loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
                    loss = 0.5 * (loss_seg+loss_seg_dice)
                    print('---test for seg:', 1 - loss_seg_dice.item())
                    DSC.append(1 - loss_seg_dice.item())
                    test_loss = test_loss + loss
                    iter_test = iter_test + 1
                print('mean DSC: ', np.mean(DSC))
                writer.add_scalar('loss_test/test_loss', test_loss/iter_test, iter_num)
                writer.add_scalar('mean DSC', np.mean(DSC), iter_num)

                net.train()
                del volume_batch, label_batch, loss_seg, outputs_soft, loss_seg_dice



            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
