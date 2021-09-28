import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math
import matplotlib.pyplot as plt
from libtiff import TIFFfile, TIFFimage
from os.path import join
from sklearn.metrics import mean_squared_error
from My_function import reorder_imec
from sewar.full_ref import ergas_matlab
from lapsrn import Net

def load_img(filepath):
    # img = Image.open(filepath+'/1.tif')
    # y = np.array(img).reshape(1,img.size[0],img.size[1])
    # m = np.tile(y, (2, 1, 1))
    tif = TIFFfile(filepath)
    picture, _ = tif.get_samples()
    img = picture[0].transpose(2, 1, 0)
    # img_test = Image.fromarray(img[:,:,1])
    return img

def mask_input(GT_image):
    mask = np.zeros((GT_image.shape[0], GT_image.shape[1], 16), dtype=np.float32)
    mask[0::4, 0::4, 0] = 1
    mask[0::4, 1::4, 1] = 1
    mask[0::4, 2::4, 2] = 1
    mask[0::4, 3::4, 3] = 1
    mask[1::4, 0::4, 4] = 1
    mask[1::4, 1::4, 5] = 1
    mask[1::4, 2::4, 6] = 1
    mask[1::4, 3::4, 7] = 1
    mask[2::4, 0::4, 8] = 1
    mask[2::4, 1::4, 9] = 1
    mask[2::4, 2::4, 10] = 1
    mask[2::4, 3::4, 11] = 1
    mask[3::4, 0::4, 12] = 1
    mask[3::4, 1::4, 13] = 1
    mask[3::4, 2::4, 14] = 1
    mask[3::4, 3::4, 15] = 1
    input_image = mask * GT_image
    return input_image

def psnr(x_true, x_pred):
    # print(x_true.shape)
    # print(x_pred.shape)
    n_bands = x_true.shape[2]
    PSNR = np.zeros(n_bands)
    MSE = np.zeros(n_bands)
    mask = np.ones(n_bands)
    x_true = x_true[:, :, :]
    for k in range(n_bands):
        x_true_k = x_true[:, :, k].reshape([-1])
        x_pred_k = x_pred[:, :, k, ].reshape([-1])

        MSE[k] = mean_squared_error(x_true_k, x_pred_k, )

        MAX_k = np.max(x_true_k)
        if MAX_k != 0:
            PSNR[k] = 10 * math.log10(math.pow(MAX_k, 2) / MSE[k])
            # print ('P', PSNR[k])
        else:
            mask[k] = 0

    psnr = PSNR.sum() / mask.sum()
    mse = MSE.mean()
    # print('psnr', psnr)
    # print('mse', mse)
    return psnr, mse

def ssim(x_true, x_pre):
    num = x_true.shape[2]
    ssimm = np.zeros(num)
    c1 = 0.0001
    c2 = 0.0009
    n = 0
    for x in range(x_true.shape[2]):
        z = np.reshape(x_pre[:, :, x], [-1])
        sa = np.reshape(x_true[:, :, x], [-1])
        y = [z, sa]
        cov = np.cov(y)
        oz = cov[0, 0]
        osa = cov[1, 1]
        ozsa = cov[0, 1]
        ez = np.mean(z)
        esa = np.mean(sa)
        ssimm[n] = ((2 * ez * esa + c1) * (2 * ozsa + c2)) / ((ez * ez + esa * esa + c1) * (oz + osa + c2))
        n = n + 1
    SSIM = np.mean(ssimm)
    # print ('SSIM',SSIM)
    return SSIM

def sam(x_true, x_pre):
    # print x_pre.shape
    # print x_true.shape
    num = (x_true.shape[0]) * (x_true.shape[1])
    samm = np.zeros(num)
    n = 0
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            z = np.reshape(x_pre[x, y, :], [-1])
            sa = np.reshape(x_true[x, y, :], [-1])
            tem1 = np.dot(z, sa) + 2.2204e-16
            tem2 = (np.linalg.norm(z) + 2.2204e-16) * (np.linalg.norm(sa) + 2.2204e-16)
            # print(tem1/tem2)
            buff1 = tem1 / tem2
            if buff1 > 1:
                samm[n] = 0
                # samm[n] = np.arccos(buff1)
            else:
                samm[n] = np.arccos(buff1)
            n = n + 1
    # print(np.mean(samm))
    buff = np.mean(samm)
    SAM = (buff) * 180 / np.pi
    # print('SAM',SAM)
    return SAM

def sam1(x_true, x_pre):
    buff1 = x_true * x_pre
    buff2 = np.sum(buff1, 2)
    buff2[buff2 == 0] = 2.2204e-16
    buff4 = np.sqrt(np.sum(x_true * x_true, 2))
    buff4[buff4 == 0] = 2.2204e-16
    buff5 = np.sqrt(np.sum(x_pre * x_pre, 2))
    buff5[buff5 == 0] = 2.2204e-16
    buff6 = buff2 / buff4
    buff8 = buff6 / buff5
    buff8[buff8 > 1] = 1
    buff9 = np.mean(np.arccos(buff8))
    SAM = (buff9) * 180 / np.pi
    return SAM

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def input_matrix_wpn(inH, inW, add_id_channel=False):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''

    outH, outW = inH, inW
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

    return pos_mat

parser = argparse.ArgumentParser(description="PyTorch LapSRN Demo")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/mcan_model.pth", type=str, help="model path")
parser.add_argument("--val_dir", default="CAVE_dataset/new_val", type=str, help="model path")
parser.add_argument("--image", default="beads_ms", type=str, help="image name")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")

opt = parser.parse_args()
cuda = True

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

model = Net()
m_state_dict = torch.load(opt.model)
model.load_state_dict(m_state_dict)
im_gt_y = load_img(opt.val_dir + "\\" + opt.image + "_" + "IMECMine_D65" + ".tif")  # 512, 512, 16 shape# PPID = load_img(join("G:\dataset\CAVE2\\validation_CAVE", opt.image, opt.image, "PPID_HA.tif"))  # 512, 512, 16 shape
max_new = np.max(im_gt_y)
im_gt_y = im_gt_y / max_new * 255

im_gt_y = im_gt_y.transpose(1, 0, 2)

im_l_y = mask_input(im_gt_y)
im_l_y = reorder_imec(im_l_y)
im_gt_y = reorder_imec(im_gt_y)

im_gt_y = im_gt_y.astype(float)
im_l_y = im_l_y.astype(float)

im_input = im_l_y / 255.
im_gt_y = im_gt_y.transpose(2, 0, 1)
im_l_y = im_l_y.transpose(2, 0, 1)
im_input = im_input.transpose(2, 0, 1)

raw = im_input.sum(axis=0)
scale_coord_map = input_matrix_wpn(raw.shape[0], raw.shape[1])
im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[1], im_input.shape[2])
raw = Variable(torch.from_numpy(raw).float()).view(1, -1, raw.shape[0], raw.shape[1])

if cuda:
    model = model.cuda()
    im_input = im_input.cuda()
    raw = raw.cuda()
    scale_coord_map = scale_coord_map.cuda()
else:
    model = model.cpu()

start_time = time.time()
HR_4x = model([im_input, raw], scale_coord_map)
elapsed_time = time.time() - start_time

HR_4x = HR_4x.cpu()
im_h_y = HR_4x.data[0].numpy().astype(np.float32)

im_h_y = im_h_y * 255.
im_h_y = np.rint(im_h_y)
im_h_y[im_h_y < 0.0] = 0.0
im_h_y[im_h_y > 255.] = 255.
im_h_y = im_h_y.astype(np.uint8)
im_h_y = im_h_y.astype(np.float)

raw = raw.cpu()
raw = raw.data[0].numpy().astype(np.float32)
raw = raw * 255.
raw[raw < 0] = 0
raw[raw > 255.] = 255.

im_input = im_input.cpu()
im_input = im_input.data[0].numpy().astype(np.float32)
im_input = im_input * 255.
im_input[im_input < 0] = 0
im_input[im_input > 255.] = 255.

[psnr_predicted, mse] = psnr(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("PSNR_multi=", psnr_predicted)
ssim_predicted = ssim(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("ssim_multi=", ssim_predicted)
sam_predicted = sam1(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("sam_multi=", sam_predicted)
ergas_predicted = ergas_matlab(im_gt_y.transpose(2, 1, 0), im_h_y.transpose(2, 1, 0))
print("ergas_predicted=", ergas_predicted)
print("It takes {}s for processing".format(elapsed_time))

# tiff = TIFFimage(raw.astype(np.uint8), description='')
# tiff.write_file('raw.tif', compression='none')  # or 'lzw'

nband = 0
fig = plt.figure()
ax = plt.subplot("221")
ax.imshow(im_gt_y[nband, :, :], cmap='gray')
ax.set_title("GT")

ax = plt.subplot("222")
ax.imshow(im_input[nband, :, :], cmap='gray')
ax.set_title("Input(band1)")

buff2 = raw[0, :, :]
ax = plt.subplot("223")
ax.imshow(raw[0, :, :], cmap='gray')
ax.set_title("Input(raw)")

ax = plt.subplot("224")
ax.imshow(im_h_y[nband, :, :], cmap='gray')
ax.set_title("Output(MCAN)")
plt.show()

