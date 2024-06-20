__name__ = 'Osama Raja'

import scipy.io as sio, os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils_osama import *
import torch.autograd
from grappa_utils import *
import numpy as np


def cnn_3layer(input_kspace, w1, w2, w3, acc_rate):
    input_kspace = torch.from_numpy(input_kspace).float()
    w1 = np.transpose(w1,(3,2,0,1))
    w2 = np.transpose(w2, (3, 2, 0, 1))
    w3 = np.transpose(w3, (3, 2, 0, 1))
    w1 = torch.from_numpy(w1).float()
    w2 = torch.from_numpy(w2).float()
    w3 = torch.from_numpy(w3).float()
    input_kspace = input_kspace.permute(0, 3, 1, 2)

    def conv2d_dilate(x, w, dilate_rate):
        return F.conv2d(x, w, padding=0, dilation=(1, dilate_rate))

    h_conv1 = F.relu(conv2d_dilate(input_kspace, w1, acc_rate))
    h_conv2 = F.relu(conv2d_dilate(h_conv1, w2, acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3, acc_rate)
    h_conv3.detach().numpy()
    h_conv3 = np.transpose(h_conv3,(0,2,3,1))
    return h_conv3

def weight_variable(shape):
    initial = torch.randn(shape, dtype=torch.float32) * 0.1
    return nn.Parameter(initial)

def conv2d(x, weight):
    return F.conv2d(x, weight, stride=1, padding=0)

def conv2d_dilate(x, weight, dilate_rate):
    return F.conv2d(x, weight, stride=1, padding=0, dilation=(1,dilate_rate))

def learning(ACS_input, target_input, accrate_input):
    input_ACS = torch.tensor(ACS_input, dtype=torch.float32)
    input_ACS = input_ACS.permute(0,3,1,2)
    input_Target = torch.tensor(target_input, dtype=torch.float32)
    _, ACS_dim_X, ACS_dim_Y, ACS_dim_Z = input_ACS.shape
    _, target_dim_X, target_dim_Y, target_dim_Z = input_Target.shape

    W_conv1 = weight_variable([layer1_channels,ACS_dim_X,kernel_x_1,kernel_y_1])
    W_conv2 = weight_variable([layer2_channels,layer1_channels,kernel_x_2,kernel_y_2])
    W_conv3 = weight_variable([target_dim_Z, layer2_channels, kernel_last_x, kernel_last_y]) #
    h_conv1 = F.relu(conv2d_dilate(input_ACS, W_conv1, accrate_input))
    h_conv2 = F.relu(conv2d_dilate(h_conv1, W_conv2, accrate_input))
    h_conv3 = conv2d_dilate(h_conv2, W_conv3, accrate_input)

    input_Target = input_Target.permute(0,3,1,2)
    error_norm = torch.norm(input_Target - h_conv3)
    optimizer = optim.Adam([W_conv1, W_conv2, W_conv3], lr=LearningRate)
    error_prev = 1
    for i in range(MaxIteration):
        optimizer.zero_grad()
        output = conv2d_dilate(input_ACS, W_conv1, accrate_input)
        output = F.relu(output)
        output = conv2d_dilate(output, W_conv2, accrate_input)
        output = F.relu(output)
        output = conv2d_dilate(output, W_conv3, accrate_input)
        error = torch.norm(input_Target - output)
        error.backward()
        optimizer.step()

        if i % 100 == 0:
            print('The', i, 'th iteration gives an error', error.item())
    error = torch.norm(input_Target - output).item()
    return [W_conv1.detach().numpy(), W_conv2.detach().numpy(), W_conv3.detach().numpy(), error]

kernel_x_1 = 5
kernel_y_1 = 2
kernel_x_2 = 1
kernel_y_2 = 1
kernel_last_x = 3
kernel_last_y = 2
layer1_channels = 32
layer2_channels = 8
MaxIteration = 1000
LearningRate = 3e-3

inputData = 'coronal2Ddataset/coronal_data1_PE_undersampledx2.mat'
input_variable_name = 'kspace'
resultName = 'RAKI_recon'
recon_variable_name = 'kspace_recon'
kspace = sio.loadmat(inputData)
kspace = kspace[input_variable_name]
no_ACS_flag = 0

normalize = 0.015 / np.max(abs(kspace[:]))
kspace = np.multiply(kspace, normalize)
[m1, n1, no_ch] = np.shape(kspace)
no_inds = 1

kspace_all = kspace
kx = np.transpose(np.int32([(range(1, m1 + 1))])) # (640,1)
ky = np.int32([(range(1, n1 + 1))]) # (1,484)
kspace = np.copy(kspace_all)

mask = np.squeeze(np.sum(np.sum(np.abs(kspace[:,:,:]), 0), 1)) > 0
picks = np.where(mask == 1)
kspace = kspace[:, np.int32(picks[0][0]):n1 + 1, :]
kspace_all = kspace_all[:, np.int32(picks[0][0]):n1 + 1, :]
kspace_NEVER_TOUCH = np.copy(kspace_all)

mask = np.squeeze(np.sum(np.sum(np.abs(kspace), 0), 1)) > 0
picks = np.where(mask == 1)
d_picks = np.diff(picks, 1)
indic = np.where(d_picks == 1)

mask_x = np.squeeze(np.sum(np.sum(np.abs(kspace), 2), 1)) > 0
picks_x = np.where(mask_x == 1)
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

indic = indic[1][:]
center_start = picks[0][indic[0]]
center_end = picks[0][indic[-1] + 1]
ACS = kspace[x_start:x_end + 1, center_start:center_end + 1, :]
[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
ACS_re = np.zeros([ACS_dim_X, ACS_dim_Y, ACS_dim_Z * 2])
ACS_re[:, :, 0:no_ch] = np.real(ACS)
ACS_re[:, :, no_ch:no_ch * 2] = np.imag(ACS)

acc_rate = d_picks[0][0]
print(f'acc_rate: {acc_rate}')
no_channels = ACS_dim_Z * 2

name_weight = resultName + ('_weight_%d%d,%d%d,%d%d_%d,%d.mat' % (
kernel_x_1, kernel_y_1, kernel_x_2, kernel_y_2, kernel_last_x, kernel_last_y, layer1_channels, layer2_channels))
name_image = resultName + ('_image_%d%d,%d%d,%d%d_%d,%d.mat' % (
kernel_x_1, kernel_y_1, kernel_x_2, kernel_y_2, kernel_last_x, kernel_last_y, layer1_channels, layer2_channels))

existFlag = os.path.isfile(name_image)

w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels], dtype=np.float32)
w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels, no_channels], dtype=np.float32)
w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels, acc_rate - 1, no_channels], dtype=np.float32)

target_x_start = np.int32(np.ceil(kernel_x_1 / 2) + np.floor(kernel_x_2 / 2) + np.floor(kernel_last_x / 2) - 1)
target_x_end = np.int32(ACS_dim_X - target_x_start - 1)

time_ALL_start = time.time()

[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
ACS = np.reshape(ACS_re, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])
ACS = np.float32(ACS)

target_y_start = np.int32(
    (np.ceil(kernel_y_1 / 2) - 1) + (np.ceil(kernel_y_2 / 2) - 1) + (np.ceil(kernel_last_y / 2) - 1)) * acc_rate
target_y_end = ACS_dim_Y - np.int32(
    (np.floor(kernel_y_1 / 2) + np.floor(kernel_y_2 / 2) + np.floor(kernel_last_y / 2))) * acc_rate - 1

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = acc_rate - 1

print('go!')
time_Learn_start = time.time()

errorSum = 0

for ind_c in range(ACS_dim_Z):

    target = np.zeros([1, target_dim_X, target_dim_Y, target_dim_Z])
    print('learning channel #', ind_c + 1)
    time_channel_start = time.time()

    for ind_acc in range(acc_rate - 1):
        target_y_start = np.int32((np.ceil(kernel_y_1 / 2) - 1) + (np.ceil(kernel_y_2 / 2) - 1) + (
                np.ceil(kernel_last_y / 2) - 1)) * acc_rate + ind_acc + 1
        target_y_end = ACS_dim_Y - np.int32(
            (np.floor(kernel_y_1 / 2) + (np.floor(kernel_y_2 / 2)) + np.floor(kernel_last_y / 2))) * acc_rate + ind_acc
        target[0, :, :, ind_acc] = ACS[0, target_x_start:target_x_end + 1, target_y_start:target_y_end + 1, ind_c]

    [w1, w2, w3, error] = learning(ACS, target, acc_rate)

    w1_all[:, :, :, :, ind_c] = np.transpose(w1,(2,3,1,0))
    w2_all[:, :, :, :, ind_c] = np.transpose(w2,(2,3,1,0))
    w3_all[:, :, :, :, ind_c] = np.transpose(w3,(2,3,1,0))
    time_channel_end = time.time()
    print('Time Cost:', time_channel_end - time_channel_start, 's')
    print('Norm of Error = ', error)
    errorSum = errorSum + error

time_Learn_end = time.time()
print('learning step costs:', (time_Learn_end - time_Learn_start) / 60, 'min')
sio.savemat(name_weight, {'w1': w1_all, 'w2': w2_all, 'w3': w3_all})

kspace_recon_all = np.copy(kspace_all)
kspace_recon_all_nocenter = np.copy(kspace_all)
kspace = np.copy(kspace_all)

over_samp = np.setdiff1d(picks, np.int32([range(0, n1, acc_rate)]))
kspace_und = kspace
kspace_und[:, over_samp, :] = 0

[dim_kspaceUnd_X, dim_kspaceUnd_Y, dim_kspaceUnd_Z] = np.shape(kspace_und)
kspace_und_re = np.zeros([dim_kspaceUnd_X, dim_kspaceUnd_Y, dim_kspaceUnd_Z * 2])
kspace_und_re[:, :, 0:dim_kspaceUnd_Z] = np.real(kspace_und)
kspace_und_re[:, :, dim_kspaceUnd_Z:dim_kspaceUnd_Z * 2] = np.imag(kspace_und)
kspace_und_re = np.float32(kspace_und_re)
kspace_und_re = np.reshape(kspace_und_re, [1, dim_kspaceUnd_X, dim_kspaceUnd_Y, dim_kspaceUnd_Z * 2])
kspace_recon = kspace_und_re
kspace_recon_for_plt = np.transpose(kspace_recon, (0,2,1,3)) # new addition
plt.imshow(abs(kspace_recon_for_plt[0,:,:,-1]), cmap='gray', norm=colors.PowerNorm(gamma=0.2) )
plt.title(f'undersampled kspace x{acc_rate}')
plt.show()

for ind_c in range(0, no_channels):
    print('Reconstruting Channel #', ind_c + 1)
    w1 = np.float32(w1_all[:, :, :, :, ind_c])
    w2 = np.float32(w2_all[:, :, :, :, ind_c])
    w3 = np.float32(w3_all[:, :, :, :, ind_c])

    res = cnn_3layer(kspace_und_re, w1, w2, w3, acc_rate)
    target_x_end_kspace = dim_kspaceUnd_X - target_x_start

    for ind_acc in range(0, acc_rate - 1):
        target_y_start = np.int32((np.ceil(kernel_y_1 / 2) - 1) + np.int32((np.ceil(kernel_y_2 / 2) - 1)) + np.int32(
            np.ceil(kernel_last_y / 2) - 1)) * acc_rate + ind_acc + 1
        target_y_end_kspace = dim_kspaceUnd_Y - np.int32(
            (np.floor(kernel_y_1 / 2)) + (np.floor(kernel_y_2 / 2)) + np.floor(kernel_last_y / 2)) * acc_rate + ind_acc
        kspace_recon[0, target_x_start:target_x_end_kspace, target_y_start:target_y_end_kspace + 1:acc_rate,
        ind_c] = res[0, :, ::acc_rate, ind_acc]

kspace_recon = np.squeeze(kspace_recon)
kspace_recon_complex = (kspace_recon[:, :, 0:np.int32(no_channels / 2)] + np.multiply(
    kspace_recon[:, :, np.int32(no_channels / 2):no_channels], 1j))
kspace_recon_all_nocenter[:, :, :] = np.copy(kspace_recon_complex)

kspace_gt = scipy.io.loadmat('coronal2Ddataset/coronal_data1_fullysampled.mat')
kspace_gt = kspace_gt['rawdata']
kspace_gt = np.array(kspace_gt)
normalize = 0.015 / np.max(abs(kspace[:]))
kspace_gt = np.multiply(kspace_gt, normalize)
freq_domain_gt = kspace_gt.copy()

no_ACS_flag= 0

if no_ACS_flag == 0:
    kspace_recon_complex[:, center_start:center_end, :] = kspace_NEVER_TOUCH[:, center_start:center_end, :]
    print('ACS signal has been putted back')

    fig, axes = plt.subplots(1, 2)
    freq_domain_gt = np.transpose(freq_domain_gt, (1, 0, 2))
    axes[0].imshow(abs(freq_domain_gt[:, :, -1]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    axes[0].set_title('GT - kspace')
    kspace_recon_complex_plt = np.transpose(kspace_recon_complex, (1,0,2))
    axes[1].imshow(abs(kspace_recon_complex_plt[:, :, -1]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    axes[1].set_title(f'RAKI_recon kspace')
    plt.tight_layout()
    plt.show()
else:
    print('No ACS signal is putted into k-space')

    fig, axes = plt.subplots(1, 2)
    freq_domain_gt = np.transpose(freq_domain_gt,(1,0,2))
    axes[0].imshow(abs(freq_domain_gt[:,:,-1]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    axes[0].set_title('GT - kspace')

    axes[1].imshow(abs(kspace_recon_complex[:, :, -1]), cmap='gray', norm=colors.PowerNorm(gamma=0.2))
    axes[1].set_title(f'RAKI_recon kspace')
    plt.tight_layout()
    plt.show()

kspace_recon_all[:, :, :] = kspace_recon_complex

for sli in range(0, no_ch):
    kspace_recon_all[:, :, sli] = np.fft.ifft2(kspace_recon_all[:, :, sli])

rssq = (np.sum(np.abs(kspace_recon_all) ** 2, 2) ** (0.5))
sio.savemat(name_image, {recon_variable_name: kspace_recon_complex})

time_ALL_end = time.time()
print('All process costs ', (time_ALL_end - time_ALL_start) / 60, 'mins')
print('Error Average in Training is ', errorSum / no_channels)

kspace_RAKI_reconstructed = sio.loadmat('RAKI_recon_image_52,11,32_32,8')
kspace_RAKI_reconstructed = kspace_RAKI_reconstructed['kspace_recon']
kspace_RAKI_reconstructed = np.array(kspace_RAKI_reconstructed)

image_space_RAKI_2d = ifft_centered(kspace_RAKI_reconstructed)
freq_domain_gt_2d = freq_domain_gt.copy()
image_space_gt_2d = ifft_centered(freq_domain_gt_2d)

compare_with_grappa = 1

if compare_with_grappa == 1:
    g23 = GRAPPA(freq_domain_gt, nACS=40, kernel_size=(kernel_y_1, kernel_x_1))

    gg23 = g23.grappa(acc_rate)
    image_space_gg23 = ifft2c(gg23, axes=(0, 1))
    combined_image_grappa = np.linalg.norm(image_space_gg23, axis=-1)  # SoS

    image_space_gt_2d_combined = np.linalg.norm(image_space_gt_2d, axis=-1)
    
    cropped_image_space_gt_2d = crop_center(image_space_gt_2d[:, :, -1])
    cropped_image_space_RAKI_2d = crop_center(image_space_RAKI_2d[:, :, -1])
    cropped_image_space_RAKI_2d = cropped_image_space_RAKI_2d.T
    cropped_combined_image_grappa = crop_center(combined_image_grappa)

    # Normalizing the images
    norm_image_space_gt_2d = normalize_image(abs(cropped_image_space_gt_2d))

    norm_image_space_RAKI_2d = normalize_image(abs(cropped_image_space_RAKI_2d))

    norm_combined_image_grappa = normalize_image(abs(cropped_combined_image_grappa))

    plt.imshow(abs(norm_image_space_gt_2d), cmap='gray')
    plt.title('THE GT normalized')
    plt.show()
    plt.imshow(abs(norm_image_space_RAKI_2d), cmap='gray')
    plt.title('RAKI normalized')
    plt.show()
    plt.imshow(abs(norm_combined_image_grappa), cmap='gray')
    plt.title('GRAPPA normalized')
    plt.show()

    mse_raki = abs(np.mean((norm_image_space_gt_2d - norm_image_space_RAKI_2d) ** 2))
    mse_grappa = abs(np.mean((norm_image_space_gt_2d - norm_combined_image_grappa) ** 2))

    psnr_raki = calculate_psnr(mse_raki)
    psnr_grappa = calculate_psnr(mse_grappa)

    ssim_value_raki = compute_ssim(norm_image_space_gt_2d, norm_image_space_RAKI_2d)
    ssim_value_grappa = compute_ssim(norm_image_space_gt_2d, norm_combined_image_grappa)

    nmse_raki = compute_nmse(norm_image_space_gt_2d, norm_image_space_RAKI_2d)
    nmse_grappa = compute_nmse(norm_image_space_gt_2d, norm_combined_image_grappa)

    print('------------------------------------------')
    print(f'MSE_RAKI: {mse_raki:.4f}')
    print(f'MSE_GRAPPA: {mse_grappa:.4f}')
    print('------------------------------------------')
    print(f"NMSE for RAKI: {nmse_raki:.4f}")
    print(f"NMSE for GRAPPA: {nmse_grappa:.4f}")
    print('------------------------------------------')
    print(f'PSNR_RAKI: {psnr_raki:.4f} dB')
    print(f'PSNR_GRAPPA: {psnr_grappa:.4f} dB')
    print('------------------------------------------')
    print(f"SSIM for RAKI: {ssim_value_raki:.4f}")
    print(f"SSIM for GRAPPA: {ssim_value_grappa:.4f}")
    print('------------------------------------------')

    fig, axes = plt.subplots(1, 3)
    image_space_gt_2d = norm_image_space_gt_2d.T
    axes[0].imshow(abs(image_space_gt_2d), cmap='gray')
    axes[0].set_title('GT')
    cropped_image_space_RAKI_2d = norm_image_space_RAKI_2d.T
    axes[1].imshow(abs(cropped_image_space_RAKI_2d), cmap='gray')
    axes[1].set_title(f'RAKI recon (R = {acc_rate})')
    combined_image_grappa = norm_combined_image_grappa.T
    axes[2].imshow(abs(combined_image_grappa), cmap='gray')
    axes[2].set_title(f'GRAPPA recon (R={acc_rate})')
    plt.tight_layout()
    plt.show()

else:
    fig, axes = plt.subplots(1, 2)
    print(f'the shape of gt in the plot: {image_space_gt_2d.shape}')
    axes[0].imshow(abs(image_space_gt_2d[:,:,-1]), cmap='gray')
    axes[0].set_title('GT')
    image_space_RAKI_2d = np.transpose(image_space_RAKI_2d, (1,0,2))
    print(f'the shape of RAKI recon in the plot: {image_space_RAKI_2d.shape}')
    axes[1].imshow(abs(image_space_RAKI_2d[:,:,-1]), cmap='gray')
    axes[1].set_title(f'RAKI recon (R = {acc_rate})')
    plt.tight_layout()
    plt.show()

