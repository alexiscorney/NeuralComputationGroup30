#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn 

import h5py, os
import numpy as np
from matplotlib import pyplot as plt
from functions import transforms as T
from torch.nn import functional as F
from functions.subsample import MaskFunc
from scipy.io import loadmat
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from math import exp


# In[2]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # check whether a GPU is available


# In[3]:


def show_slices(data, slice_nums, cmap=None): # visualisation
    fig = plt.figure(figsize=(15,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap)
        plt.axis('off')


# In[4]:


class MRIDataset(DataLoader):
    def __init__(self, data_list, acceleration, center_fraction, use_seed):
        self.data_list = data_list
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.use_seed = use_seed

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subject_id = self.data_list[idx]
        return get_epoch_batch(subject_id, self.acceleration, self.center_fraction, self.use_seed)


# In[5]:


def get_epoch_batch(subject_id, acc, center_fract, use_seed=True):
    ''' random select a few slices (batch_size) from each volume'''

    fname, rawdata_name, slice = subject_id  
    
    with h5py.File(rawdata_name, 'r') as data:
        rawdata = data['kspace'][slice]
                      
    slice_kspace = T.to_tensor(rawdata).unsqueeze(0)
    S, Ny, Nx, ps = slice_kspace.shape

    # apply random mask
    shape = np.array(slice_kspace.shape)
    # mask_func = MaskFunc(center_fractions=[center_fract], accelerations=[acc])
    mask_func = MaskFunc(center_fractions=center_fract, accelerations=acc)
    seed = None if not use_seed else tuple(map(ord, fname))
    mask = mask_func(shape, seed)
      
    # undersample
    masked_kspace = torch.where(mask == 0, torch.Tensor([0]), slice_kspace)
    masks = mask.repeat(S, Ny, 1, ps)

    img_gt, img_und = T.ifft2(slice_kspace), T.ifft2(masked_kspace)

    # perform data normalization which is important for network to learn useful features
    # during inference there is no ground truth image so use the zero-filled recon to normalize
    norm = T.complex_abs(img_und).max()
    if norm < 1e-6: norm = 1e-6
    
    # normalized data
    img_gt, img_und, rawdata_und = img_gt/norm, img_und/norm, masked_kspace/norm
        
    return img_gt.squeeze(0), img_und.squeeze(0), rawdata_und.squeeze(0), masks.squeeze(0), norm


# In[6]:


def load_data_path(train_data_path, val_data_path):
    """ Go through each subset (training, validation) and list all 
    the file names, the file paths and the slices of subjects in the training and validation sets 
    """

    data_list = {}
    train_and_val = ['train', 'val']
    data_path = [train_data_path, val_data_path]
      
    for i in range(len(data_path)):

        data_list[train_and_val[i]] = []
        
        which_data_path = data_path[i]
    
        for fname in sorted(os.listdir(which_data_path)):
            
            subject_data_path = os.path.join(which_data_path, fname)
                     
            if not os.path.isfile(subject_data_path): continue 
            
            with h5py.File(subject_data_path, 'r') as data:
                num_slice = data['kspace'].shape[0]
                
            # the first 5 slices are mostly noise so it is better to exlude them
            data_list[train_and_val[i]] += [(fname, subject_data_path, slice) for slice in range(5, num_slice)]
    
    return data_list  


# In[7]:


#PREPARE THE DATA 
data_list = load_data_path('/data/local/NC2019MRI/train', '/data/local/NC2019MRI/train')
# slices, height, width = input_k.shape()

acc = [4,8]
cen_fract = [0.08, 0.04]
# acc = [4]
# cen_fract = [0.08]
seed = False # random masks for each slice 
num_workers = 16 # data loading is faster using a bigger number for num_workers. 0 means using one cpu to load data

def my_collate(batch):
    batch_len = len(batch)
    data = torch.ones(batch_len, 1, 320, 320)
    target_list = torch.ones(batch_len, 1, 320, 320)
    
    for batch_value in range(len(batch)):
        input = batch[batch_value][1]
        input = T.complex_abs(input)
        input = T.center_crop(input, (320, 320))
        data[batch_value, 0, :, :] = input
        
        target = batch[batch_value][0]
        target = T.complex_abs(target)
        target = T.center_crop(target, (320, 320))
        target_list[batch_value, 0, :, :] = target
    return (target_list, data, None, None, None)
    
# create data loader for training set. It applies same to validation set as well
train_dataset = MRIDataset(data_list['train'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=14, num_workers=num_workers, collate_fn=my_collate)



# In[ ]:





# In[8]:


#build the network

#fully connected input layer (64)

#--

#convolutional layers (relu)

#fully connected output layer (64)


# In[9]:


class ConvolutionalBlock(nn.Module):
    """
    2 lots of:
        3x3 convolution
        ReLu  
    """

    def __init__(self, in_chans, out_chans):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, padding_mode='zeros', bias=False),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        #print('input', input.shape)
        return self.layers(input)


# In[10]:


class UNet(nn.Module):
    """
        Unet model
        
        Encoder:
            input -> Convolutional Block (output 1)
            
            maxpool(output 1) -> Convolutional Block (output 2)
            maxpool(output 2) -> Convolutional Block (output 3)
                        
            concat(output 2, deconv(output 3)) -> Convolutional Block (output 4)
            
            concat(output 1, deconv(output 4)) -> Convolutional Block (output 5)
            
            output5 -> 1x1 Convolution (output 6)
            
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        
        '''ENCODER'''
        self.encoder1 = ConvolutionalBlock(in_chans, chans) # 1 -> 32
        
        self.encoder2 = ConvolutionalBlock(chans, chans) # 3 -> 32
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #3
        
        self.encoder3 = ConvolutionalBlock(chans, chans * 2) #32 -> 64
        
        self.encoder4 = ConvolutionalBlock(chans * 2, chans * 2) # 64 -> 64
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder5 = ConvolutionalBlock(chans * 2, chans * 4) # 64 -> 128 
        
        self.encoder6 = ConvolutionalBlock (chans * 4, chans * 4) # 128 -> 128 
        
        
        
        '''DECODER'''
        
        self.deconv1 = nn.ConvTranspose2d(chans * 4, chans * 2, kernel_size=2, stride=2) # 128 -> 64
        
        #concatenate to give 128 
        
        self.decoder1 = ConvolutionalBlock(chans * 4, chans * 2) # 128 -> 64 
        
        self.decoder2 = ConvolutionalBlock(chans * 2, chans * 2) # 64 -> 64
        
        self.deconv2 = nn.ConvTranspose2d(chans * 2, chans, kernel_size=2, stride=2 ) # 64 -> 32 
        
        #concatenate to give 64 
                                          
        self.decoder3 = ConvolutionalBlock(chans * 2, chans) # 64 -> 32
        
        self.decoder4 = ConvolutionalBlock(chans, chans) # 32 -> 32
        
        
        '''FINAL CONV'''
        
        self.finalConv = nn.Conv2d(in_channels=chans, out_channels=out_chans, kernel_size=1)
        
       
        ''''''
        
        

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        
        '''APPLY THE LAYERS'''
        #print('input shape: ')
        #print(input.shape)
        out1 = self.encoder1(input) #32
        #print('out1 shape: ')
        #print(out1.shape)
        out2 = self.encoder2(out1) #32
        #print('out2 shape: ')
        #print(out2.shape)
        #max pool 
        pool2 = self.pool1(out2)
        #print('pool2 shape: ')
        #print(pool2.shape)
        
        out3 = self.encoder3(pool2) #64
        #print('out3 shape: ')
        #print(out3.shape)
        out4 = self.encoder4(out3) #64
        #print('out4 shape: ')
        #print(out4.shape)
        #max pool
        pool4 = self.pool2(out4)
        #print('pool4 shape: ')
        #print(pool4.shape)
        
        out5 = self.encoder5(pool4) #128
        #print('out5 shape: ')
        #print(out5.shape)
        out6 = self.encoder6(out5) #128 
        #print('out6 shape: ')
        #print(out6.shape)
        
        '''DECODE'''
        out7 = self.deconv1(out6) #64
        #print('out7 shape: ')
        #print(out7.shape)
        
        #concatenate 
        #scaled1 = F.interpolate(out4, scale_factor=1.98125, mode='bilinear', align_corners=False)
        out8 = torch.cat((out7, out4), dim=1) #128
        #print('out8 concat shape: ')
        #print(out8.shape)
        
        out9 = self.decoder1(out8) #64
        #print('out9 shape: ')
        #print(out9.shape)
        out10 = self.decoder2(out9) #64
        #print('out10 shape: ')
        #print(out10.shape)
        
        decon11 = self.deconv2(out10) #32
        #print('decon11 shape: ')
        #print(decon11.shape)
        
        #concatenate 
        scaled2 = F.interpolate(out2, scale_factor=1.990625, mode='bilinear', align_corners=False)
        out11 = torch.cat((decon11, out2), dim=1) #64
        #print('out11 concat shape: ')
        #print(out11.shape)
        
        out12 = self.decoder3(out11) #32
        #print('out12 shape: ')
        #print(out12.shape)
        out13 = self.decoder4(out12) #32
        #print('out13 shape: ')
        #print(out13.shape)
        
        output = self.finalConv(out13)
        #print('output shape: ')
        #print(output.shape)
        
        return output 


# In[11]:


import torch.optim as optim
torch.manual_seed(42)

#create a model
model = UNet(
    in_chans=1,
    out_chans=1,
    chans=32,
    num_pool_layers=4,
).to(device)

#inspect parameters 
# print("Before training: \n", model.state_dict())


# In[12]:


#ssim loss
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.type == img1.type:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1 - ssim(img2, img1, window=window, window_size=self.window_size, size_average=self.size_average, val_range=img2.max())


# In[13]:


# set learning rate
lr = 1e-4
wd = 0.0
#optimiser
#stochastic gradient descent (SGD)
# optimiser = optim.SGD(model.parameters(), lr=lr)
optimiser = optim.Adam(model.parameters(), lr=lr)
# optimiser = torch.optim.RMSprop(model.parameters(), lr, weight_decay=wd)


# In[14]:


ssim_loss = SSIM()


# In[15]:


def save_model(path, loss):
    full_path = path + "-loss-" str(loss) + '.h5'
    torch.save(model.state_dict(), full_path)


# In[ ]:


#train the network 

# set number of epoches, i.e., number of times we iterate through the training set
epoches = 2000


for epoch in range(epoches):
    model.train() 
    mean = []
    for iter, data in enumerate(train_loader):
        target, input, _, _, _ = data
        # print(input.shape)
        input = input.to(device)
        # print(input.shape)
        # print(target.shape)
        target = target.to(device)

        output = model(input)
        # print(output.shape)
        loss = ssim_loss(output, target)
        # loss = msssim_loss(output, target)
        # loss = F.l1_loss(output, target)
        mean.append(loss)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    l = sum(mean)/len(mean)
    print("Epoch {}'s loss: {}".format(epoch, l))
    if(epoch % 10 == 0):
        save_model('models/model', l)

# print("After training: \n", model.state_dict())


# In[ ]:


PATH = 'model_last.h5'
torch.save(model.state_dict(), PATH)


#     

# In[ ]:


model = UNet(
    in_chans=1,
    out_chans=1,
    chans=32,
    num_pool_layers=4,
).to(device)

model.load_state_dict(torch.load(PATH))


# In[ ]:


model.eval()


# In[ ]:


for iteration, sample in enumerate(train_loader):
    
    img_gt, img_und, _, _, _  = sample
    D = img_gt.squeeze()
    
    C = img_und
    input = C.to(device)
    # print(img_gt.shape) torch.Size([14, 1, 320, 320])
    
    output = model(input).cpu().detach().numpy().squeeze()
    
    # print(output.shape) (14, 320, 320)
    # print(D.shape) torch.Size([14, 320, 320])
    #print(ssim_np(D.cpu().detach().numpy(), output))
    # from left to right: mask, masked kspace, undersampled image, ground truth
    show_slices([D[0], output[0], C.squeeze()[0]], [0, 1, 2], cmap='gray')
    print(model(input).squeeze(1).unsqueeze(0).shape)
    print(img_gt.squeeze(1).unsqueeze(0).to(device).shape)
    print(1 - ssim_loss(model(input).squeeze(1).unsqueeze(0), img_gt.squeeze(1).unsqueeze(0).to(device)))
    if iteration < 1: break


# In[ ]:


# create data loader for training set. It applies same to validation set as well
test_dataset = MRIDataset(data_list['val'], acceleration=acc, center_fraction=cen_fract, use_seed=seed)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=1, num_workers=num_workers)


# In[ ]:


from skimage.measure import compare_ssim 
def ssim_old(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


# In[ ]:


for iter, data in enumerate(test_loader):
    target, input, mean, std, norm = data
    mg_gt, img_und, rawdata_und, masks, norm = sample
    D = T.complex_abs(img_gt).squeeze()
    D = T.center_crop(D, (320, 320))
    
    C = T.complex_abs(img_und)
    C = T.center_crop(C, (320, 320))
    input = C.unsqueeze(1).to(device)
    
    output = model(input).cpu().detach().numpy().squeeze(1)
    
    print(C.shape)
    print(output.shape)
    print(ssim(C.cpu().detach().numpy(), output))
    
    if iteration < 1: break


# In[ ]:





# In[ ]:





# In[ ]:




