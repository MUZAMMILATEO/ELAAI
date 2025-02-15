from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.SLIP_Net import CONFIGS as CONFIGS_TM
import models.SLIP_Net as SLIP_Net
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn.functional as F

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    batch_size = 5 #8
    train_dir = '/home/khanm/workfolder/SLIP_Net/Train_data/train/'
    val_dir = '/home/khanm/workfolder/SLIP_Net/Train_data/val/'
    weights = [1, 1, 1, 1] # loss weights
    save_dir = 'SLIP_ssim_{}_diffusion_{}/'.format(weights[0], weights[1])
    if not os.path.exists('experiments/'+save_dir):
        os.makedirs('experiments/'+save_dir)
    if not os.path.exists('logs/'+save_dir):
        os.makedirs('logs/'+save_dir)
    sys.stdout = Logger('logs/'+save_dir)
    lr = 0.0001 # learning rate
    epoch_start = 0
    max_epoch = 1 # 600 #200 #100 #400 #max traning epoch
    cont_training = False #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['SLIPNet']
    model = SLIP_Net.SLIPNet(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 201
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.RandomFlip([2]),
                                         trans.NumpyType((np.float32, np.float32)),
                                         ])
    train_set = datasets.RaFDDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    val_set = datasets.RaFDInferDataset(glob.glob(val_dir + '*.pkl'), transforms=None)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=50, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.SSIM_loss(False)
    ssim = SSIM(data_range=255, size_average=True, channel=1)
    criterions = [criterion]
    criterions += [losses.Grad('l2')]
    criterions += [losses.UncertaintyLoss()]
    criterions += [losses.UncertaintyLoss()]
    ########## criterions += [losses.KLDivergenceLoss()]                  # Nieuwe verliezen toegevoegd
    ########## criterions += [losses.MutualInformation()]                 # Nieuwe verliezen toegevoegd
    
    # *************** Add the loss functions to deal with mu, logvar, muvar *****************
    
    best_ncc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            print(f"length of data::: {len(data)}")
            x      = data[0]
            y      = data[1]
            x_mask = data[2]
            y_mask = data[3]
            
            print(f"shape of x::: {x.shape}")
            print(f"shape of y::: {y.shape}")
            print(f"shape of x_mask::: {x_mask.shape}")
            print(f"shape of y_mask::: {y_mask.shape}")
            
            # Reshape x and y tensors to size (256, 256)
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
            y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=True)
            
            x_in = torch.cat((x,y), dim=1)
            
            print(f"x: {x.shape} \ny: {y.shape} \nx_in: {x_in.shape}")
            
            output0, output1, uncertainty_angle, uncertainty_magnitude, uncertainty_angle_pred, uncertainty_magnitude_pred = model(x_in)
            
            ################## MK UPDATE ###################

            # Display the image
            # print(f"SHAPE OF X AND Y: {x.shape} and {y.shape}")
            
            # print("output0 shape: {}".format(output0.shape))
            # k = x[0][0,:,:]
            # k = k/k.max()
            # print("output0 new shape: {}".format(k.shape))
            # k = torch.unsqueeze(k, dim=2)
            # k = torch.cat((k,k,k), dim=2)
            # print("output0 new cat shape: {}".format(k.shape))
            # k = k.detach().cpu()
            # plt.imshow(k.numpy(), cmap='gray')
            # plt.show()
            # print(this_variable)
            ################## ## ###### ###################
            
            # MK UPDATE ------------------------------------
            
            output = [output0, output1, [uncertainty_angle, uncertainty_angle_pred], [uncertainty_magnitude, uncertainty_magnitude_pred]]
            print("The length of the model output is: {}".format(len(output)))
            
            # ## ###### -----------------------------------
            
            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
            
            # MK UPDATE -----------------------------------
                print(f"LOSS FUNCTION::: {loss_function}")
                curr_loss = loss_function(output[n], y) * weights[n]                        # The loop works in a way as i^th output goes to i^th loff function
                print(f"CURRENT LOSS::: {curr_loss}")
                
            # ## ###### -----------------------------------
            
                #if n == 0:
                #    curr_loss = loss_function(output[n], y) * weights[n]
                #else:
                #    curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            output0, output1, uncertainty_angle, uncertainty_magnitude, uncertainty_angle_pred, uncertainty_magnitude_pred = model(x_in)
            
            # MK UPDATE ------------------------------------
            
            output = [output0, output1, [uncertainty_angle, uncertainty_angle_pred], [uncertainty_magnitude, uncertainty_magnitude_pred]]
            print("The length of the model output is: {}".format(len(output)))
            
            # ## ###### -----------------------------------
            for n, loss_function in enumerate(criterions):
            
                # MK UPDATE -----------------------------------
                
                print(loss_function)
                curr_loss = loss_function(output[n], y) * weights[n]                        # The loop works in a way as i^th output goes to i^th loff function
                print(curr_loss)
                
                # ## ###### -----------------------------------
                #if n == 0:
                #    curr_loss = loss_function(output[n], x) * weights[n]
                #else:
                #    curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item() / 2, loss_vals[1].item() / 2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_ncc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_mask = data[2]
                y_mask = data[3]
                
                # Reshape x and y tensors to size (256, 256)
                x     = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
                y     = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=True)
                x_mask = F.interpolate(x_mask, size=(256, 256), mode='bilinear', align_corners=True)
                y_mask = F.interpolate(y_mask, size=(256, 256), mode='bilinear', align_corners=True)
                
                # x = x * x_mask
                # y = y * y_mask
                
                ################## MK UPDATE ###################

                # Display the image
                # print(f"SHAPE OF X AND Y: {x.shape} and {y.shape}")
                
                # k = y[0][0,:,:]
                # k = k/k.max()
                # print("output0 new shape: {}".format(k.shape))
                # k = torch.unsqueeze(k, dim=2)
                # k = torch.cat((k,k,k), dim=2)
                # print("output0 new cat shape: {}".format(k.shape))
                # k = k.detach().cpu()
                # plt.imshow(k.numpy(), cmap='gray')
                # plt.show()
                # print(this_variable)
                ################## ## ###### ###################
                
                # MK UPDATE --------------------------------------
                print(f"---- shape of x: {x.shape}")
                
                x_maskUse = torch.unsqueeze(x_mask, axis = 1)
                x_maskUse = torch.cat((x_maskUse, x_maskUse, x_maskUse), axis = 1)
                
                y_maskUse = torch.unsqueeze(y_mask, axis = 1)
                y_maskUse = torch.cat((y_maskUse, y_maskUse, y_maskUse), axis = 1)
                
                # ## ###### --------------------------------------
                
                print(f"VALIDATON:\nx:{x.shape},y: {y.shape}, x_mask: {x_mask.shape}, y_mask: {y_mask.shape}")

                x_in = torch.cat((y, x), dim=1)
                output = model(x_in)
                ncc = ssim(output[0], x)
                eval_ncc.update(ncc.item(), x.numel())

                #flip image
                x_in = torch.cat((x, y), dim=1)
                output = model(x_in)
                ncc = ssim(output[0], y)
                eval_ncc.update(ncc.item(), y.numel())
                
                print("The length of the model output is:    {}".format(len(output)))
                print("The length of the model output[0] is: {}".format(output[0].shape))
                print("The length of the model output[1] is: {}".format(output[1].shape))

                grid_img = mk_grid_img(8, 1, (x.shape[0], config.img_size[0], config.img_size[1]))
                print(f"---grid image shape: {grid_img.shape}")
                def_out = []
                
                for idx in range(3):
                    print(f"idx is {idx}")
                    x_def = reg_model_bilin([x_maskUse[..., idx].cuda().float(), output[1].cuda()])
                    def_out.append(x_def[..., None])
                def_out = torch.cat(def_out, dim=-1)
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])

                print(eval_ncc.avg)
        best_ncc = max(eval_ncc.avg, best_ncc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_ncc': best_ncc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/'+save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_ncc.avg))
        writer.add_scalar('DSC/validate', eval_ncc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        # x_fig = comput_fig(x_rgb)
        # tar_fig = comput_fig(y_rgb)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        # writer.add_figure('input', x_fig, epoch)
        # plt.close(x_fig)
        # writer.add_figure('ground truth', tar_fig, epoch)
        # plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0:16, 0, :, :]
    if img.shape[-1] == 3:
        img = img.astype(np.uint8)[...,  ::-1]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(64, 256, 256)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[:, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
