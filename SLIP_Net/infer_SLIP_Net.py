import glob
import os
import losses
import utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import models.configs_SLIP_Net as configs  # Ensure correct import
import models.SLIP_Net as SLIP_Net
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch.nn.functional as F
import cv2
import matplotlib.colors as mcolors
import flow_vis
import time  # Import the time module
import scipy.ndimage as ndimage
from skimage.morphology import closing, disk
from skimage.filters import gaussian
import torchvision.transforms.functional as TF
import lpips

def main():
    test_dir = '/home/khanm/workfolder/SLIP_Net/Test_data/pkl_pair/'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'SLIP_ssim_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder

    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_folder[:-1]+'.csv'):
        os.remove('Quantitative_Results/'+model_folder[:-1]+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + model_folder[:-1])
    line = ',SSIM,det'
    csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])

    config = configs.get_2DSLIPNet_config()  # Use the correct config function
    
    # Debugging: print the configuration to ensure all expected attributes are present
    print("Configuration being used:")
    for key, value in config.items():
        print(f"{key}: {value}")

    model = SLIP_Net.SLIPNet(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    test_set = datasets.RaFDInferDataset(natsorted(glob.glob(test_dir + '*.pkl')), transforms=None)
    # test_set = datasets.RaFDInferDataset(glob.glob(test_dir + '*.pkl'), transforms=None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    ssim = SSIM(data_range=255, size_average=True, channel=1)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        a = True
        for data, data_filename in zip(test_loader, natsorted(glob.glob(test_dir + '*.pkl'))):
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_mask = data[2]
            y_mask = data[3]

            # Reshape x and y tensors to size (256, 256)
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
            y = F.interpolate(y, size=(256, 256), mode='bilinear', align_corners=True)
            x_mask = F.interpolate(x_mask, size=(256, 256), mode='bilinear', align_corners=True)
            y_mask = F.interpolate(y_mask, size=(256, 256), mode='bilinear', align_corners=True)
            
            x_disp = x
            y_disp = y
            
            x = x * x_mask
            y = y * y_mask
            
            x_in = torch.cat((y, x), dim=1)
            
            # Record the start time
            start_time = time.time()
            
            output = model(x_in, disp=y_disp ,mc_dropout=True, test=True)  # Ensure mc_dropout and test are True
            
            # Record the end time
            end_time = time.time()
            
            # Calculate the elapsed time
            elapsed_time = end_time - start_time
            print(f"Time taken for model inference: {elapsed_time:.4f} seconds")
            
            # Print the shapes of all six elements in the output list
            for i in range(6):
                print(f'Shape of output[{i}]: {output[i].shape}')
            
            # Extract filename without extension
            base_filename = os.path.splitext(os.path.basename(data_filename))[0]
            
            # Display the outputs including the flow field as a color map
            
            if a == True:
                # Initialize the LPIPS model
                lpips_model = lpips.LPIPS(net='alex').cuda()
                a = False
                
            visualize_outputs(y_disp, x, output, y_disp, y_mask, x_mask, x_disp, base_filename, lpips_model)
            
            ncc = ssim(y, x)
            eval_dsc_raw.update(ncc.item(), x.numel())
            ncc = ssim(output[0], x)
            eval_dsc_def.update(ncc.item(), x.numel())
            jac_det = utils.jacobian_determinant_vxm(output[1].detach().cpu().numpy()[0, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(x.shape), x.numel())
            line = 'p{}'.format(stdy_idx) + ',' + str(ncc.item()) + ',' + str(np.sum(jac_det <= 0) / np.prod(x.shape))
            csv_writter(line, 'Quantitative_Results/' + model_folder[:-1])
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def visualize_outputs(x_disp, y, outputs, fixed_image, x_mask, y_mask, y_disp, filename, lpips_model):
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))

    titles = ['(a)', '(b)', '(c)', '(d)',
              '(f)', '(g)', '(h)', '(i)']
    
    # Create a custom colormap
    cdict = {
        'red':   [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
    }
    custom_cmap = 'magma' #mcolors.LinearSegmentedColormap('WhiteRed', cdict)
    
    ssim_map = compute_ssim_map(x_disp, y_disp, size_average=False)
    
    # Assuming ssim_map is the SSIM map computed by compute_ssim_map function
    ssim_map_min = ssim_map.min()
    ssim_map_max = ssim_map.max()

    ssim_map = 1 - ((ssim_map - ssim_map_min) / (ssim_map_max - ssim_map_min))
    
    combined_flow = torch.exp(((torch.abs(outputs[4]) + torch.abs(outputs[5]))*100))
    combined_flow = TF.gaussian_blur(combined_flow, kernel_size=5, sigma=1.0)
            
    # Find the min and max values of the combined flow
    min_val = combined_flow.min()
    max_val = combined_flow.max()
            
    print(f"min_val: {min_val} and max_val: {max_val}")
            
    # Normalize combined flow to the range 0-1
    normalized_flow = (combined_flow - min_val) / (max_val - min_val)
            
    # Scale normalized flow to the range 0-10
    scaled_flow = normalized_flow * 1
    
    ###############################################################################
    flow_variance_maksed = outputs[2] * torch.logical_or(x_mask, y_mask) * ssim_map
    # flow_variance = outputs[i].detach().cpu().numpy()[0]
    flow_variance = flow_variance_maksed.detach().cpu().numpy()[0]
    flow_variance_magnitude = np.exp(np.sqrt(np.sum(flow_variance**2, axis=0)))

    def compute_lpips_map(image1, image2, model, patch_size=32, stride=16):
        _, _, h, w = image1.shape

        lpips_map = np.zeros((h, w))

        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch1 = image1[:, :, i:i+patch_size, j:j+patch_size].cuda()
                patch2 = image2[:, :, i:i+patch_size, j:j+patch_size].cuda()
                
                # Resize patches to 32x32 for LPIPS evaluation
                patch1_resized = F.interpolate(patch1, size=(32, 32), mode='bilinear', align_corners=False)
                patch2_resized = F.interpolate(patch2, size=(32, 32), mode='bilinear', align_corners=False)
                
                with torch.no_grad():
                    lpips_value = model(patch1_resized, patch2_resized).item()
                lpips_map[i:i+patch_size, j:j+patch_size] = lpips_value

        return lpips_map

    def normalize_tensor(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def fuse_uncertainties(mc_uncertainty, model_uncertainty, fixed_image, moving_image):
        # Compute LPIPS score map
        lpips_map = compute_lpips_map(fixed_image, moving_image, lpips_model)
    
        # Convert LPIPS map to tensor and move to GPU
        lpips_map_tensor = torch.tensor(lpips_map).float().cuda()
    
        # Normalize the LPIPS map
        lpips_map_normalized = normalize_tensor(lpips_map_tensor)
    
        # Inverse LPIPS map for weights
        inverse_lpips_map = 1 - lpips_map_normalized
    
        # Normalize uncertainty maps
        mc_uncertainty_normalized = normalize_tensor(np.log(mc_uncertainty))
        model_uncertainty_normalized = normalize_tensor(model_uncertainty)
    
        # Apply Gaussian blur to MC uncertainty
        # mc_uncertainty_smoothed = F.gaussian_blur(mc_uncertainty_normalized.unsqueeze(0).unsqueeze(0), kernel_size=(5, 5), sigma=(2, 2)).squeeze(0).squeeze(0)
        
        # Ensure all tensors are on the same device
        lpips_map_normalized = lpips_map_normalized.cuda()
        inverse_lpips_map = inverse_lpips_map.cuda()
        # mc_uncertainty_smoothed = mc_uncertainty_smoothed.cuda()
        model_uncertainty_normalized = model_uncertainty_normalized.cuda()
    
        # Compute the combined uncertainty map
        combined_uncertainty = lpips_map_normalized * model_uncertainty_normalized + inverse_lpips_map * model_uncertainty_normalized
    
        return combined_uncertainty, lpips_map_normalized
        
    weighted_uncertainty, lpips_map_normalized = fuse_uncertainties(flow_variance_magnitude, scaled_flow, x_disp, y_disp)
    
    weighted_uncertainty = weighted_uncertainty.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()   
    print(f"weighted_uncertainty.shape: {weighted_uncertainty.shape}")
    print(f"flow_variance_magnitude.shape: {flow_variance_magnitude.shape}")
            

    def flow_to_quiver(ax, flow, step=10):
        """
        Converts flow field to a quiver plot.
    
        Parameters:
        - ax: The axis to plot on.
        - flow: The flow field (2xHxW tensor).
        - step: The step size to reduce the number of arrows.
        """
        flow = flow.detach().cpu().numpy()
        u = flow[0, :, :]
        v = flow[1, :, :]
    
        # Create a grid of coordinates
        x = np.arange(0, u.shape[1], step)
        y = np.arange(0, u.shape[0], step)
        x, y = np.meshgrid(x, y)
    
        u = u[::step, ::step]
        v = v[::step, ::step]
    
        ax.quiver(x, y, u, v, color='r')

    for i in range(8):
        ax = axs[i // 4, i % 4]
        if i == 1:  # For flow field
            flow = outputs[i].detach().cpu().numpy()[0]
            flow_rgb = flow_to_color(flow)
            # flow_to_quiver(ax, outputs[i][0])
            ax.imshow(flow_rgb)
        elif i == 2:  # For flow variance
            # flow_variance_maksed = outputs[i] * torch.logical_or(x_mask, y_mask) * ssim_map
            # flow_variance = outputs[i].detach().cpu().numpy()[0]
            # flow_variance = flow_variance_maksed.detach().cpu().numpy()[0]
            # flow_variance_magnitude = np.sqrt(np.sum(flow_variance**2, axis=0))
            ax.imshow(flow_variance_magnitude)
            #overlay_image(ax, fixed_image, flow_variance, custom_cmap)
            
            varVal = compute_masked_variance(flow_variance_magnitude,x_mask)
            print(f"The variance in the distribution is: {varVal}")
            
        elif i == 4:  # For image variance
            lpips_map_normalized = lpips_map_normalized.unsqueeze(dim=0).unsqueeze(dim=0)
            # y_disp
            ax.imshow(y_disp.detach().cpu().numpy()[0, 0], cmap='gray')
            # image_variance_masked = outputs[i] * x_mask
            # image_variance = outputs[i].detach().cpu().numpy()[0, 0]
            # image_variance = image_variance_masked.detach().cpu().numpy()[0, 0]
            # overlay_image(ax, fixed_image, image_variance, custom_cmap)
            
        elif i == 0:  # For image variance
            ax.imshow(x_disp.detach().cpu().numpy()[0, 0], cmap='gray')
        elif i == 5:  # For image variance 
            ax.imshow(ssim_map.detach().cpu().numpy()[0, 0], cmap='gray')
        elif i == 6:
        
            flow_variance_maksed = (scaled_flow) * torch.logical_or(x_mask, y_mask) * ssim_map
            flow_variance = flow_variance_maksed.detach().cpu().numpy()[0]
            flow_variance_magnitude = np.sqrt(np.sum(flow_variance**2, axis=0))
            ax.imshow(np.exp(flow_variance_magnitude))
            
            varVal = compute_masked_variance(flow_variance_magnitude, x_mask)
            print(f"The variance in the distribution is: {varVal}")
        
        elif i == 7:
            
            flow_variance_maksed = scaled_flow * x_mask * ssim_map #scaled_flow * torch.logical_or(x_mask, y_mask) * ssim_map
            # flow_variance = outputs[i].detach().cpu().numpy()[0]
            flow_variance = flow_variance_maksed.detach().cpu().numpy()[0]
            flow_variance_magnitude = np.exp(np.sqrt(np.sum(flow_variance**2, axis=0)))
            
            # Compute the threshold
            threshold = 0.25 * np.max(flow_variance_magnitude)

            # Set pixels with uncertainty less than the threshold to zero
            flow_variance_magnitude[flow_variance_magnitude < threshold] = 0
            
            # Label connected regions
            labeled_array, num_features = ndimage.label(flow_variance_magnitude > 0)

            # Define the minimum size for regions to keep
            min_size = 20  # Adjust this value based on your needs

            # Remove small regions
            for region in range(1, num_features + 1):
                if np.sum(labeled_array == region) < min_size:
                    flow_variance_magnitude[labeled_array == region] = 0
                    
            # Apply morphological closing to smooth the shape
            selem = disk(2)  # Structuring element, adjust the size as needed
            closed_map = closing(flow_variance_magnitude > 0, selem)

            # Apply the closed map to the original flow variance magnitude
            flow_variance_magnitude[closed_map == 0] = 0

            # Apply Gaussian filter to further smooth the boundaries
            flow_variance_magnitude = gaussian(flow_variance_magnitude, sigma=1)
            
            print(f"Maximum of flow_variance_magnitude is {(flow_variance_magnitude - flow_variance_magnitude.min()).max()}")
            cv2.imwrite(f'/home/khanm/workfolder/SLIP_Net/Test_data/un_uncertainty/{filename}.png', (np.abs(flow_variance_magnitude - flow_variance_magnitude.min())*200).astype(np.uint8))
            
            # Normalize flow_variance_magnitude to the range [0, 1]
            flow_variance_magnitude = flow_variance_magnitude - flow_variance_magnitude.min()
            flow_variance_magnitude = flow_variance_magnitude / flow_variance_magnitude.max()
            
            flow_variance_magnitude_uint8 = (flow_variance_magnitude * 255).astype(np.uint8)
            cv2.imwrite(f'/home/khanm/workfolder/SLIP_Net/Test_data/uncertainty/{filename}.png', flow_variance_magnitude_uint8)
            
            pathname = f'/home/khanm/workfolder/SLIP_Net/Test_data/overlaid/{filename}.png'
            overlay_image(ax, fixed_image, flow_variance_magnitude, custom_cmap, pathname)
            # ax.imshow(outputs[i].detach().cpu().numpy()[0, 0], cmap='gray')
            
        elif i == 3:
        
            x_mask_sq = torch.squeeze(x_mask, axis=0)
            x_mask_sq = torch.squeeze(x_mask_sq, axis=0)
            ssim_map_sq = torch.squeeze(ssim_map, axis=0)
            ssim_map_sq = torch.squeeze(ssim_map_sq, axis=0)
            x_mask_np = x_mask_sq.detach().cpu().numpy()
            ssim_map_np = ssim_map_sq.detach().cpu().numpy()
            
            flow_variance_magnitude = flow_variance_magnitude * x_mask_np * ssim_map_np #scaled_flow * torch.logical_or(x_mask, y_mask) * ssim_map
            
            # Compute the threshold
            threshold = 0.25 * np.max(flow_variance_magnitude)
            
            flow_variance_magnitude_temp = flow_variance_magnitude

            # Set pixels with uncertainty less than the threshold to zero
            flow_variance_magnitude_temp[flow_variance_magnitude_temp < threshold] = 0
            
            # Label connected regions
            labeled_array, num_features = ndimage.label(flow_variance_magnitude_temp > 0)

            # Define the minimum size for regions to keep
            min_size = 20  # Adjust this value based on your needs

            # Remove small regions
            for region in range(1, num_features + 1):
                if np.sum(labeled_array == region) < min_size:
                    flow_variance_magnitude_temp[labeled_array == region] = 0
                    
            # Apply morphological closing to smooth the shape
            selem = disk(2)  # Structuring element, adjust the size as needed
            closed_map = closing(flow_variance_magnitude_temp > 0, selem)

            # Apply the closed map to the original flow variance magnitude
            flow_variance_magnitude_temp[closed_map == 0] = 0

            # Apply Gaussian filter to further smooth the boundaries
            flow_variance_magnitude_temp = gaussian(flow_variance_magnitude_temp, sigma=1)
            
            overlay_image(ax, fixed_image, flow_variance_magnitude_temp, custom_cmap)
            # ax.imshow(outputs[i].detach().cpu().numpy()[0, 0], cmap='gray')
            
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'/home/khanm/workfolder/SLIP_Net/Test_data/img_grid/{filename}_visualization.png')
    # plt.show()

def compute_ssim_map(img1, img2, window_size=11, size_average=False):
    """
    Compute the SSIM map between two input images.
    
    Parameters:
    img1 (torch.Tensor): The first input image.
    img2 (torch.Tensor): The second input image.
    window_size (int): The size of the Gaussian window.
    size_average (bool): If True, return the average SSIM over the image. If False, return the SSIM map.
    
    Returns:
    torch.Tensor: The SSIM map (or average SSIM if size_average is True).
    """
    def gaussian(window_size, sigma):
        gauss = torch.tensor([np.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map

    channel = img1.size(1)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return ssim(img1, img2, window, window_size, channel, size_average)
    

def compute_masked_variance(uncertainty_map, binary_mask):
    """
    Computes the variance of the values in the uncertainty map that correspond
    to the binary mask.
    
    Parameters:
    uncertainty_map (numpy.ndarray): A 2D array representing the uncertainty at each pixel.
    binary_mask (numpy.ndarray): A 2D array where the mask is applied (1 for mask, 0 otherwise).
    
    Returns:
    float: The variance of the masked values.
    """
    binary_mask = binary_mask.detach().cpu().numpy()[0][0]
    
    # Ensure the mask is converted to a boolean array
    binary_mask = binary_mask.astype(bool)
    
    # Extract the values in the uncertainty map that correspond to the mask
    masked_values = uncertainty_map[binary_mask]
    
    # Compute the variance of the masked values
    variance = np.var(masked_values)
    
    return variance


def overlay_image(ax, fixed_image, variance, cmap, pathname=False):
    fixed_image = fixed_image.detach().cpu().numpy()[0, 0]
    ax.imshow(fixed_image, cmap='gray')
    ax.imshow(variance, cmap=cmap, alpha=0.5)
    
    if pathname:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(fixed_image, cmap='gray')
        ax.imshow(variance, cmap=cmap, alpha=0.5)
        ax.axis('off')  # Hide axes for the RGB image
        rgb_filename = pathname
        plt.savefig(rgb_filename, bbox_inches='tight', pad_inches=0, dpi=300)  # High DPI for quality
        plt.close(fig)

def flow_to_color(flow):
    
    flow_permuted = np.transpose(flow, (1, 2, 0))
    print(f"Shape of flow field: {flow_permuted.shape}")

    flow_color = flow_vis.flow_to_color(flow_permuted, convert_to_bgr=False)

    return flow_color

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
