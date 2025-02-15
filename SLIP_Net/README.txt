# Uncertainty estimation

cd /home/khanm/workfolder/SLIP_Net/
conda activate transmorph

# create paired pkl files
python /home/khanm/workfolder/SLIP_Net/extra_tools/pklMakerPair.py --image_dir /home/khanm/workfolder/SLIP_Net/Test_data/imgs --mask_dir /home/khanm/workfolder/SLIP_Net/Test_data/masks --output_dir /home/khanm/workfolder/SLIP_Net/Test_data/pkl_pair

# run the uncertainty estimation inference
python infer_TransMorph_Bayes.py



#############################################
Description of folders:
imgs: Bladder MRI scans
masks: Binary masks for bladder
pkl_pair: pkl files composed of channel-wise concatenated consecutive MRI scans and their respective masks
img_grid: A grid of images that shows a visual comparison between SLIP-Net and Monte Carlo dropouts
overlaid: Uncertainty maps overlaid upon their corresponding MRIs
uncertainty: Normalized uncertainty maps between 0 and 1
un_uncertainty: Predicted raw uncertainty maps


