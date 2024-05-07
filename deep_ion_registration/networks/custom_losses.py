import torch

def mask_loss2D(output, groundtruth, mask):
    loss = torch.mean((output[mask]-groundtruth[mask])**2)
    return loss

def central_slice_mask_loss_recon(recon_output, output, groundtruth, mask):
    #print(output.size(), groundtruth.size())
    nSliceBlock = output.size(dim=2)
    s = int((nSliceBlock-1)/2)
    output_c = output[:,:,s,:,:]
    groundtruth_c = groundtruth[:,:,s,:,:]
    recon_output_c = recon_output[:,:,s,:,:]
    mask_c = mask[:,:,s,:,:]
    
    loss = torch.mean((recon_output_c[mask_c]-groundtruth_c[mask_c])**2)+ torch.mean((output_c[mask_c]-groundtruth_c[mask_c])**2)
    return loss

def central_slice_loss(output, groundtruth):
    #print(output.size(), groundtruth.size())
    nSliceBlock = output.size(dim=2)
    s = int((nSliceBlock-1)/2)
    
    loss = torch.mean((output[:,:,s,:,:]-groundtruth[:,:,s,:,:])**2)
    return loss
