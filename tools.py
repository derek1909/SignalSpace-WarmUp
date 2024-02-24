import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

class SparseCodingModel(nn.Module):
    # input_dim = pixels in an image patch = 5*5
    # code_dim = num of basis functions
    def __init__(self, input_dim, code_dim):
        super(SparseCodingModel, self).__init__()
        self.dictionary = nn.Parameter(torch.randn(input_dim, code_dim))  # Random basis functions, Learnable
        self.encoder = nn.Linear(input_dim, code_dim, bias=False, device = 'mps')
        self.decoder = nn.Linear(code_dim, input_dim)  # Reconstruction layer

    def forward(self, data):
        sparse_codes = self.encoder(data)
        reconstruction = self.decoder(sparse_codes)

        return sparse_codes, reconstruction
    
    def loss(data, sparse_codes, reconstruction):

        sigma = 1
        lambd = 1
        # data should be flattened before passed into loss 
        recon_loss = ((data - reconstruction)**2).mean()  # Reconstruction loss
        sparsity_penalty = ( np.log(np.square(sparse_codes)/(sigma**2)+1) * lambd ).mean()

        return recon_loss + sparsity_penalty
    

def Image2Patch(image, patch_size=5, plot=False):
    # Unfold the image to extract patches
    patches = image.unfold(1, patch_size, 1).unfold(2, patch_size, 1)

    # Reshape to get the patches in the desired shape
    patches = patches.contiguous().view(-1, 1, patch_size, patch_size)

    if plot:
        print('patches.shape = ', patches.shape)  # Should print: torch.Size([576, 1, 5, 5])

        col = 28-patch_size+1
        figure = plt.figure(figsize=(col,col))
        for i in range(1, col * col + 1):
            figure.add_subplot(col, col, i)
            plt.axis("off")
            plt.imshow(patches[i-1].squeeze(), cmap="gray")

        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.show()

    return patches
   