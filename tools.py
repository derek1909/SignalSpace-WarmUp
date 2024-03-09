import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

# Check that MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is enabled!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("cuda is enabled!")


def GenerateDataset(a=2, b=0.7, num_noisy_samples=30, noise_sigma=0.2, PLOT=True):
    # Range of theta values
    t = np.linspace(0, 4 * np.pi, 400)  # From 0 to 5*pi

    # Single curve
    theta = t
    r = a + b * t

    # twin curve
    theta = np.hstack((theta, theta))
    r = np.hstack((r, -r))

    # Generate original curve
    xy = np.array([r * np.cos(theta), r * np.sin(theta)]).T

    # Initialize arrays to hold the noisy data
    xy_noisy = np.tile(xy, [num_noisy_samples, 1]) 

    # Generate noisy data points
    noise = np.random.normal(0, noise_sigma, xy_noisy.shape)
    xy_noisy += noise

    if PLOT:
        # Plotting in polar coordinates
        plt.figure(figsize=[8, 8])

        # Original curve
        ax = plt.subplot(111)
        ax.plot(xy[:,0], xy[:,1], label='Original Curve', color='blue')

        # Noisy data points
        ax = plt.subplot(111)
        ax.scatter(xy_noisy[:, 0], xy_noisy[:, 1], label='Noisy Data', color='red', alpha=0.1)

        # Setting the title and labels
        ax.set_title("Original Curve and Noisy Data Points: r(θ) = a + bθ")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        # Show the plot
        plt.show()

    return xy_noisy


class SparseCoding():
    def __init__(self, data_size, activ_dim, basisfunc_num, sparsity_level):
        super(SparseCoding, self).__init__()
        self.Basis = torch.rand(basisfunc_num, activ_dim, requires_grad=True, device = device)
        with torch.no_grad():
            self.Basis -= 0.5  # This is not right!!!!!!!!
            self.Basis *= 20  # This is not right!!!!!!!!
        self.Activ = torch.randn(data_size, basisfunc_num, requires_grad=False, device = device) #manually update A
        self.sparsity_level = sparsity_level

    def loss(self,data):
        reconstruction = self.Activ @ self.Basis

        # Compute the squared differences
        squared_error= (reconstruction - data) ** 2

        # Compute the sum of squared differences
        sum_squared_error = torch.sum(squared_error)

        # Compute the number of samples
        num_samples = data.shape[0]

        return sum_squared_error/num_samples
    
    def activ_panalty(self):
        return self.sparsity_level * torch.norm(abs(self.Activ), p=0.5) 

    def plot_basis(self):
        # Plotting in polar coordinates
        plt.figure(figsize=[6, 6])

        # Detach the tensor before converting to numpy
        basis_np = self.Basis.detach().cpu().numpy()

        # plot dictionary elements
        ax = plt.subplot(111)
        ax.scatter(basis_np[:, 0], basis_np[:, 1], label='landmarks', color='blue')

        # Setting the title and labels
        ax.set_title("Landmarks learnt by Sparse coding")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()

        # Show the plot
        plt.show()

    @torch.no_grad()
    def KNN_Data(self, data, k, PLOT=True):
    # input = k - number of neighbors to find
    # output = (N,N) (k-elements in second dimension =1 )
        
        ## move data back to cpu, because some unknown error for topk ##
        data = data.cpu()

        # Compute pairwise distances between all elements in model.Basis
        distances = torch.cdist(data, data, p=2) # (N,N)

        # Exclude the element itself by setting its distance to infinity
        distances.fill_diagonal_(float('inf'))

        # Find the indices of the K-nearest neighbors for each element
        _, indices = torch.topk(distances, k, largest=False) # (N,k)


        if PLOT==True:
            ## plot neighbors for randomly selected data point
            idx = torch.randint(low=0,high=data.shape[0],size=(1,)).item()
            print('idx: ', idx, '/', data.shape[0])

            print('neighbor idx: ', indices[idx])

            print('dist: ', distances[idx,indices[idx]])

            fig, [ax, ax1] = plt.subplots(figsize=[13,6],nrows=1, ncols=2)

            # plot dictionary elements
            ax.scatter(data[:, 0], data[:, 1], label='All landmarks', color='gray', alpha=0.3)
            ax.scatter(data[idx,0], data[idx,1], label='the data point', color='blue', alpha=1)
            neighbors = data[indices[idx],:]
            print(neighbors)
            ax.scatter(neighbors[:,0], neighbors[:,1], label='neighbors', color='red', alpha=1)

            # Setting the title and labels
            ax.set_title("Visualise K-nearest neighbors for datapoints")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()

            # Zoom in version
            ax1.scatter(data[idx,0], data[idx,1], label='the data point', color='blue', alpha=1)
            neighbors = data[indices[idx],:]
            ax1.scatter(neighbors[:,0], neighbors[:,1], label='neighbors', color='red', alpha=1)

            # Setting the title and labels
            ax1.set_title("Visualise K-nearest neighbors (zoomed in)")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.legend()
            plt.show()

        return indices

    @torch.no_grad()
    def KNN_B(self, k):
    # input = k - number of neighbors to find
    # output = (#basis,#basis) (k-elements in second dimension =1 )

        # Compute pairwise distances between all elements in model.Basis
        distances = torch.cdist(self.Basis, self.Basis, p=2) # (#basis,#basis) 

        # Exclude the element itself by setting its distance to infinity
        distances.fill_diagonal_(float('inf'))

        # Find the indices of the K-nearest neighbors for each element
        _, indices = torch.topk(distances, k, largest=False) #  (#basis,k)

        return indices
