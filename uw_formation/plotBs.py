import torch
import matplotlib.pyplot as plt
import os

def save_backscatter_plot(x_tensor_in, y_tensor_in, str_name, wandb, iteration):
    """
    Save all channels of a CxHxW tensor as a plot in the same figure using Seaborn.

    Parameters:
    x_tensor (torch.Tensor): A tensor of shape (C, H, W) for x-axis values.
    y_tensor (torch.Tensor): A tensor of shape (C, H, W) for y-axis values.
    file_name (str): The file name to save the plot.
    """
    x_tensor = x_tensor_in.detach().cpu()
    y_tensor = y_tensor_in.detach().cpu()
    channels, height, width = y_tensor.shape
    rgb_name = ['R', 'G', 'B']
    
    # Prepare the data
    x_channel = x_tensor[0, :, :]
    x_values = x_channel.flatten().numpy()
    
    #plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(11.7, 8.27))
    for c in range(channels):
        y_channel = y_tensor[c, :, :]
        y_values = y_channel.flatten().numpy()
        
        # Using Seaborn to plot
        plt.plot(x_values, y_values, '.', label=f'Channel {rgb_name[c]}', color=rgb_name[c].lower())
        #sns.scatterplot(x=x_values, y=y_values, label=f'Channel {rgb_name[c]}', color=rgb_name[c].lower(), s=10)
    
    plt.xlabel('z values')
    plt.ylabel('Bc values')
    plt.title('Plot of All Channels of the Tensor')
    plt.legend()  # Add a legend to distinguish between channels
    
    # Save the plot to a file
    file_name = os.path.join('uw_formation', 'plots', f'bs_plot.png')
    plt.savefig(file_name)
    plt.close()
    # Log the plot with wandb
    wandb.run.log(data={str_name: wandb.Image(file_name)}, step=iteration)