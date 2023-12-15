import nibabel as nib
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from IPython.display import HTML
sns.set_style("darkgrid")

def show_irm(data: np.ndarray):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 1)

    # Initialize images for each subplot
    im1 = axs.imshow(data[:, :, 0], cmap='gray', vmin=0, vmax=1)

    # Function to update the plots for each frame
    def update(frame):
        im1.set_data(data[:, :, frame])
        axs.set_title(f'Slice {frame}')
        axs.grid(False)

    # Create an animation
    num_slices = data.shape[2]
    anim = animation.FuncAnimation(fig, update, frames=num_slices, interval=100, blit=False)
    plt.close()
    return HTML(anim.to_html5_video())