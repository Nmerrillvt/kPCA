import numpy as np
import matplotlib.pyplot as plt
import keras

def Seedy(seed_value= 10):
    
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.compat.v1.set_random_seed(seed_value)
    
def tile_images(images, row=4,):
    shape = images.shape[1:]
    column = int(images.shape[0] / row)
    height = shape[0]
    width = shape[1]
    tile_height = row * height
    tile_width = column * width
    output = np.zeros((tile_height, tile_width, shape[-1]), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            image = images[i*column+j]
            output[i*height:(i+1)*height,j*width:(j+1)*width] = image
    return output

def display_tile(data, idx = np.arange(16)):

    
    if len(data.shape) == 2:
        images = np.expand_dims(np.reshape(data[idx,:], (16,28,28)),axis = 3)
    elif len(data.shape) == 4:
        images = data[idx,:,:]
    else:
        assert False, "Wrong Dimensions"
        
    
    original_images = np.array(images * 255, dtype=np.uint8)
    fig3 = np.squeeze(tile_images(original_images))
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4,4))
    axes.imshow(fig3, cmap='Greys_r')
    axes.xaxis.set_visible(False)
    axes.yaxis.set_visible(False)
    return fig

        
def display_one(sample):
    sample = np.squeeze(sample)
    plt.figure()
    plt.imshow(sample, cmap='Greys_r')
    plt.show()
    

def param_heatmap(method,fig,ax,xparam,yparam,mesh,xlabel,ylabel,bestx,besty,log=False):

    x, y = np.meshgrid(xparam,yparam)
    z = mesh#-np.log(np.abs(1-(mesh)+1e-8)) #scaling to emphasize upper range differences in AUC
    c = ax.pcolormesh(x, y, z, cmap='viridis', vmin=z.min(), vmax=z.max())
    ax.set_title(method)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    if log:
        ax.set_xscale('log')
    cbar =fig.colorbar(c,  ticks=[z.min(), np.median(z),z.max()],ax=ax)
    ax.scatter(bestx,besty,marker = "*",s = 10,color = 'r')
    #cbar.ax.set_yticklabels([round(1-np.exp(-z.min()),3), 0.5, round(1-np.exp(-z.max()),3)])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)   
    
    
def param_scatter(method,fig,ax,xparam,aucs,xlabel,log=False):
    x = xparam
    y = aucs
    c = ax.scatter(x,y)
    ax.set_title(method)
    ax.axis([x.min(), x.max(), y.min(), y.max()]) 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('AUC')
    if log:
        ax.set_xscale('log')
    
    
    
    
    