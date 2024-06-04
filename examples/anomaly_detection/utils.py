import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import ae

def scheduler(epoch, lr):
    if epoch in [25, 27]:
        return lr * 0.1
    return lr


def plot_manifold(autoencoder, x, y, filesave=None):
    x_latent = autoencoder.get_x_latent(x=x, batch_size=32)
    #latent_transform = TSNE(n_components=2).fit_transform(x_latent)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_latent[:, 0],
                    y=x_latent[:, 1],
                    hue=y,
                    palette=sns.color_palette("hsv", 10), 
                    legend='full')
    
    plt.title('Latent space')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    if filesave is not None:
        plt.savefig(filesave)
    plt.show()

def mse(img1, img2):
    err = np.sum((img1.astype('float') - img2.astype('float')) ** 2)
    err /= float(img1.shape[0])
    return err

def plot_history(history, filesave=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    if filesave is not None:
        plt.savefig(filesave)
    plt.show()
    
def plot_rescontruction_and_actual_images(autoencoder, x, y):
    num_images = len(np.unique(y))
    
    unique_labels = np.unique(y)
    plt.figure(figsize=(20, 4))
    
    x_show = []
    y_show = []
    for img, label in zip(x, y):
        if label not in y_show:
            y_show.append(label)
            x_show.append(img)
        if len(y_show) == num_images:
            break
    
    if type(autoencoder) != ae.AutoEncoder:
        x_show_reshape = np.concatenate(x_show).reshape(-1, 28, 28, 1)
        x_reconstructed = autoencoder.predict(x=x_show_reshape).reshape(-1, 784)
    else:
        x_show_reshape = np.concatenate(x_show).reshape(-1, 784)
        x_reconstructed = autoencoder.predict(x=x_show_reshape)
    
    for i in range(num_images):
        ax = plt.subplot(2, num_images, i+1)
        plt.imshow(x_show[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(x_reconstructed[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show() 
    
        
    
    