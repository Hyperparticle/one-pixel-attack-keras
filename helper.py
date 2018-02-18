# CIFAR - 10

# To decode the files
import pickle
# For array manipulations
import numpy as np
# To make one-hot vectors
from keras.utils import np_utils
# To plot graphs and display images
from matplotlib import pyplot as plt


#constants

path = "data/"  # Path to data 

# Height or width of the images (32 x 32)
size = 32 

# 3 channels: Red, Green, Blue (RGB)
channels = 3  

# Number of classes
num_classes = 10 

# Each file contains 10000 images
image_batch = 10000 

# 5 training files
num_files_train = 5  

# Total number of training images
images_train = image_batch * num_files_train

# https://www.cs.toronto.edu/~kriz/cifar.html


def unpickle(file):  
    
    # Convert byte stream to object
    with open(path + file,'rb') as fo:
        print("Decoding file: %s" % (path+file))
        dict = pickle.load(fo, encoding='bytes')
       
    # Dictionary with images and labels
    return dict




def convert_images(raw_images):
    
    # Convert images to numpy arrays
    
    # Convert raw images to numpy array and normalize it
    raw = np.array(raw_images, dtype = float) / 255.0
    
    # Reshape to 4-dimensions - [image_number, channel, height, width]
    images = raw.reshape([-1, channels, size, size])

    images = images.transpose([0, 2, 3, 1])

    # 4D array - [image_number, height, width, channel]
    return images




def load_data(file):
    # Load file, unpickle it and return images with their labels
    
    data = unpickle(file)
    
    # Get raw images
    images_array = data[b'data']
    
    # Convert image
    images = convert_images(images_array)
    # Convert class number to numpy array
    labels = np.array(data[b'labels'])
        
    # Images and labels in np array form
    return images, labels




def get_test_data():
    # Load all test data
    
    images, labels = load_data(file = "test_batch")
    
    # Images, their labels and 
    # corresponding one-hot vectors in form of np arrays
    return images, labels, np_utils.to_categorical(labels,num_classes)




def get_train_data():
    # Load all training data in 5 files
    
    # Pre-allocate arrays
    images = np.zeros(shape = [images_train, size, size, channels], dtype = float)
    labels = np.zeros(shape=[images_train],dtype = int)
    
    # Starting index of training dataset
    start = 0
    
    # For all 5 files
    for i in range(num_files_train):
        
        # Load images and labels
        images_batch, labels_batch = load_data(file = "data_batch_" + str(i+1))
        
        # Calculate end index for current batch
        end = start + image_batch
        
        # Store data to corresponding arrays
        images[start:end,:] = images_batch        
        labels[start:end] = labels_batch
        
        # Update starting index of next batch
        start = end
    
    # Images, their labels and 
    # corresponding one-hot vectors in form of np arrays
    return images, labels, np_utils.to_categorical(labels,num_classes)
        


def get_class_names():

    # Load class names
    raw = unpickle("batches.meta")[b'label_names']

    # Convert from binary strings
    names = [x.decode('utf-8') for x in raw]

    # Class names
    return names



def plot_images(images, labels_true, class_names, labels_pred=None):

    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize = (8,8))

    # Adjust the vertical spacing
    if labels_pred is None:
        hspace = 0.2
    else:
        hspace = 0.5
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i], interpolation='spline16')
            
            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: "+labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: "+labels_true_name+"\nPredicted: "+ labels_pred_name

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Show the plot
    plt.show()
    

def plot_model(model_details):

    # Create sub-plots
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # Summarize history for accuracy
    axs[0].plot(range(1,len(model_details.history['acc'])+1),model_details.history['acc'])
    axs[0].plot(range(1,len(model_details.history['val_acc'])+1),model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_details.history['acc'])+1),len(model_details.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # Summarize history for loss
    axs[1].plot(range(1,len(model_details.history['loss'])+1),model_details.history['loss'])
    axs[1].plot(range(1,len(model_details.history['val_loss'])+1),model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_details.history['loss'])+1),len(model_details.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    
    # Show the plot
    plt.show()



def visualize_errors(images_test, labels_test, class_names, labels_pred, correct):
    
    incorrect = (correct == False)
    
    # Images of the test-set that have been incorrectly classified.
    images_error = images_test[incorrect]
    
    # Get predicted classes for those images
    labels_error = labels_pred[incorrect]

    # Get true classes for those images
    labels_true = labels_test[incorrect]
    
    
    # Plot the first 9 images.
    plot_images(images=images_error[0:9],
                labels_true=labels_true[0:9],
                class_names=class_names,
                labels_pred=labels_error[0:9])
    
    
def predict_classes(model, images_test, labels_test):
    
    # Predict class of image using model
    class_pred = model.predict(images_test, batch_size=32)

    # Convert vector to a label
    labels_pred = np.argmax(class_pred,axis=1)

    # Boolean array that tell if predicted label is the true label
    correct = (labels_pred == labels_test)

    # Array which tells if the prediction is correct or not
    # And predicted labels
    return correct, labels_pred

def random_pixel():
    # x,y,r,g,b
    gen = np.array([
        np.random.randint(0, 32),
        np.random.randint(0, 32),
        *truncnorm(-0.5, 0.5).rvs(size=3)])
    gen += np.array([0, 0, 0.5, 0.5, 0.5])
    return gen

def perturb_image_relative(x, img):
    img = np.copy(img)
    pixels = np.split(x.astype(int), len(x) // 5)
    scale = np.repeat(128, 3)
    for x in pixels:
        x_pos, y_pos, rgb = *x[:2], x[2:]
        img[x_pos][y_pos] = np.clip(img[x_pos][y_pos] + rgb - scale, 0, 255)
    return img
