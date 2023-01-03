""" Neural Network in PyTorch
first, loading and preprocessing datasets using pytorch.
and then implementing and training a neural network (multi-layer perceptron) for handwriting recognition (MNIST dataset),
 using Pytorch.

In this exercise, we are going to build a neural network that identify handwritten digits.
We will use the MNIST dataset which consists of greyscale handwritten digits.
Each image is 28x28 pixels and there are 10 different digits.
The network will take these images and predict the digit in them.
implementing a neural network with pytorch that gets mnist images and recognize the digit in them.

Let's visualize the data before working with it.
1. We can use the "torchvision" package to download the trainset.
 Set ```transform``` as to be the transform function below (It normalizes each image) and ```train=True```.
2. We use torch.utils.data.DataLoader to load the data. Set ```batch_size=64```.
"""

# load packages
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

data_path = "./MNIST_data"

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the data
mnist_data = datasets.MNIST(data_path, download=True, train=True, transform=transform)
mnist_dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=64, shuffle=True)

# get single batch
dataiter = iter(mnist_dataloader)
batch_images, batch_labels = next(dataiter)

# Print the number of samples in the whole dataset.
print(len(mnist_data))

# Print the number of samples in a single batch.
print(len(batch_labels))

# Print the shape of images in the data (image dimensions).
print(batch_images[0].numpy().shape)

# Print the number of labels in the whole dataset (using the targets in the dataloader).
print(len(mnist_dataloader.dataset.targets))

# plot three images and print their labels
idx = np.random.choice(range(64), 3)  # three rundom indices
plt.subplot(1, 3, 1)
plt.imshow(batch_images[idx[0]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1, 3, 2)
plt.imshow(batch_images[idx[1]].numpy().squeeze(), cmap='Greys_r')
plt.subplot(1, 3, 3)
plt.imshow(batch_images[idx[2]].numpy().squeeze(), cmap='Greys_r')
print("Labels:", batch_labels[idx])

"""
Neural Network Architecture: 
784 input units, a hidden layer with 128 units and a ReLU activation,
then a hidden layer with 64 units and a ReLU activation,
and finally an output layer with a log-softmax activation.

Since simple neural networks get vectors as inputs, and not images (unlike CNNs), we should flatten the data.
Each sample with shape (28,28) becomes (784,). That is why the input layer has 784 units.
"""

from torch import nn, optim
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    # The constructor contains definitions of layers like Linear or Relu

    def __init__(self):
        '''
        Declare layers for the model
        '''
        super().__init__()

        ############## option 1: ##################
        self.flatten = nn.Flatten()
        # defining the layers dimentions for our model: 784->128->64->10
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)  # softmax dim???
        )

        ############## option 2: ##################
        # self.fc0 = nn.Linear(28*28, 128)
        # self.fc1 = nn.Linear(128, 64)
        # self.fc2 = nn.Linear(64, 10)

    # The 'forward' function contains the logic between the layers declared in the constructor
    def forward(self, x):
        ''' Forward pass through the network, returns log_softmax values '''

        ############## option 1: ##################
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

        ############## option 2: ##################
        # x = F.relu(self.fc0(x))
        # x = F.relu(self.fc1(x))
        # # softmax is used in the output layer
        # x = F.log_softmax(self.fc2(x), dim=1) # softmax dim???
        # return x


model = NeuralNetwork()
model

"""  
choose a random image and pass it through the network to get the prediction - 'confidences' for each class.
The class with the highest confidence is the prediction of the model for that image.   
"""
# visualize the results
def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    image - the input image to the network
    ps - the class confidences (network output)
    '''
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


def random_prediction_example(data_loader, model):
    '''
    The function sample an image from the data, pass it through the model (inference)
    and show the prediction visually. It returns the predictions confidences.
    '''
    # take a batch and randomly pick an image
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    images.resize_(64, 1, 784)
    img = images[0]

    # Forward pass through the network
    # we use torch.no_grad() for faster inference and to avoid gradients from
    # moving through the network.
    with torch.no_grad():
        ps = model(img)
        # the network outputs log-probabilities, so take exponential for probabilities
        ps = torch.exp(ps)

    # visualize image and prediction
    view_classify(img.view(1, 28, 28), ps)
    return ps


preds_conf = random_prediction_example(mnist_dataloader, model)

# Print the prediction of the network for that sample:
# print(np.array(preds_conf[0]))
print(np.array(preds_conf.argmax()))
# print("predicted number probability:", np.array(preds_conf[0][preds_conf.argmax()]))

"""Neural Network - Training anf evaluating """

from torch.utils import data

# 1. split trainset into train and validation (use torch.utils.data.random_split())
# Train-set is 80%, test-set is 20%
train_size = int(0.8 * len(mnist_data))
test_size = len(mnist_data) - train_size
train_dataset, test_dataset = data.random_split(mnist_data, [train_size, test_size])

# create data loader for the trainset (batch_size=64, shuffle=True)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# create data loader for the valset (batch_size=64, shuffle=False)
val_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# set hyper parameters
learning_rate = 0.003
nepochs = 5

model = NeuralNetwork()

# create sgd optimizer. It should optimize our model parameters with learning_rate defined above
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# create a criterion object. It should be negative log-likelihood loss since the task
#    is a multi-task classification (digits classification)
criterion = nn.NLLLoss()


# Train the model.
def train_model(model, optimizer, criterion,
                nepochs, train_loader, val_loader, is_image_input=False):
    '''
    Train a pytorch model and evaluate it every epoch.
    Params:
    model - a pytorch model to train
    optimizer - an optimizer
    criterion - the criterion (loss function)
    nepochs - number of training epochs
    train_loader - dataloader for the trainset
    val_loader - dataloader for the valset
    is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                  Should be True for Neural Networks
                                  but False for Convolutional Neural Networks.
    '''
    train_losses, val_losses = [], []
    for e in range(nepochs):
        running_loss = 0
        running_val_loss = 0
        for images, labels in train_loader:
            if is_image_input:
                # Flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)

            # Training pass
            model.train()  # set model in train mode
            # zero the gradient buffers
            optimizer.zero_grad()
            # Passing the data (images) into the neural network model to compute prediction error
            prediction_output = model(images)
            # measuring the loss function
            loss = criterion(prediction_output, labels)

            # Backpropagation
            loss.backward()
            # Update all the weights of the model
            optimizer.step()
            running_loss += loss.item()
        else:
            val_loss = 0
            # 6.2 Evalaute model on validation at the end of each epoch.
            with torch.no_grad():
                for images, labels in val_loader:
                    if is_image_input:
                        # Flatten MNIST images into a 784 long vector
                        images = images.view(images.shape[0], -1)
                        # Compute prediction error
                    prediction_output = model(images)
                    val_loss = criterion(prediction_output, labels)
                    running_val_loss += val_loss.item()

            # 7. track train loss and validation loss
            train_losses.append(running_loss / len(train_loader))
            val_losses.append(running_val_loss / len(val_loader))

            print("Epoch: {}/{}.. ".format(e + 1, nepochs),
                  "Training Loss: {:.3f}.. ".format(running_loss / len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(running_val_loss / len(val_loader)))
    return train_losses, val_losses


# Train the model
train_losses, val_losses = train_model(model, optimizer, criterion, nepochs,
                                       train_loader, val_loader, is_image_input=True)


# Generate a sequence of integers to represent the epoch numbers
epochs = range(nepochs)

# Plot train loss and validation loss as a function of epoch
# Plot and label the training and validation loss values
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.show()

# check predictions of trained network
random_prediction_example(mnist_dataloader, model)

"""Calculate the model's accuracy on the validation-set."""

def evaluate_model(model, val_loader, is_image_input=False):
    '''
    Evaluate a model on the given dataloader.
    Params:
    model - a pytorch model to train
    val_loader - dataloader for the valset
    is_image_input (default False) - If false, flatten 2d images into a 1d array.
                                     Should be True for Neural Networks
                                     but False for Convolutional Neural Networks.
    '''
    validation_accuracy = 0
    with torch.no_grad():
        for images, labels in val_loader:
            if is_image_input:
                # flatten MNIST images into a 784 long vector
                images = images.view(images.shape[0], -1)
            # forward pass
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            # count correct predictions
            equals = top_class == labels.view(*top_class.shape)

            validation_accuracy += torch.sum(equals.type(torch.FloatTensor))
    res = validation_accuracy / len(val_loader.dataset)
    return res


print(f"Validation accuracy: {evaluate_model(model, val_loader, is_image_input=True)}")
