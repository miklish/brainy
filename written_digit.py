import torch
import torchvision
from written_digit_image import load_image

# GReLU
def grelu(x) -> torch.Tensor:
    return torch.where(x >= 0, x, torch.exp(x) - 1)

# ENABLE GPU ACCELERATION
# get a torch.device for GPU acceleration if available
def enable_cuda() -> torch.device:
    device: torch.device

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA Available")
    else:
        device = torch.device('cpu')
        print("GPU Acceleration Not Available")
    return device

# THE MODEL
# Defines the forward pass (training / inference)
# Inherits from torch.nn.Module
# NOTE: Instantiate the mode with
#   model = Net().to(device) # puts this on GPU if available
# Calling
#   outputs = model(inputs)
# wil automatically run the forward pass and also setup the accumulation of gradients in each batch
# Do not call forward() directly
#
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Create the first fully connected layers (input->hidden layer)
        # Input layer is 28x28 inputs nodes (image size) and output layer is 500 nodes
        self.fc1 = torch.nn.Linear(28*28, 500)
        # Create the second fully connected layer (hidden->output layer)
        # This layer has 500 input nodes and 10 output nodes (for 10 digits classes)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        # Flatten the input image from 28*28 to a 1D tensor of size 784 (28*28)
        #
        # Some details of the pytorch API:
        # The `x.view()` function reshapes tensors
        # It returns a new tensor with the same data but in a different shape
        #
        # E.g.:
        #
        #     x = torch.randn(4, 4)
        #     z = x.view(-1, 8)
        #
        # the size -1 is inferred from other dimensions
        # z is a tensor with size (2, 8). z shares data with x
        #
        x = x.view(-1, 28*28)
        # Apply the first fully connected layer and ReLU (or GReLU) activation function
        x = torch.relu(self.fc1(x))
        # x = grelu(self.fc1(x))

        # Do calcs for hidden->output layers
        # We don't apply activation in the fc network containing the output layer since it
        # will be included in the loss function (specifically: cross entropy loss)
        x = self.fc2(x)
        return x

def train(
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    criterion: torch.nn.CrossEntropyLoss,
    trainloader: torch.utils.data.DataLoader):

    # TRAIN
    # Loop over the entire dataset (an epoch) multiple times (multiple epochs)
    for epoch in range(1):
        # For each mini-batch of images and labels...
        # this gives us the 'stocahstic' part of 'stochastic gradient descent'... its the random selection of mini-batches
        # due to the "shuffle=True" parameter in the DataLoader
        for i, data in enumerate(trainloader, 0):
            # get the inputs -- data is a list of [inputs, labels]
            # NOTE: we make sure we tell pytorch what device to put our raw data on
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward pass: compute predicted digits by passing inputs to the model
            outputs = model(inputs)

            # zero the parameter gradients
            # We have to do this because for some reason pytorch accumulates gradients across mini-batches by default
            # NOTE: We have to accumulate and then average gradients across all inputs in the same mini-batch,
            #       but not *across* different mini-batches
            optimizer.zero_grad()
            # calculate the loss for each element, then average to et a single scalar loss
            loss = criterion(outputs, labels)
            # backward pass: compute gradient of the scalar loss with respect to model node values (parameters)
            # this is the backpropagation step -- this is not technically 'gradient descent'
            loss.backward()
            # performs the gradient descent (e.g.: updates the weights by taking a step in the direction of the gradient)
            # NOTE: in pytorch they often refer to the partial derivatives of the loss with respect to each parameter
            # as a 'gradient', and therefore may talk of the 'gradients' (plural), even though there is really only one
            # gradient of the loss surface used in the optimizer step
            optimizer.step()

    print('Finished Training')

def test(testloader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device):
    total: int = 0
    correct: int = 0
    # torch.no_grad() disables gradient calculation -- We don't need to update the model during testing or inference
    with torch.no_grad():
        for data in testloader:
            images: torch.Tensor
            labels: torch.Tensor
            predicted: torch.Tensor
            outputs: torch.Tensor

            images, labels = data[0].to(device), data[1].to(device)
            # forward pass
            outputs = model(images)
            # get the predicted digit based on the max value in the output-list of class scores
            _, predicted = torch.max(outputs.data, 1)
            # count total number of images
            total += labels.size(0)
            # count number of correctly predicted images
            # correct_bool_tensor: torch.Tensor = (predicted == labels.to(device))
            # correct += correct_bool_tensor.sum().item()
            correct += (predicted == labels.to(device)).sum().item()

    # print accuracy of the model
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def train_run_test():
    transform: torchvision.transforms.Compose
    trainset: torchvision.datasets.MNIST        # labeled digit image dataset for training
    trainloader: torch.utils.data.DataLoader    # code to load in images in batches
    model: torch.nn.Module                      # neural network model
    criterion: torch.nn.CrossEntropyLoss        # standard loss function for classification
    optimizer: torch.optim.Optimizer            # 'stochastic' gradient descent learning algorithm
    testset: torchvision.datasets.MNIST         # labeled digit image dataset for testing
    testloader: torch.utils.data.DataLoader     # code to load in test images in batches
    device: torch.device                        # device to run the model on

    # NOTE: The SGD optimizer is NOT stochastic! The stocahstic selection of mini-batches is done by the dataloader
    #       'shuffle'. It is NOT due to the 'stochastic' optimizer! The SGD optimizer is just vanilla gradient descent

    # Define a transform to convert the data to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Download and load the MNIST training dataset (this is auto-cached so only downloads once)
    # Apply the transform when loading the data
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # DataLoader loads data in batches, shuffles it and parallelize the loading process
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Create an instance of the model
    device = enable_cuda()      # Enable a GPU acceleration device if available
    model = Net().to(device)    # Move the model to the GPU if available

    # Define the loss function and the optimizer
    # CrossEntropyLoss is standard for classification problems
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    train(device, optimizer, model, criterion, trainloader)

    # TEST
    # Download and load the test data
    # Apply the graphic transform when loading the test data
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    test(testloader, model, device)

    return model

# Load a digit image from a file and make a digit prediction
def inference(model_weights_file: str, image_tensor: torch.Tensor):
    device: torch.device
    model_pt: torch.nn.Module

    device = enable_cuda()
    model_pt = Net().to(device)
    model_pt.load_state_dict(torch.load(model_weights_file))

    model_pt.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model_pt(image_tensor)
        _, predicted = torch.max(output.data, 1)
        print(f'Predicted digit: {predicted.item()}')

######

# if __name__ == "__main__":
#     model: torch.nn.Module = train_run_test()
#     torch.save(model.state_dict(), 'model_weights.pth')

# if __name__ == "__main__":
#     inference('model_weights.pth')

if __name__ == "__main__":
    image_tensor: torch.Tensor = load_image('digit7.png')
    inference('model_weights.pth', image_tensor)

# if __name__ == "__main__":
#     compare_mnist_user_imgs()
