import torch
import torch.nn as nn
import torchvision
import PIL.Image as Image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import string
from typing import List
from typing import Tuple
import os

# THE EXPERT NETWORK
# - Defines the forward pass (training / inference)
# - Inherits from torch.nn.Module
# - NOTE: Instantiate the model with
#     expert = Expert().to(device) # puts this on GPU
# - Calling
#       outputs = expert(inputs)
#   will run the forward pass
# - Do not call forward() directly
#
# The Expert class
#   The Expert class defines a neural network model that consists of two fully connected layers.
# Initialization
#   - fc1: The first fully connected layer, takes an input of size 28*28 (flattened image) and outputs 500 nodes.
#   - fc2: The second fully connected layer, takes the 500 nodes from fc1 and outputs 1 node.
# Forward Pass (forward method)
#   - Flattening: The input image is flattened from a 2D tensor of size 28*28 to a 1D tensor of size 784.
#   - First Layer: The flattened input is passed through the first fully connected layer (fc1) and a ReLU activation function.
#   - Second Layer: The output of the first layer is passed through the second fully connected layer (fc2).
# Output
#   The output of the Expert class are 10 'logits' (unactivated output) for each input image.
#
class Expert(nn.Module):
    def __init__(self, input_size, hidden_size=500):
        super(Expert, self).__init__()

        # Create the first fully connected layers (input->hidden layer)
        # Input layer is 28x28 inputs nodes (image size) and output layer is 500 nodes
        self.fc1 = torch.nn.Linear(input_size, 500)
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
        x = x.view(-1, INPUT_SIZE)  # Flatten the image -> [MINI_BATCH_SIZE, 784]

        # Apply the first fully connected layer and ReLU (or GReLU) activation function
        x = self.fc1(x)     # -> [MINI_BATCH_SIZE, 500]
        x = torch.relu(x)   # -> [MINI_BATCH_SIZE, 500]

        # Do calcs for hidden->output layers
        # We don't apply activation in the fc network containing the output layer since it
        # will be included in the loss function (specifically: cross entropy loss)
        x = self.fc2(x)     # -> [MINI_BATCH_SIZE, 10]

        # Return the "logits" (non-softmax'd probability distribution over 10 digits)
        #
        # CrossEntropyLoss is applied AFTER the logits from each expert are weighted by
        # the gating network and combined
        #
        # Note: While softmax is applied to the gating network's probability over all experts,
        #       the experts' outputs (logits) do NOT have softmax applied
        #
        # 1. The loss function (CrossEntropyLoss) automatically
        #   a. Computes the loss for each example in the mini-batch
        #   b. Averages these losses to get a single scalar loss value
        #
        # 2. When loss.backward() is called:
        #   a. PyTorch calculates gradient of this average loss with respect to all model parameters
        #   b. The gradient reperesents how each parameter should change to minimize the average loss
        #
        # 3. The optimizer then updates all parameters using the gradient in a single step
        #    using optimizer.step().
        #
        #    Parameters are updated once per mini-batch.
        #
        return x    # [MINI_BATCH_SIZE, 10]

# THE GATING MODEL
# Inherits from torch.nn.Module
# NOTE: Instantiate the mode with
#     gating_network = GatingNetwork().to(device) # puts this on GPU
# - Calling
#       outputs = gating_network(inputs)
#   will run the forward pass
# - Do not call forward() directly
#
# GatingNetwork class
#   This class initializes two fully connected layers:
#     - fc1: Takes the flattened input image (size 28*28) and outputs GATING_HIDDEN_LAYER_SIZE nodes
#     - fc2: Takes the GATING_HIDDEN_LAYER_SIZE nodes from fc1 and outputs a "logit" for each expert
#       (NUM_EXPERTS experts in this case) for each digit (10 classes)
#
# Forward Pass (forward method):
#   The input image is flattened into a 1D tensor of size 28*28.
#   The flattened image is passed through the first fully connected layer (fc1) and a ReLU activation function.
#   The output of fc1 is passed through the second fully connected layer (fc2).
#   The output of fc2 is passed through a softmax function to produce a probability distribution over experts
#
# How the experts are used during training
#   The GatingNetwork outputs a probability distribution over the experts for each input image
#
#   Its important to note that this is a true probability distribution since the output of layer fc2 is passed through
#   a softmax function. This will come up again when we see how this differs to how we do inference
#
#   Between NUM_EXPERTS and TOP_K experts are utilized. If TOP_K < NUM_EXPERTS, then only the gating network results
#   for the top-k experts for each picture are used (the rest are thrown away). The remainder are re-normalized
#   (but not re-softmax'd)
#
#   The outputs of the selected experts are combined by weighting the expert outputs by these probabilities
#   and summed to make the final prediction
#
# How the experts are used during inference
#   During inference (as in training) the GatingNetwork outputs a probability distribution over some subset of experts
#   for each input image
#
#   However now, we only use exactly TOP_K experts -- determined by the largest (top) k probabilities generated by the
#   gating network
#
#   The outputs of the selected experts are combined by weighting the expert outputs by their respective probabilities
#   and summed to make the final prediction
#
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()

        # Create the first fully connected layers (input->hidden layer)
        # Input layer is 28x28 inputs nodes (image size) and output layer is GATING_HIDDEN_LAYER_SIZE nodes
        self.fc1 = nn.Linear(input_size, GATING_HIDDEN_LAYER_SIZE)

        # Second fully connected layer: creates probability distribution over experts
        self.fc2 = nn.Linear(GATING_HIDDEN_LAYER_SIZE, num_experts)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)  # Flatten the image

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.softmax(x, dim=1)

        # output is a probability distribution over the experts for each input image
        # where the probability represents the confidence of the gating network in each expert
        return x

########################################################################################################################

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

def get_device_experts_gate_transform(is_test: bool) -> Tuple [
    torch.device,
    List[torch.nn.Module],
    torch.nn.Module,
    torchvision.transforms.Compose]:

    # Enable GPU if available
    device = enable_cuda()

    # Create models
    experts: List[nn.Module]
    gating_network: nn.Module
    if is_test:
        # Load models (for inference)
        experts, gating_network = load_models(INPUT_SIZE, NUM_EXPERTS, device)
    else:
        # load models for training
        experts, gating_network = load_blank_models(INPUT_SIZE, NUM_EXPERTS, device)

    # Define a transform to convert the data to tensor
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    return device, experts, gating_network, transform

########################################################################################################################

def train():
    device, experts, gating_network, transform = get_device_experts_gate_transform(is_test=False)

    # Load datasets
    training_dataset = MNIST(root='./data', train=True, download=True, transform=transform)

    # Create data loader
    # NOTE: The SGD optimizer is NOT stochastic! The stocahstic selection of mini-batches is done by the dataloader
    #       'shuffle'. It is NOT due to the 'stochastic' optimizer! The SGD optimizer is just vanilla gradient descent
    training_minibatch_loader = DataLoader(training_dataset, batch_size=MINI_BATCH_SIZE, shuffle=True)

    # Set the loss function
    loss_ftn = nn.CrossEntropyLoss()

    # Combine parameters from the gating network and expert networks
    all_parameters = list(gating_network.parameters()) + [p for expert in experts for p in expert.parameters()]

    # Create SGD optimizer with momentum and weight decay
    optimizer = torch.optim.SGD(all_parameters, lr=LEARNING_RATE)

    # Train the mixture of experts
    train_mixture_of_experts(
        device, experts, gating_network, loss_ftn, optimizer, training_minibatch_loader,
        epochs=EPOCHS, num_experts=NUM_EXPERTS, top_k=TOP_K)

    # Save models
    save_models(experts, gating_network)

def train_mixture_of_experts(
    device: torch.device,
    experts: List[nn.Module],
    gating_network: torch.nn.Module,
    loss_ftn: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    training_minibatch_loader: torch.utils.data.DataLoader,
    epochs: int,
    num_experts: int,
    top_k: int):

    current_top_k: int
    epochs = max(epochs, top_k)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        epochs_left = epochs - epoch

        if epochs_left >= num_experts:
            current_top_k = num_experts
        else:
            # Decrease current_top_k each epoch until reach TOP_K
            current_top_k = max(top_k, epochs_left)

        print(f'Epoch {epoch + 1}, using top-{current_top_k} experts')

        for image_minibatch, label_minibatch in training_minibatch_loader:
            # Move data to device
            image_minibatch = image_minibatch.to(device) # [MINI_BATCH_SIZE, 1, IMAGE_WIDTH, IMAGE_HEIGHT]
            label_minibatch = label_minibatch.to(device) # [MINI_BATCH_SIZE]
            mini_batch_size = image_minibatch.size(dim=0) # size of dim=0 which is MINI_BATCH_SIZE

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass through gating network
            gating_outputs = gating_network(image_minibatch) # Shape [mini_batch_size, num_experts]

            # Get top k experts using current_top_k
            _, top_k_indices = torch.topk(input=gating_outputs, k=current_top_k, dim=1)

            # The shape of top_k_indices is [mini_batch_size, current_top_k]

            # Create mask for top k experts
            mask = torch.zeros_like(gating_outputs) # Shape [mini_batch_size, num_experts]
            mask.scatter_(dim=1, index=top_k_indices, value=1.0) # Shape [mini_batch_size, num_experts]

            # Zero out non-top-k expert weights
            gating_outputs_masked = gating_outputs * mask

            # sum all values along dim=1 (the experts dimension) for each sample in the batch
            # keepdim=True preserves the dimensionality, resulting in shape [mini_batch_size, 1]
            gating_outputs_sum = gating_outputs_masked.sum(dim=1, keepdim=True) # Shape [mini_batch_size, 1]

            # renormalize gating_outputs_masked to ensure the gating weights for the top-k experts sum to 1
            # for each sample in the mini batch

            # [mini_batch_size, num_experts] / [mini_batch_size, 1]
            # broadcast the second dimension of gating_outputs_sum from 1 to num_experts
            # by repeating the value in dim=1 num_experts times
            gating_outputs_k = gating_outputs_masked / gating_outputs_sum # Shape [mini_batch_size, num_experts]

            # Initialize tensor to hold all expert outputs
            # The shape of e_outputs is [mini_batch_size, num_experts, 10]
            e_outputs = torch.zeros(mini_batch_size, num_experts, 10).to(device)  # 10 classes for MNIST

            # For each expert, find which images need it
            for e in range(num_experts):
                # Find images that selected this expert in their top-k

                # these are the mask values for expert e. The shape of this tensor is [mini_batch_size]
                # mask[:, e] = [0, 1, 0, 1, 0, 0, 1, ..., 0, 1]
                mask_values_for_e = mask[:, e]

                # this gets the indexes of the non-zero values in the mask for e
                # these are in fact the indexes of the images we want to run expert e on
                # the `[0]` is needed since this could return a tuple of multiple tensors (one per dimension),
                # however, since mask_values_for_e is only 1D, there is just 1 tensor in the tuple
                indexes_of_images_needing_expert_e = torch.where(mask_values_for_e == 1)[0]

                # Only process if at least one image needs this expert
                if len(indexes_of_images_needing_expert_e) > 0:
                    # Create mini-batch of just those images
                    e_minibatch = image_minibatch[indexes_of_images_needing_expert_e]

                    # Run expert on just those images
                    outputs = experts[e](e_minibatch)

                    '''
                    Place outputs in the correct positions in our result tensor
                    
                    Code pytorch code we use for is equivalent to the following 'plain python':

                    for i, image in enumerate(indexes_of_images_needing_expert_e):
                        # Place the output for this image at the correct position in e_outputs
                        e_outputs[image, e] = outputs[i]
                    '''
                    e_outputs[indexes_of_images_needing_expert_e, e] = outputs

            # Weight and combine outputs
            gate_confidence = gating_outputs_k.unsqueeze(2)  # Shape: [mini_batch_size, num_experts] -> [batch_size, num_experts, 1]

            # [mini_batch_size, num_experts, 10] x [mini_batch_size, num_experts, 1] = [mini_batch_size, num_experts, 10]
            # weighted_outputs(i, e, d) = gate_conf(i, e, 1) * e_outputs(i, e, d)
            weighted_outputs = gate_confidence * e_outputs

            # [i, d] = +(i, e=*, d)
            combined_outputs = torch.sum(weighted_outputs, dim=1)  # Shape: [batch_size, num_classes]

            # Compute the loss
            loss = loss_ftn(combined_outputs, label_minibatch)

            # Accumulate loss for logging
            epoch_loss += loss.item()
            batch_count += 1

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

        avg_loss = epoch_loss / batch_count
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

########################################################################################################################

# loads user-created image from disk and converts it to a tensor
def load_image(image_path: string) -> torch.Tensor:
    image: Image = Image.open(image_path)

    # Extract the alpha channel (handles transparency correctly)
    if image.mode == "RGBA":
        _, _, _, image = image.split()  # Get the alpha channel and use as the black-and-white image

    # Convert to grayscale explicitly (ensures it's in single-channel format)
    image = image.convert("L")

    # Apply thresholding to enforce black (0) and white (255)
    image = image.point(lambda p: 255 if p > 128 else 0)

    # Define the transform: Convert to tensor and normalize to [0,1] (default for ToTensor)
    transform = torchvision.transforms.ToTensor()

    # Apply the transformation
    tensor_image = transform(image)  # Shape: [1, 28, 28]

    return tensor_image

def inference():
    # Load the image
    image_path = input("Enter the path to the image: ")
    image_tensor = load_image(image_path)

    # Get models and device
    device, experts, gating_network, _ = get_device_experts_gate_transform(is_test=True)

    # Perform inference
    inference_run(image_tensor, device, experts, gating_network, TOP_K)

def inference_run(
    image_tensor: torch.Tensor,
    device: torch.device,
    experts: List[nn.Module],
    gating_network: nn.Module,
    top_k: int):

    # Move image to device and add batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 1, 28, 28]
    image_tensor = image_tensor.to(device)

    predicted_digit: int

    # Disable gradient computation for inference
    with torch.no_grad():
        # Get gating outputs
        gating_outputs = gating_network(image_tensor)  # [1, num_experts]


        #### TOP_K EXPERTS ####

        # Get indices of top k experts
        top_k_values, top_k_indices = torch.topk(gating_outputs, top_k, dim=1)  # [1, top_k]

        # Only run selected experts
        expert_outputs_list = []
        for idx in top_k_indices[0]:  # [0] because batch size is 1
            expert_output = experts[idx](image_tensor).unsqueeze(1)  # [1, 1, num_classes]
            expert_outputs_list.append(expert_output)

        # Only combine outputs from selected experts
        expert_outputs = torch.cat(expert_outputs_list, dim=1)  # [1, top_k, num_classes]
        gating_weights = top_k_values.unsqueeze(2)  # [1, top_k, 1]
        
        # Normalize weights to sum to 1
        gating_weights = gating_weights / gating_weights.sum(dim=1, keepdim=True)
        
        #######################


        # Weight and combine expert outputs
        weighted_outputs = expert_outputs * gating_weights  # [1, top_k, num_classes]

        # This is the element-wise multiplication of the expert outputs and the gating weights
        # The gating weights are "broadcasted" to match the shape of the expert outputs
        # This multiplies the outputs of the selected experts by their respective gating weights
        # The result is a tensor where each expert's output is scaled by how much the gating network trusts that expert
        # Sum over expert dimension
        # This combines the outputs of all experts for each image + digit
        combined_outputs = torch.sum(weighted_outputs, dim=1)  # [1, num_classes]

        # Get prediction
        _, predicted = torch.max(combined_outputs.data, 1)
        predicted_digit = predicted.item()

    print(f'Predicted digit: {predicted_digit}')
    return predicted_digit

########################################################################################################################

def test():
    device, experts, gating_network, transform = get_device_experts_gate_transform(is_test=True)

    # TEST
    # Download and load the test data
    # Apply the graphic transform when loading the test data
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=MINI_BATCH_SIZE, shuffle=False)

    # Test the model
    # Always use TOP_K for testing/inference
    test_run(device, experts, gating_network, test_loader, TOP_K)

# TODO: Parallelize this more so each expert can work on a custom batch of images
#       it was selected for simultaneously
def test_run(
    device: torch.device,
    experts: List[nn.Module],
    gating_network: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    top_k: int):

    correct = 0
    total = 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            mini_batch_size = images.size(0)

            # Get gating outputs
            gating_outputs = gating_network(images)  # [mini_batch_size, num_experts]


            #### TOP_K EXPERTS ####

            # Get indices of top k experts for each image
            top_k_values, top_k_indices = torch.topk(gating_outputs, top_k, dim=1)  # [mini_batch_size, top_k]

            # Initialize expert outputs tensor
            expert_outputs = torch.zeros(mini_batch_size, top_k, 10).to(device)  # [mini_batch_size, top_k, num_classes]            

            # For each image in batch, run only its top k experts
            for image_i in range(mini_batch_size):
                for k_i, expert_i in enumerate(top_k_indices[image_i]):
                    # Process one image through one expert
                    output = experts[expert_i](images[image_i:image_i+1])  # [1, num_classes]
                    expert_outputs[image_i, k_i] = output

            # Normalize weights
            gating_weights = top_k_values.unsqueeze(2)  # [mini_batch_size, top_k, 1]
            gating_weights = gating_weights / gating_weights.sum(dim=1, keepdim=True)

            #######################


            # Weight and combine expert outputs
            weighted_outputs = expert_outputs * gating_weights  # [mini_batch_size, top_k, num_classes]

            # This is the element-wise multiplication of the expert outputs and the gating weights
            combined_outputs = torch.sum(weighted_outputs, dim=1)  # [mini_batch_size, num_classes]

            # Get predictions
            _, predicted = torch.max(combined_outputs.data, 1)
            total += labels.size(0)
            correct += torch.eq(predicted, labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

########################################################################################################################

def model_weights_file() -> str:
    return 'moe_model_weights-' + MODEL_VERSION

    # # Load models (for inference)
    # experts, gating_network = load_models(device, INPUT_SIZE, NUM_EXPERTS)
    #
    # # Test
    # inference(device, experts, gating_network, test_dataset)

def save_models(
    experts: List[nn.Module],
    gating_network: nn.Module,
    save_dir: str = './models'):

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save gating network
    torch.save(gating_network.state_dict(), f'{save_dir}/{model_weights_file()}_gate.pth')

    # Save each expert
    for i, expert in enumerate(experts):
        torch.save(expert.state_dict(), f'{save_dir}/{model_weights_file()}_expert_{i}.pth')

    print(f'Models saved to {save_dir}')

def load_models(
    input_size: int,
    num_experts: int,
    device: torch.device,
    save_dir: str = './models') -> Tuple[List[nn.Module], nn.Module]:

    # Create new model instances
    gating_network = GatingNetwork(input_size, num_experts)
    experts = [Expert(input_size) for _ in range(num_experts)]

    # Load saved weights
    gating_network.load_state_dict(torch.load(f'{save_dir}/{model_weights_file()}_gate.pth'))
    gating_network = gating_network.to(device)

    for i in range(num_experts):
        experts[i].load_state_dict(torch.load(f'{save_dir}/{model_weights_file()}_expert_{i}.pth'))
        experts[i] = experts[i].to(device)

    print(f'Models loaded from {save_dir}')
    return experts, gating_network

def load_blank_models(
    input_size: int,
    num_experts: int,
    device: torch.device) -> Tuple[List[nn.Module], nn.Module]:

    experts = [Expert(input_size) for _ in range(num_experts)]
    gating_network = GatingNetwork(input_size, num_experts)

    # Move models to GPU
    gating_network = gating_network.to(device)
    experts = [expert.to(device) for expert in experts]

    return experts, gating_network

########################################################################################################################

def show(name, ftn):
    print(name + "...")
    ftn()
    print("done.")

########################################################################################################################

# Hyperparameters
MODEL_VERSION = "1.0.5.k"
NUM_EXPERTS = 6
TOP_K = 3
GATING_HIDDEN_LAYER_SIZE = 64
LEARNING_RATE = 0.05
EPOCHS = 10
MINI_BATCH_SIZE = 16
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# 0 = train and test
# 1 = test
# 2 = inference
run = 0

def main():
    if NUM_EXPERTS < 1 or TOP_K < 1 or TOP_K > NUM_EXPERTS:
        print("Invalid configuration. Ensure NUM_EXPERTS >= 1 and 1 <= TOP_K <= NUM_EXPERTS.")
        return

    if run == 0:
        show("training", train)
        show("testing", test)
    elif run == 1:
        show("testing", test)
        test()
    elif run == 2:
        show("inference", inference)
    else:
        print("Invalid run option. Use 0 for training or 1 for inference.")
        return

if __name__ == "__main__":
    main()
