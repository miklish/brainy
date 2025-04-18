import torch
import torch.nn as nn
import torchvision
import PIL.Image as Image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import string
from typing import List
from typing import Tuple

# THE EXPERT NETWORK
# - Defines the forward pass (training / inference)
# - Inherits from torch.nn.Module
# - NOTE: Instantiate the model with
#     expert = Expert().to(device) # puts this on GPU
# - Calling
#     outputs = expert(inputs)
#   will automatically run the forward pass
# - Do not call forward() directly
#   - forward() is called on all images of a mini_batch
#
# The Expert class
#   The Expert class defines a neural network model that consists of two fully connected layers.
# Initialization
#   - fc1: The first fully connected layer, which takes an input of size 28*28 (flattened image) and outputs 500 nodes.
#   - fc2: The second fully connected layer, which takes the 500 nodes from fc1 and outputs 1 node.
# Forward Pass (forward method)
#   - Flattening: The input image is flattened from a 2D tensor of size 28*28 to a 1D tensor of size 784.
#   - First Layer: The flattened input is passed through the first fully connected layer (fc1) and a ReLU activation function.
#   - Second Layer: The output of the first layer is passed through the second fully connected layer (fc2).
# Output
#   The output of the Expert class is a logit (unactivated output) of size 1 for each input image.
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
        x = x.view(-1, INPUT_SIZE)  # Flatten the image -> [1, 784]

        # Apply the first fully connected layer and ReLU (or GReLU) activation function
        x = self.fc1(x)     # -> [1, 500]
        x = torch.relu(x)   # -> [1, 500]

        # Do calcs for hidden->output layers
        # We don't apply activation in the fc network containing the output layer since it
        # will be included in the loss function (specifically: cross entropy loss)
        x = self.fc2(x)     # -> [1, 10]

        # Return the "logit" (non-softmax'd probability distribution over 10 digits)
        return x    # [1, 10]

# THE GATING MODEL
# Defines the forward pass (training / inference)
# Inherits from torch.nn.Module
# NOTE: Instantiate the mode with
#   gating_network = GatingNetwork().to(device) # puts this on GPU
# Calling
#   outputs = gating_network(inputs)
# wil automatically run the forward pass
# Do not call forward() directly
#
# GatingNetwork class
#   This class initializes two fully connected layers:
#     - fc1: Takes the flattened input image (size 28*28) and outputs GATING_HIDDEN_LAYER_SIZE nodes
#     - fc2: Takes the GATING_HIDDEN_LAYER_SIZE nodes from fc1 and outputs a "logit" for each expert
#       (NUM_EXPERTS experts in this case)
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
#   During training, ALL experts are activated
#
#   The outputs of the experts are combined by weighting the expert outputs by these probabilities
#   and summed to make the final prediction
#
# How the experts are used during inference
#   As in training, during inference, the GatingNetwork outputs a probability distribution over the experts for each
#   input image
#
#   However now, we only use the top k experts -- determined by the largest (top) k probabilities generated by the
#   gating network
#
#   The outputs of the selected experts are combined by weighting the expert outputs by their respective probabilities
#   and summed to make the final prediction
#
#   This allows the model to focus on the most relevant experts for each input image
#   The gating network's output is a probability distribution over the experts for each input image
#   The top k experts are selected based on the gating network's output
#   The outputs of the selected experts are combined by weighting the expert outputs by their respective probabilities
#   and summed to make the final prediction
#
class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()

        # Create the first fully connected layers (input->hidden layer)
        # Input layer is 28x28 inputs nodes (image size) and output layer is GATING_HIDDEN_LAYER_SIZE nodes
        self.fc1 = nn.Linear(input_size, GATING_HIDDEN_LAYER_SIZE)

        # Second fully connected layer: creates a logit
        # (unactivated output) for each expert
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

    # Combine parameters from gating network and experts
    all_parameters = list(gating_network.parameters()) + [p for expert in experts for p in expert.parameters()]

    # Create SGD optimizer with momentum and weight decay
    optimizer = torch.optim.SGD(all_parameters, lr=LEARNING_RATE)

    # Train the mixture of experts
    train_mixture_of_experts(
        device, experts, gating_network, loss_ftn, optimizer, training_minibatch_loader,
        epochs=EPOCHS, num_experts=NUM_EXPERTS, top_k=TOP_K)

    # Save models
    save_models(experts, gating_network)

# def train_mixture_of_experts(
#     device: torch.device,
#     experts: List[nn.Module],
#     gating_network: torch.nn.Module,
#     loss_ftn: torch.nn.CrossEntropyLoss,
#     optimizer: torch.optim.Optimizer,
#     training_minibatch_loader: torch.utils.data.DataLoader,
#     epochs: int,
#     num_experts: int,
#     top_k: int):
#
#     current_top_k: int
#     for epoch in range(epochs):
#         epoch_loss = 0.0
#         batch_count = 0
#
#         current_top_k = max(top_k, num_experts - epoch)     # Decrease top_k each epoch until reaching TOP_K
#         print(f'Epoch {epoch + 1}, using top-{current_top_k} experts')
#
#         for image_minibatch, label_minibatch in training_minibatch_loader:
#             # Move data to device
#             image_minibatch = image_minibatch.to(device)
#             label_minibatch = label_minibatch.to(device)
#
#             # Zero gradients
#             optimizer.zero_grad()
#
#             # Forward pass through gating network
#             gating_outputs = gating_network(image_minibatch) # Shape [mini_batch_size, num_experts]
#
#             # Get top k experts using current_top_k
#             _, top_k_indices = torch.topk(gating_outputs, current_top_k, dim=1)
#
#             # Create mask for top k experts
#             mask = torch.zeros_like(gating_outputs)
#             ones = torch.ones_like(top_k_indices, dtype=mask.dtype)
#             mask.scatter_(dim=1, index=top_k_indices, src=ones)
#
#             #### TOP_K EXPERTS ####
#
#             # Renormalize gating output to sum to 1 for the top current_top_k experts
#             gating_outputs = gating_outputs * mask  # Zero out non-top-k expert weights - pairwise multiplication
#             gating_outputs = gating_outputs / gating_outputs.sum(dim=1, keepdim=True) # Renormalize gating weights to sum to 1
#
#             # Initialize a list to hold the outputs from each expert for each image
#             expert_outputs_list = []
#
#             ####################
#
#             # Get top k experts for each image based on the gating network's probability distribution
#             # - gating_outputs: [mini_batch_size, num_experts]
#             _, top_k_indices = torch.topk(input=gating_outputs, k=TOP_K, dim=1)
#
#             # Create mask for top k experts
#             mask = torch.zeros_like(gating_outputs) # set to 0s
#             ones = torch.ones_like(top_k_indices, dtype=mask.dtype) # create 1s at top_k_indices indexes
#             mask.scatter_(dim=1, index=top_k_indices, src=ones) # change mask's 0s to 1s at top_k_indices indexes
#
#             # Zero out non-top-k expert weights
#             gating_outputs = gating_outputs * mask  # pairwise multiplication
#
#             # Renormalize gating weights to sum to 1
#             # Need to do this since we break the softmax normalization by selecting only the top k experts
#             # - gating_outputs: [mini_batch_size, num_experts]
#             gating_outputs = gating_outputs / gating_outputs.sum(dim=1, keepdim=True)
#
#             #######################
#
#
#             # Initialize a list to hold the outputs from each expert
#             top_k_expert_outputs_list = []
#
#             # Forward pass through experts
#
#             # Get outputs from the current_top_k experts using a loop
#             for i in range(len(experts)):
#
#                 # Skip experts that are not in the top k
#                 if mask[i] == 0:
#                     continue
#
#                 # if
#                 #   e_i, e_j are top k experts
#                 # then if
#                 #   e_i = experts[i] and e_j = experts[j]
#                 # then if
#                 #   e_i = expert_outputs[a] and e_j = expert_outputs[b]
#                 # then
#                 #   a < b iff i < j
#
#                 # Get output for all images in batch from current expert
#                 top_k_expert_output = experts[i](image_minibatch)  # Shape: [mini_batch_size, num_classes]
#                 top_k_expert_output = top_k_expert_output.unsqueeze(1)   # Shape: -> [mini_batch_size, expert_index, num_classes]
#                 top_k_expert_outputs_list.append(top_k_expert_output)
#
#             # Concatenate the top k expert outputs along the expert_index dimension
#             # - this means to expand dim=1 (the expert_index dimension) from a dimension of 1 to a dimension of size TOP_K
#             # expert_outputs[i][e][d] contains the logit (unactivated output) that expert e predicts for digit class d when looking at image i in the batch
#             top_k_expert_outputs = torch.cat(top_k_expert_outputs_list, dim=1)  # Shape: [mini_batch_size, num_experts, num_classes]
#
#             # Combine outputs
#             # Shape [mini_batch_size, num_experts]
#
#             # Shape [mini_batch_size, num_experts] -> [mini_batch_size, num_experts, 1]
#             gate_confidence_of_expert_e_for_image_i = gating_outputs.unsqueeze(2)
#
#             # [mini_batch_size, num_experts, num_classes] * [mini_batch_size, num_experts, 1]
#             # Perform element-wise multiplication
#             #
#             # expert_outputs: [mini_batch_size, num_experts, num_classes]
#             # - expert_outputs[i][e][d] = the logit (unactivated output) produced by expert e for digit d in image i
#             #
#             # gate_confidence_of_expert_e_for_image_i: [mini_batch_size, num_experts, 1]
#             # - gate_confidence_of_expert_e_for_image_i[i][e][1]
#             #
#             # weighted_expert_outputs[i][e][d]
#             #   = expert_outputs[i][e][d] * gate_confidence_of_expert_e_for_image_i[i][e][1]
#             # weighted_expert_outputs: [mini_batch_size, num_experts, num_classes]
#             #
#             # This multiplies all 10 logits (one for each digit) produced by expert e for image i by the gating
#             # network's confidence that expert e is the 'best' expert for image i
#             #
#             # For each image i and expert e:
#             #   - expert_outputs[i][e] is a vector of 10 logits (one for each digit)
#             #   - gate_confidence_of_expert_e_for_image_i[i][e][0] is a single confidence value
#             #   - The multiplication broadcasts the confidence value across all 10 logits, scaling each logit by how
#             #      much the gating network trusts that expert for that image
#             #
#             # This weighting means that if the gating network has high confidence in expert e for image i,
#             # then that expert's predictions will contribute more strongly to the final combined output for that image.
#             #
#             # Tensor broadcasting
#             #   In PyTorch, broadcasting refers to the automatic expansion of tensor dimensions to make them compatible
#             #   for element-wise operations. In this specific context:
#             #
#             #   1. gate_confidence_of_expert_e_for_image_i has shape [mini_batch_size, num_experts, 1]
#             #   2. expert_outputs has shape [mini_batch_size, num_experts, 10]
#             #
#             #   When multiplying these tensors, the 1 in the last dimension of gate_confidence_of_expert_e_for_image_i
#             #   is automatically "broadcast" (expanded) to match the 10 in expert_outputs. This means:
#             #
#             #   1. The single confidence value for each expert/image pair is automatically repeated 10 times
#             #   2. Each of the 10 logits from an expert for an image is multiplied by the same confidence value
#             #   3. No additional memory is actually allocated for this expansion
#             #
#             #   For example, if we have:
#             #     gate_confidence_of_expert_e_for_image_i[0][0][0] = 0.7  # confidence for expert 0, image 0
#             #     expert_outputs[0][0] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # logits for digits 0-9
#             #
#             #   The multiplication will effectively do:
#             #     weighted_outputs[0][0] = [0.1*0.7, 0.2*0.7, 0.3*0.7, 0.4*0.7, 0.5*0.7, 0.6*0.7, 0.7*0.7, 0.8*0.7, 0.9*0.7, 1.0*0.7]
#             #
#             weighted_outputs = top_k_expert_outputs * gate_confidence_of_expert_e_for_image_i
#
#             # Sum over expert dimension
#             #
#             #   # This combines the outputs of all experts for each image + digit
#             #   For each image i and digit d:
#             #     combined_outputs[i][d] = Sum[ e = 1..6 ]( weighted_expert_outputs[i][e][d] )
#             #
#             combined_outputs = torch.sum(weighted_outputs, dim=1) # Shape [mini_batch_size, num_classes]: (i, d)
#
#             # Compute the loss
#             # - combined_outputs:
#             #       [mini_batch_size, num_classes] ~ [image_i, digit]
#             # - labels:
#             #       [mini_batch_size]              ~ [image_i_label]
#             loss = loss_ftn(combined_outputs, label_minibatch)
#
#             # Accumulate loss for logging
#             epoch_loss += loss.item()
#             batch_count += 1
#
#             # Backward pass
#             loss.backward()
#
#             # Update weights
#             optimizer.step()
#
#         avg_loss = epoch_loss / batch_count
#         print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

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

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0

        current_top_k = max(top_k, num_experts - epoch)  # Decrease top_k each epoch until reaching TOP_K
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
            gating_outputs = gating_outputs * mask

            # sum all values along dim=1 (the experts dimension) for each sample in the batch
            # keepdim=True preserves the dimensionality, resulting in shape [mini_batch_size, 1]
            gating_outputs_sum = gating_outputs.sum(dim=1, keepdim=True) # Shape [mini_batch_size, 1]

            # renormalize gating_outputs to ensure the gating weights for the top-k experts sum to 1
            # for each sample in the mini batch

            # [mini_batch_size, num_experts] / [mini_batch_size, 1]
            # broadcast the second dimension of gating_outputs_sum from 1 to num_experts
            # by repeating the value in dim=1 num_experts times
            gating_outputs_k = gating_outputs / gating_outputs_sum # Shape [mini_batch_size, num_experts]

            # Initialize tensor to hold all expert outputs
            # The shape of e_outputs is [mini_batch_size, num_experts, 10]
            e_outputs = torch.zeros(mini_batch_size, num_experts, 10).to(device)  # 10 classes for MNIST

            # For each expert, find which images need it
            for e in range(num_experts):
                # Find images that selected this expert in their top-k

                '''
                This line is identifying which images in the batch need to be processed by the current expert e.
                                
                1. mask is a tensor of shape [mini_batch_size, num_experts] where each element is either 0 or 1:                            
                
                  - If mask[i, e] = 1, image i selected expert e in its top-k
                  - If mask[i, e] = 0, image i did not select expert e in its top-k
                
                2. mask[:, e] selects the column for the current expert e, giving a 1D tensor of shape
                   [mini_batch_size]. This contains 1s for images that need this expert and 0s otherwise.                
                
                3. .nonzero() returns the indices where the value is non-zero (i.e., equal to 1). This gives a
                   2D tensor of shape [num_images_needing_expert, 1] where each row contains an index.
                                
                4. .squeeze(dim=1) removes the second dimension (which is just a singleton dimension),
                   resulting in a 1D tensor of shape [num_images_needing_expert] containing the indices of
                   images that need expert e.                
                
                This approach is efficient because it allows the model to only run each expert on the subset of
                images that actually selected that expert in their top-k, rather than running every expert on
                every image.                
                '''
                indexes_of_images_needing_expert_e = mask[:, e].nonzero().squeeze(dim=1)

                # Only process if at least one image needs this expert
                if len(indexes_of_images_needing_expert_e) > 0:
                    # Create mini-batch of just those images
                    e_minibatch = image_minibatch[indexes_of_images_needing_expert_e]

                    # Run expert on just those images
                    outputs = experts[e](e_minibatch)

                    '''
                    Place outputs in the correct positions in our result tensor
                    Code below is equivalent to:

                    # For each image that needs this expert
                    # e_outputs Shape: [mini_batch_size, num_experts, 10]
                    # e_outputs[i, j] Shape: [10]
                    # e_outputs is initially fully zeroed

                    outputs has shape [num_images_needing_expert_e, 10]
                    outputs[i] has shape [10]

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
    image: Image
    image = Image.open(image_path)

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
        # The gating weights are broadcasted to match the shape of the expert outputs
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

    # TODO
    # Test the model
    # Always use TOP_K for testing/inference
    test_run(device, experts, gating_network, test_loader, TOP_K)

def test_run(device, experts, gating_network, test_loader, top_k):
    correct = 0
    total = 0

    # Disable gradient computation for inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            mini_batch_size = images.size(0)

            # # Initialize expert outputs tensor
            # expert_outputs = torch.zeros(images.size(0), top_k, 10).to(device)  # [GATING_HIDDEN_LAYER_SIZE, top_k, num_classes]

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

def save_models(experts, gating_network, save_dir='./models'):
    import os

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save gating network
    torch.save(gating_network.state_dict(), f'{save_dir}/{model_weights_file()}_gate.pth')

    # Save each expert
    for i, expert in enumerate(experts):
        torch.save(expert.state_dict(), f'{save_dir}/{model_weights_file()}_expert_{i}.pth')

    print(f'Models saved to {save_dir}')

def load_models(input_size, num_experts, device, save_dir='./models'):
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

def load_blank_models(input_size, num_experts, device):
    experts = [Expert(input_size) for _ in range(num_experts)]
    gating_network = GatingNetwork(input_size, num_experts)

    # Move models to GPU
    gating_network = gating_network.to(device)
    experts = [expert.to(device) for expert in experts]

    return experts, gating_network

########################################################################################################################

# Hyperparameters
NUM_EXPERTS = 6
GATING_HIDDEN_LAYER_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5
MINI_BATCH_SIZE = 64
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
INPUT_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
MODEL_VERSION = "1.0.0.k"
TOP_K = 3

def main():
    test()

# def main():
#     train()

if __name__ == "__main__":
    main()
