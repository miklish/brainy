import PIL.Image as Image
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import string

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
    transform = transforms.ToTensor()

    # Apply the transformation
    tensor_image = transform(image)  # Shape: (1, 28, 28)

    return tensor_image

def compare_mnist_user_imgs():
    transform: torchvision.transforms.Compose
    mnist_dataset: torchvision.datasets.MNIST
    image: Image
    image_tensor: torch.Tensor

    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Get the first image and label from the dataset
    image, label = mnist_dataset[0]

    # Convert the tensor to a numpy array and display it
    image_tensor = image.squeeze()

    print("Shape of the mnist image tensor:", image_tensor.shape)
    print("Size of the mnist image tensor:", image_tensor.size())

    print("Pixel values of MNIST after conversion:", list(image_tensor)[:784])

    plt.imshow(image_tensor, cmap='gray')
    plt.title(f'MNIST Image - Label: {label}')
    plt.show()

    ####

    # Load the user-created image
    image_tensor = load_image("digit0.png")

    # Convert the tensor to a numpy array and display it
    image_tensor = image_tensor.squeeze()

    print("Shape of the user image tensor:", image_tensor.shape)
    print("Size of the user image tensor:", image_tensor.size())

    print("Pixel values of USER after conversion:", list(image_tensor)[:784])

    plt.imshow(image_tensor, cmap='gray')
    plt.title('User Image')
    plt.show()