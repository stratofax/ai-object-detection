import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


def people_detected(image_path):

    # Load the pre-trained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()

    # Define the transform to convert the image to the format required by the model
    transform = weights.transforms()

    # Load your image
    image = Image.open(image_path)
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Perform the detection
    with torch.no_grad():
        predictions = model(image_tensor)

    # Get the predictions
    pred_scores = predictions[0]["scores"]
    pred_labels = predictions[0]["labels"]

    # Set a confidence threshold
    confidence_threshold = 0.8

    # Determine if a person is detected
    return any(
        score > confidence_threshold and label == 1
        for score, label in zip(pred_scores, pred_labels)
    )
