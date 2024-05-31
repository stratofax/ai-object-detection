import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


def people_detected(image_path):
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


# Load the pre-trained model
weights_default = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights_default)
model.eval()

# Define the transform to convert the image to the format required by the model
transform = T.Compose([T.ToTensor()])

image_dir = "/Volumes/ExtSSD/Media/NHCC/slideshow/selected"

images = [
    "2022.9.12-Flat-topped-aster-1.jpg",
    "2022.9.12-Flat-topped-aster-2.jpg",
    "2022.9.12-Flat-topped-aster.jpg",
    "2022.9.12-Zigzag-Goldenrod-1.jpg",
    "2022.9.12-Zigzag-Goldenrod.jpg",
    "2022.9.23-New-England-aster-2-1.jpg",
    "2022.9.23-New-England-aster-2.jpg",
    "20230730_105331.jpg",
    "3.jpg",
    "364152569_326215643069265_5517925055997897645_n.jpg",
    "364797069_326215633069266_8727814195000133004_n-2.jpg",
    "6.19-1.jpeg",
]
# Load your image

for image_file in images:
    image_file = f"/Volumes/ExtSSD/Media/NHCC/slideshow/selected/{image_file}"

    print(f"People detected in {image_file}? {people_detected(image_file)}")
