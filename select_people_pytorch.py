import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load a pre-trained model from torchvision
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define the transform to convert the image to the format required by the model
transform = T.Compose([T.ToTensor()])

# Load your image
image_path = "/Volumes/ExtSSD/Media/NHCC/slideshow/selected/20230730_105331.jpg"
image = Image.open(image_path)
image_tensor = transform(image)

# Add batch dimension
image_tensor = image_tensor.unsqueeze(0)

# Perform the detection
with torch.no_grad():
    predictions = model(image_tensor)

# Get the predictions
pred_boxes = predictions[0]["boxes"]
pred_scores = predictions[0]["scores"]
pred_labels = predictions[0]["labels"]

# Set a confidence threshold
confidence_threshold = 0.8

# Convert the image to OpenCV format for display
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Loop over the predictions and draw bounding boxes for people
# (label 1 in COCO dataset)
for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
    if score > confidence_threshold and label == 1:  # Person class label is 1
        startX, startY, endX, endY = box.int().numpy()
        cv2.rectangle(image_cv, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the output image
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
