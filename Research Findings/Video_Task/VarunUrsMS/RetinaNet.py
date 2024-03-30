import cv2
import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import ToTensor
from PIL import Image

# Load the RetinaNet model
model = retinanet_resnet50_fpn(pretrained=True)
model = model.eval()  # Set the model to evaluation mode

# Define the transforms
transform = ToTensor()

# Open the video file
video_path = "people.mp4"  # Replace with the actual path to your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the output video
output_path = "output.mp4"  # Replace with the desired output file path
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Specify the codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create an OpenCV window
# cv2.namedWindow("Video with Bounding Boxes", cv2.WINDOW_NORMAL)

# Function to process each frame and draw bounding boxes
def process_frame(frame, frame_num):
    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply transforms
    img_tensor = transform(img)

    # Add a batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Pass the image through the model
    with torch.no_grad():
        predictions = model(img_tensor)

    # Get the predicted bounding boxes, labels, and scores
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Draw bounding boxes on the frame
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.7:  # Set a threshold for displaying bounding boxes
            box = box.int()
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)
            cv2.putText(frame, f"Label: {label}, Score: {score:.2f}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame with bounding boxes
#     cv2.imshow("Video with Bounding Boxes", frame)
    out.write(frame)  # Write the frame to the output video
    
    # Calculate and display the percentage completion
    percent_complete = (frame_num / total_frames) * 100
    print(f"Saving video: {percent_complete:.2f}% complete")

# Process each frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process the frame
    process_frame(frame, frame_num)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

    frame_num += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
