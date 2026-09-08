import cv2
import torch
import time
import numpy as np
from PIL import Image
from torchvision import transforms
import pyvirtualcam
from models.unet import UNet
from config.config import IMAGE_SIZE
MODEL_PATH = "checkpoints/modelV5.pth"
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)
model = UNet().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded.")
transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])
# ==============================
# Open Webcam
# ==============================

camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30
virtual_cam = pyvirtualcam.Camera(
    width=width,
    height=height,
    fps=fps
)

print("Virtual Camera Started")
if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. "
        "Check camera permissions or try another camera index."
    )

print("Webcam opened.")
print("Press Q to quit.")


# ==============================
# Webcam Loop
# ==============================

while True:

    # Read one frame
    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Save original size
    original_height, original_width = frame.shape[:2]

    # Convert OpenCV BGR -> RGB
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL
    image = Image.fromarray(frame_rgb)

    # Convert to tensor
    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)
    input_mean = input_tensor.mean().item()

    print(f"Brightness: {input_mean:.3f}")

    # (Next we'll run the model here)
    with torch.no_grad():

        if input_mean < 0.150:

            prediction = model(input_tensor)

        else:

            prediction = input_tensor.clone()

    prediction = prediction.squeeze(0)
    prediction = torch.clamp(
        prediction,
        0,
        1
    )

    output_image = transforms.ToPILImage()(
        prediction.cpu()
    )

    prediction = torch.clamp(
        prediction,
        0,
        1
    )

    output_image = transforms.ToPILImage()(
        prediction.cpu()
    )
    output_image = output_image.resize(
        (original_width, original_height)
    )
    output_frame = cv2.cvtColor(
        np.array(output_image),
        cv2.COLOR_RGB2BGR
    )
    cv2.imshow(
        "Original Webcam",
        frame
    )

    cv2.imshow(
        "Enhanced Webcam",
        output_frame
    )
    rgb_frame = cv2.cvtColor(
        output_frame,
        cv2.COLOR_BGR2RGB
    )

    virtual_cam.send(rgb_frame)
    virtual_cam.sleep_until_next_frame()
        

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
print("Webcam Mean:", input_tensor.mean().item())

# ==============================
# Cleanup
# ==============================

camera.release()

virtual_cam.close()
camera.release()
cv2.destroyAllWindows()

print("Webcam closed.")