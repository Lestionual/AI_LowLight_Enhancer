import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pyvirtualcam
from models.unet import UNet
from config.config import IMAGE_SIZE
import time
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage


class CameraWorker(QObject):

    frame_ready = Signal(QImage)
    enhanced_frame_ready = Signal(QImage)
    error = Signal(str)
    finished = Signal()
    fps_updated = Signal(float)

    def __init__(self, camera_index=0):

        super().__init__()
        self.threshold = 0.15
        self.camera_index = camera_index
        self.running = False
        
        from utils.path_utils import resource_path

        MODEL_PATH = resource_path(
            "checkpoints/modelV5.pth"
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        print("Using:", self.device)

        self.model = UNet().to(self.device)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print("Model loaded.")

        # IDENTICAL TO WEBCAM.PY
        self.transform = transforms.Compose([
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor()
        ])

    @Slot()
    @Slot(float)
    def set_threshold(self, value):

        self.threshold = value

        print(f"Threshold: {value:.2f}")
    def run(self):

        self.running = True

        backends = [
            ("MSMF", cv2.CAP_MSMF),
            ("DSHOW", cv2.CAP_DSHOW),
            ("ANY", cv2.CAP_ANY),
        ]

        camera = None

        for backend_name, backend in backends:

            print(f"Trying backend: {backend_name}")

            cap = cv2.VideoCapture(
                self.camera_index,
                backend
            )

            if cap.isOpened():

                success, frame = cap.read()

                if success:

                    print(f"Using backend: {backend_name}")

                    camera = cap
                    break

            cap.release()

        if camera is None:

            self.error.emit(
                "Could not open the selected camera."
            )

            self.finished.emit()
            return
        virtual_cam = None

        if not camera.isOpened():

            self.error.emit(
                "Could not open webcam."
            )

            self.finished.emit()
            return

        print("Webcam opened.")
  
        previous_time = time.time()

        while self.running:

            success, frame = camera.read()

            if not success:

                self.error.emit(
                    "Could not read webcam frame."
                )

                break

            original_height, original_width = frame.shape[:2]
            if virtual_cam is None:

                virtual_cam = pyvirtualcam.Camera(
                    width=original_width,
                    height=original_height,
                    fps=30
                )

                print("Virtual Camera:", virtual_cam.device)
            # =====================================
            # ORIGINAL FRAME (GUI)
            # =====================================

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            h, w, c = frame_rgb.shape

            original_qimage = QImage(
                frame_rgb.data,
                w,
                h,
                c * w,
                QImage.Format_RGB888
            ).copy()

            self.frame_ready.emit(
                original_qimage
            )

            # =====================================
            # PREPROCESS (IDENTICAL TO webcam.py)
            # =====================================

            image = Image.fromarray(
                frame_rgb
            )

            input_tensor = self.transform(
                image
            )

            input_tensor = input_tensor.unsqueeze(0)

            input_tensor = input_tensor.to(
                self.device
            )

            input_mean = input_tensor.mean().item()

            # Uncomment for debugging
            # print(f"Brightness: {input_mean:.3f}")

            # =====================================
            # MODEL
            # =====================================

            with torch.no_grad():

                if input_mean < self.threshold:

                    prediction = self.model(
                        input_tensor
                    )

                else:

                    prediction = input_tensor.clone()

            # =====================================
            # POSTPROCESS (IDENTICAL TO webcam.py)
            # =====================================

            prediction = prediction.squeeze(0)

            prediction = torch.clamp(
                prediction,
                0,
                1
            )

            output_image = transforms.ToPILImage()(
                prediction.cpu()
            )

            output_image = output_image.resize(
                (
                    original_width,
                    original_height
                )
            )

            # Save for comparison if needed
            # output_image.save("gui_output.png")

            rgb_output = np.array(
                output_image
            )

            h, w, c = rgb_output.shape

            enhanced_qimage = QImage(
                rgb_output.data,
                w,
                h,
                c * w,
                QImage.Format_RGB888
            ).copy()

            self.enhanced_frame_ready.emit(
                enhanced_qimage
            )
            virtual_cam.send(rgb_output)

            virtual_cam.sleep_until_next_frame()
            current_time = time.time()

            fps = 1.0 / (current_time - previous_time)

            previous_time = current_time

            self.fps_updated.emit(fps)

        camera.release()

        print("Webcam released.")

        self.running = False

        self.finished.emit()

    def stop(self):

        self.running = False