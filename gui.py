import cv2
import sys
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap
from camera_worker import CameraWorker
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QComboBox
)

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        
        self.thread = None
        self.worker = None
        self.setWindowTitle("AI Low-Light Video Enhancer")
        self.resize(1200, 700)
        central = QWidget()
        self.setCentralWidget(central)
        self.camera_combo = QComboBox()
        main_layout = QVBoxLayout()

        # -----------------------------
        # Video Preview
        # -----------------------------

        preview_layout = QHBoxLayout()

        self.original_label = QLabel("Original Feed")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(500, 350)

        self.enhanced_label = QLabel("Enhanced Feed")
        self.enhanced_label.setAlignment(Qt.AlignCenter)
        self.enhanced_label.setMinimumSize(500, 350)

        preview_layout.addWidget(self.original_label)
        preview_layout.addWidget(self.enhanced_label)



        # -----------------------------
        # Threshold Slider
        # -----------------------------

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(15)
        self.threshold_slider.valueChanged.connect(
            self.threshold_changed
        )
        self.threshold_label = QLabel("0.15")
        self.fps_label = QLabel("FPS: 0.0")

        # -----------------------------
        # Buttons
        # -----------------------------

        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        # -----------------------------
        # Assemble Layout
        # -----------------------------

        main_layout.addLayout(preview_layout)
        main_layout.addWidget(QLabel("Camera"))
        main_layout.addWidget(self.camera_combo)
        main_layout.addWidget(QLabel("Low-Light Threshold"))
        main_layout.addWidget(self.threshold_slider)
        main_layout.addWidget(self.fps_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.threshold_label)
        central.setLayout(main_layout)
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.detect_cameras()

    def threshold_changed(self, value):

        threshold = value / 100.0

        self.threshold_label.setText(
            f"{threshold:.2f}"
        )

        if self.worker is not None:

            self.worker.set_threshold(
                threshold
            )
    def detect_cameras(self):

        self.camera_combo.clear()

        for index in range(5):

            cap = cv2.VideoCapture(index)

            if cap.isOpened():

                self.camera_combo.addItem(
                    f"Camera {index}",
                    index
                )

            cap.release()
    def update_fps(self, fps):

        self.fps_label.setText(
            f"FPS: {fps:.1f}"
        )
    def start_camera(self):

        if self.thread is not None:
            return

        self.thread = QThread()

        camera_index = self.camera_combo.currentData()

        self.worker = CameraWorker(
            camera_index
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.finished.connect(self.thread_finished)
        self.worker.frame_ready.connect(
            self.update_original_frame
        )
        self.worker.enhanced_frame_ready.connect(
            self.update_enhanced_frame
        )
        self.worker.fps_updated.connect(
            self.update_fps
        )
        self.worker.error.connect(
            self.handle_camera_error
        )
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def closeEvent(self, event):

        if self.worker is not None:
            self.worker.stop()

        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(2000)

        event.accept()
    def stop_camera(self):

        if self.worker:

            self.worker.stop()


    def thread_finished(self):

        self.thread = None

        self.worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print("Thread Finished")
        
    def update_original_frame(self, image):

        pixmap = QPixmap.fromImage(image)

        scaled_pixmap = pixmap.scaled(
            self.original_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.original_label.setPixmap(
            scaled_pixmap
        )
    def update_enhanced_frame(self, image):

        pixmap = QPixmap.fromImage(image)

        scaled_pixmap = pixmap.scaled(
            self.enhanced_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.enhanced_label.setPixmap(
            scaled_pixmap
        )

    def handle_camera_error(self, message):

        print("Camera error:", message)

        self.original_label.setText(
            message
        )
    


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec())