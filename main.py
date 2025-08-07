import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QProgressBar, QFrame, QSplitter, QFileDialog,
                             QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QFont
import cv2
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

class ImageUpscalerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, input_path, output_path, scale_factor):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.scale_factor = scale_factor
    
    def run(self):
        try:
            self.progress.emit(5)
            
            image = cv2.imread(self.input_path, cv2.IMREAD_COLOR)
            if image is None:
                self.error.emit("Could not load image")
                return
            
            self.progress.emit(15)
            
            if self.scale_factor == 2:
                model_name = 'RealESRGAN_x2plus'
                model_scale = 2
            elif self.scale_factor == 4:
                model_name = 'RealESRGAN_x4plus'
                model_scale = 4
            else:
                model_name = 'RealESRGAN_x4plus'
                model_scale = 4
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_scale)
            
            self.progress.emit(25)
            
            upsampler = RealESRGANer(
                scale=model_scale,
                model_path=f'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{model_name}.pth',
                dni_weight=None,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            
            self.progress.emit(40)
            
            if self.scale_factor == 3:
                output, _ = upsampler.enhance(image, outscale=4)
                self.progress.emit(70)
                
                height, width = image.shape[:2]
                target_size = (width * 3, height * 3)
                output = cv2.resize(output, target_size, interpolation=cv2.INTER_LANCZOS4)
            else:
                output, _ = upsampler.enhance(image, outscale=self.scale_factor)
            
            self.progress.emit(80)
            
            cv2.imwrite(self.output_path, output)
            
            self.progress.emit(100)
            self.finished.emit(self.output_path)
            
        except Exception as e:
            self.error.emit(f"Error during upscaling: {str(e)}")

class ImagePreviewWidget(QLabel):
    def __init__(self, title="Preview"):
        super().__init__()
        self.title = title
        self.setMinimumSize(300, 300)
        self.setMaximumSize(600, 600)
        self.setScaledContents(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f5f5f5;
                color: #666;
                font-size: 14px;
            }
        """)
        self.setText(f"{title}\n(No image loaded)")
    
    def set_image(self, image_path):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                self.setStyleSheet("""
                    QLabel {
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        background-color: white;
                    }
                """)
    
    def clear_image(self):
        self.clear()
        self.setText(f"{self.title}\n(No image loaded)")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f5f5f5;
                color: #666;
                font-size: 14px;
            }
        """)

class DragDropWidget(QFrame):
    file_dropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet("""
            QFrame {
                border: 3px dashed #2196F3;
                border-radius: 15px;
                background-color: #E3F2FD;
                color: #1976D2;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        
        layout = QVBoxLayout()
        self.label = QLabel("Drag & Drop Image Here\nor Click to Browse")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)
        self.setLayout(layout)
        
        self.mousePressEvent = self.browse_file
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                file_path = urls[0].toLocalFile()
                if self.is_image_file(file_path):
                    event.acceptProposedAction()
                    self.setStyleSheet("""
                        QFrame {
                            border: 3px solid #4CAF50;
                            border-radius: 15px;
                            background-color: #E8F5E8;
                            color: #2E7D2E;
                            font-size: 16px;
                            font-weight: bold;
                        }
                    """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 3px dashed #2196F3;
                border-radius: 15px;
                background-color: #E3F2FD;
                color: #1976D2;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files and self.is_image_file(files[0]):
            self.file_dropped.emit(files[0])
        
        self.setStyleSheet("""
            QFrame {
                border: 3px dashed #2196F3;
                border-radius: 15px;
                background-color: #E3F2FD;
                color: #1976D2;
                font-size: 16px;
                font-weight: bold;
            }
        """)
    
    def browse_file(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Image File",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file_path:
            self.file_dropped.emit(file_path)
    
    def is_image_file(self, file_path):
        extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
        return file_path.lower().endswith(extensions)

class ImageUpscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_input_path = None
        self.upscaler_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("AI Image Upscaler")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        title_label = QLabel("@TAKTOPYTHON - AI Image Upscaler")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: #2E7D2E; margin: 20px;")
        main_layout.addWidget(title_label)
        
        self.drag_drop_widget = DragDropWidget()
        self.drag_drop_widget.file_dropped.connect(self.load_image)
        main_layout.addWidget(self.drag_drop_widget)
        
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Upscale Factor:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "3x", "4x"])
        self.scale_combo.setCurrentText("2x")
        controls_layout.addWidget(self.scale_combo)
        
        controls_layout.addStretch()
        
        self.process_btn = QPushButton("Upscale Image")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.upscale_image)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        controls_layout.addWidget(self.process_btn)
        
        self.save_btn = QPushButton("Save Result")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        controls_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(controls_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        preview_layout = QHBoxLayout()
        
        self.before_preview = ImagePreviewWidget("Original Image")
        preview_layout.addWidget(self.before_preview)
        
        self.after_preview = ImagePreviewWidget("Upscaled Image")
        preview_layout.addWidget(self.after_preview)
        
        main_layout.addLayout(preview_layout)
        
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Drop an image to get started")
    
    def load_image(self, file_path):
        if os.path.exists(file_path):
            self.current_input_path = file_path
            self.before_preview.set_image(file_path)
            self.after_preview.clear_image()
            self.process_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}")
    
    def upscale_image(self):
        if not self.current_input_path:
            return
        
        scale_text = self.scale_combo.currentText()
        scale_factor = int(scale_text.replace('x', ''))
        
        input_path = Path(self.current_input_path)
        temp_output_path = input_path.parent / f"{input_path.stem}_upscaled_{scale_text}{input_path.suffix}"
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.process_btn.setEnabled(False)
        self.status_bar.showMessage("Upscaling image...")
        
        self.upscaler_thread = ImageUpscalerThread(
            self.current_input_path,
            str(temp_output_path),
            scale_factor
        )
        self.upscaler_thread.progress.connect(self.progress_bar.setValue)
        self.upscaler_thread.finished.connect(self.upscaling_finished)
        self.upscaler_thread.error.connect(self.upscaling_error)
        self.upscaler_thread.start()
    
    def upscaling_finished(self, output_path):
        self.after_preview.set_image(output_path)
        self.current_output_path = output_path
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("Upscaling completed successfully!")
    
    def upscaling_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_bar.showMessage("Upscaling failed")
        QMessageBox.critical(self, "Error", error_message)
    
    def save_result(self):
        if hasattr(self, 'current_output_path') and os.path.exists(self.current_output_path):
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Upscaled Image",
                "",
                "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            
            if save_path:
                try:
                    # Copy the temporary file to the selected location
                    import shutil
                    shutil.copy2(self.current_output_path, save_path)
                    self.status_bar.showMessage(f"Saved: {os.path.basename(save_path)}")
                    QMessageBox.information(self, "Success", f"Image saved successfully to:\n{save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")

def main():
    app = QApplication(sys.argv)
    window = ImageUpscalerApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()