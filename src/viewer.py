import sys

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if __package__ in (None, ""):
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.pair_utils import build_pairs, load_image


class PairViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gursu Change Viewer")
        self.setGeometry(100, 100, 1250, 760)

        self.result_dir = (
            config.VERIFIED_MASKS_DIR
            if config.VERIFIED_MASKS_DIR.exists()
            and any(config.VERIFIED_MASKS_DIR.glob("pair_*.jpg"))
            else config.RAW_MASKS_DIR
        )
        self.result_files = sorted(self.result_dir.glob("pair_*.jpg"))
        self.pair_map = {pair.pair_id: pair for pair in build_pairs()}
        self.index = 0
        self._init_ui()

        if self.result_files:
            self.update_display()
        else:
            self.info_label.setText("No result images found.")

    def _init_ui(self):
        self.left_label = QLabel()
        self.right_label = QLabel()
        for widget in [self.left_label, self.right_label]:
            widget.setFixedSize(560, 560)
            widget.setStyleSheet("border: 2px solid #555; background: #111;")

        self.info_label = QLabel("Loading...")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        prev_button = QPushButton("Prev")
        next_button = QPushButton("Next")
        prev_button.clicked.connect(self.prev_item)
        next_button.clicked.connect(self.next_item)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.left_label)
        image_layout.addWidget(self.right_label)

        controls = QHBoxLayout()
        controls.addWidget(prev_button)
        controls.addWidget(self.info_label)
        controls.addWidget(next_button)

        root = QVBoxLayout()
        root.addLayout(image_layout)
        root.addLayout(controls)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def update_display(self):
        result_path = self.result_files[self.index]
        pair_id = result_path.stem.split("_")[-1]
        pair = self.pair_map.get(pair_id)
        if not pair:
            self.info_label.setText(f"Pair not found for {pair_id}")
            return

        left = load_image(pair.old_path)
        right = load_image(result_path)

        self.left_label.setPixmap(self.to_pixmap(left))
        self.right_label.setPixmap(self.to_pixmap(right))
        self.info_label.setText(
            f"{self.index + 1}/{len(self.result_files)} | pair={pair_id}"
        )

    def to_pixmap(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)
        return pixmap.scaled(
            560,
            560,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def next_item(self):
        if not self.result_files:
            return
        self.index = (self.index + 1) % len(self.result_files)
        self.update_display()

    def prev_item(self):
        if not self.result_files:
            return
        self.index = (self.index - 1) % len(self.result_files)
        self.update_display()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            self.next_item()
        elif event.key() == Qt.Key.Key_Left:
            self.prev_item()


def main():
    app = QApplication(sys.argv)
    window = PairViewer()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

