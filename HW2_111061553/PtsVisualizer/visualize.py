import os
import sys
import numpy as np
import shader

from PyQt5 import QtWidgets, QtGui, QtOpenGL
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import PyQt5.QtCore as QtCore

import glm
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from Viewer import PointCloudView

class TopWindow(QMainWindow):
    def __init__(self, color, pts, parent=None):
        super().__init__(parent)
        sizeObject = QtWidgets.QDesktopWidget().screenGeometry(-1)
        [self.h, self.w] = [sizeObject.height(), sizeObject.width()]
        ratio = 0.9
        self.h = int(self.h * ratio)
        self.w = int(self.w * ratio)
        self.setGeometry(20, 60, self.w, self.h)
        self.setWindowTitle("Point Cloud Visualizer")
        self.centeralWidget = QWidget(self)

        self.color = color
        self.pts = pts
        self.viewer = PointCloudView.GLWindow(color, pts, self, self, sz=1.2)
        '''
        self.viewer.cam_pos = glm.vec3(0, 0, -3.98)
        self.viewer.mat_view = np.asarray(
                [[ 1.0,         0.0,        -0.0,         0.0       ],
                 [ 0.0,        -1.0,        -0.0,         0.0       ],
                 [-0.0,         0.0,        -1.0,         0.0       ],
                 [-0.0,        -0.0,        -3.98,  1.0       ]
                 ])
        self.viewer.Transform = np.asarray([
                   [ 0.9673812,   0.1344713,  -0.21468577,  0.        ],
                   [ 0.1722323,   0.27233148,  0.9466628,   0.        ],
                   [ 0.18576485, -0.95276004,  0.24028815,  0.        ],
                   [ 0.,          0.,          0.,          0.9996966 ]
                   ])
        '''
        layout = QGridLayout()
        layout.setRowStretch(0, 1)
        layout.setColumnStretch(0, 1)
        layout.addWidget(self.viewer, 0, 0, 1, 1)
        self.centeralWidget.setLayout(layout)
        self.setCentralWidget(self.centeralWidget)

    def enterEvent(self, event):
        self.setFocus(True)


if __name__ == '__main__':
    #path = sys.argv[1]
    pts = np.load(sys.argv[1])
    pts[:, 1] *= -1
    color = np.ones_like(pts)

    app = QtWidgets.QApplication(sys.argv)
    window = TopWindow(color=color, pts=pts)
    window.show()
    #app.exec_()
    sys.exit(app.exec_())
