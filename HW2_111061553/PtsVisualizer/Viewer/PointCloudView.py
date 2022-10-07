import os
import sys
import numpy as np
import shader
import ArcBall

from PyQt5 import QtWidgets, QtGui, QtOpenGL
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
import PyQt5.QtCore as QtCore

import glm
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def GenPtsVAO(color, pts):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, pts.astype(np.float32), GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, color.astype(np.float32), GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(1)

    return vao, None

def GenPtsShader():
    program = glCreateProgram()
    vertexShader = glCreateShader(GL_VERTEX_SHADER)
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER)
    
    glShaderSource(vertexShader, shader.vertex_pts.src)
    glShaderSource(fragmentShader, shader.fragment_pts.src)

    glCompileShader(vertexShader)
    glCompileShader(fragmentShader)

    print (glGetShaderInfoLog(vertexShader))
    print (glGetShaderInfoLog(fragmentShader))
    #exit()
    glAttachShader(program, vertexShader)
    glAttachShader(program, fragmentShader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not result:
        print(glGetProgramInfoLog(program))
    #print (result)

    um4p = glGetUniformLocation(program, "um4p")
    um4v = glGetUniformLocation(program, "um4v")
    um4m = glGetUniformLocation(program, "um4m")
    return program, um4p, um4v, um4m


class GLWindow(QOpenGLWidget):
    def __init__(self, color, pts, main, parent=None, sz=5):
        super(GLWindow, self).__init__(parent)
        self.main = main
        self.lastPos = None
        self.width = 720
        self.height = 720
        self.ball = ArcBall.ArcBallT(self.width, self.height)
        self.LastRot = ArcBall.Matrix3fT()
        self.ThisRot = ArcBall.Matrix3fT()
        self.Transform = ArcBall.Matrix4fT()
        #self.cam_pos = glm.vec3(0, -6, 0)
        #self.cam_tgt = glm.vec3(0, 1, 0)
        self.cam_pos = glm.vec3(0, 0, 0)
        self.cam_tgt = glm.vec3(0, 0, 100)

        self.first_time = True

        glFormat = QtGui.QSurfaceFormat()
        glFormat.setVersion(4, 1)
        glFormat.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        self.setFormat(glFormat)
        QtGui.QSurfaceFormat.setDefaultFormat(glFormat)

        self.color = color
        self.pts = pts
        self.sz = sz
        

    def initializeGL(self):
        print (glGetString(GL_VERSION))
        print (glGetString(GL_SHADING_LANGUAGE_VERSION))
        glEnable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glDepthFunc(GL_LEQUAL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self.program_id, self.um4p, self.um4v, self.um4m = GenPtsShader()
        self.vao, _ = GenPtsVAO(self.color, self.pts)

    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        self.ball = ArcBall.ArcBallT(self.width, self.height)
        glViewport(0, 0, width, height)
        viewportAspect = float(width) / float(height)
        projection = glm.perspective(50.0/180.0*np.pi, viewportAspect, 0.001, 100.0);
        view = glm.lookAt(self.cam_pos,    self.cam_tgt,   glm.vec3(0, -1, 0))

        self.mat_proj = np.asarray(projection).T
        self.mat_view = np.asarray(view).T

    def paintGL(self):
        glClearColor(0.0, 0.0, 0.0, 1)
        #glClearColor(0.0, 0.0, 0.0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glUseProgram(self.program_id)
        glBindVertexArray(self.vao)
        
        glUniformMatrix4fv(self.um4p, 1, GL_FALSE, self.mat_proj.astype(np.float32))
        glUniformMatrix4fv(self.um4v, 1, GL_FALSE, self.mat_view.astype(np.float32))
        glUniformMatrix4fv(self.um4m, 1, GL_FALSE, self.Transform.astype(np.float32))
        
        #print ('#######################')
        #print (self.cam_pos)
        #print (self.cam_tgt)
        #print (self.mat_view)
        #print (self.Transform)

        glPointSize(float(self.sz))
        glDrawArrays(GL_POINTS, 0, self.pts.shape[0])
        glUseProgram(0)
        glFlush()

    def mousePressEvent(self, event):
        pos = event.pos()
        pt = ArcBall.Point2fT(pos.x(), pos.y())
        self.ball.click(pt)
        
    
    def mouseMoveEvent(self, event):
        pos = event.pos()
        pt = ArcBall.Point2fT(pos.x(), pos.y())
        try:
            ThisQuat = self.ball.drag (pt)
        except:
            return
        ThisRot = ArcBall.Matrix3fSetRotationFromQuat4f (ThisQuat)
        if self.LastRot is None:
            self.LastRot = ArcBall.Matrix3fT()
        if self.Transform is None:
            self.Transform = ArcBall.Matrix4fT()
        self.ThisRot = ArcBall.Matrix3fMulMatrix3f (self.LastRot, ThisRot)
        self.Transform = ArcBall.Matrix4fSetRotationFromMatrix3f(self.Transform, self.ThisRot)

        self.update()

    def mouseReleaseEvent(self,event):
        self.LastRot = self.ThisRot 

    def keyPressEvent(self, event):
        step = 0.5
        key = event.key()
        modifiers = event.modifiers()
        if key == ord('S'):
            self.cam_pos[2] -= step
            self.cam_tgt[2] -= step
        elif key == ord('W'):
            self.cam_pos[2] += step
            self.cam_tgt[2] += step
        elif key == ord('A'):
            self.cam_pos[0] -= step
            self.cam_tgt[0] -= step
        elif key == ord('D'):
            self.cam_pos[0] += step
            self.cam_tgt[0] += step

        self.mat_view = np.asarray(glm.lookAt(self.cam_pos, self.cam_tgt, glm.vec3(0, -1, 0))).T
        self.update()

    def enterEvent(self, event):
        self.setFocus(True)

    def wheelEvent(self,event):
        numAngle = float(event.angleDelta().y()) / 120
        self.cam_pos[2] += numAngle
        if self.cam_pos[2] > 5: self.cam_pos[2] = 5
        self.mat_view = np.asarray(glm.lookAt(self.cam_pos, self.cam_tgt, glm.vec3(0, -1, 0))).T
        self.update()

    
    
        

if __name__ == '__main__':
    img = imread('color.png', pilmode='RGB')
    with open('label.json', 'r') as f:
        label = json.load(f)
    app = QtWidgets.QApplication(sys.argv)
    window = GLWindow(img, label)
    window.show()
    sys.exit(app.exec_())
