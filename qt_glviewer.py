from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import *
import OpenGL.GL as gl
import OpenGL.GLU as glu
from OpenGL.arrays import vbo
import math
import numpy as np

"""
OpenGL pointcloud rendering.
Minor modifications of original laspy code.
Generalized - is not LAS specific.
"""


class VBOProvider(object):

    def __init__(self, x, y, z, colors):
        n = x.shape[0]
        i0 = 0
        vbsize = 2000000
        self.vbos = []
        while i0 < n:
            i1 = min(n, i0 + vbsize)
            data = np.column_stack((x[i0:i1], y[i0:i1], z[i0:i1], colors[i0:i1])).astype(np.float32)
            vbo_ = vbo.VBO(data=data, usage=gl.GL_DYNAMIC_DRAW,
                           target=gl.GL_ARRAY_BUFFER)
            self.vbos.append((vbo_, i1 - i0))
            i0 += vbsize

    def draw(self):
        for vbo_, n in self.vbos:
            vbo_.bind()
            gl.glVertexPointer(3, gl.GL_FLOAT, 24, vbo_)
            gl.glColorPointer(3, gl.GL_FLOAT, 24, vbo_ + 12)
            gl.glDrawArrays(gl.GL_POINTS, 0, n)
            vbo_.unbind()


class PointcloudViewerWidget(QGLWidget):
    """
    A simple QGLWidget which can display a pointcloud with some
    colors. This object is completely oblivious to where the data comes
    from - will just need x, y, z and some colors.
    """

    def __init__(self, parent):
        QGLWidget.__init__(self, parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.parent = parent
        self.initial_z = 1500.0
        # The far and near z clipping planes -
        # we should be able to change these at some point, e.g. for
        # 'galaxies'
        self.far_z = 3000.0
        self.near_z = 0.01
        # Initial locations and camera position
        self.location = np.array([0.0, 0.0, self.initial_z])
        self.focus = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.center = np.array([0.0, 0.0, 0.0])
        self.dist = self.initial_z
        self.real_pos = self.location + self.center
        self.data_buffer = None
        # Movement stuff..
        self.movement_granularity = 6.0
        self.look_granularity = 16.0
        self.old_mouse_x = 0
        self.old_mouse_y = 0
        self.mouse_speed = 0.4
        self.point_size = 1
        # We will store x, y, z and colors here to increase speed, e.g.
        # when changing colors. Comes at the cost of increasing memory
        # consumption. So only store as float32 in viewer system.
        self.x = None
        self.y = None
        self.z = None
        self.colors = None
        self.mask = None  # We can mask points ...
        # Appearence
        self.setMinimumSize(600, 600)

    def set_mask(self, mask, refine=False):
        if self.mask is None or not refine:
            self.mask = mask
        else:
            self.mask &= mask

    def clear_mask(self):
        self.mask = None

    def increase_point_size(self):
        if self.point_size < 5:
            self.point_size += 1
            gl.glPointSize(self.point_size)
            self.update()
            self.setFocus()

    def decrease_point_size(self):
        if self.point_size > 1:
            self.point_size -= 1
            gl.glPointSize(self.point_size)
            self.update()
            self.setFocus()

    def set_points(self, x, y, z):
        """
        Input will (probably) be float64 arrays.
        Subtract mean and store as float32.
        """
        # The center in real coordinates
        self.center[0] = x.mean()
        self.center[1] = y.mean()
        self.center[2] = z.mean()
        # store as float32
        self.x = (x-self.center[0]).astype(np.float32)
        self.y = (y-self.center[1]).astype(np.float32)
        self.z = (z-self.center[2]).astype(np.float32)
        # Check how much we need to move 'up'
        r = max(self.x.max(), self.y.max())
        self.initial_z = r * 1.5
        # The location in viewer coordinates
        self.location = np.array([0.0, 0.0, self.initial_z])
        self.movement_granularity = max(r / 500.0 * 6, 1)
        # Position in real coordinates
        self.real_pos = self.location + self.center

    def set_colors(self, colors):
        self.colors = colors.astype(np.float32)

    def update_view(self):
        """
        Regenerate VBOs. If self.mask is not None, perform a prefiltering.
        Note, this may consume more memory.
        """
        x = self.x[self.mask] if self.mask is not None else self.x
        y = self.y[self.mask] if self.mask is not None else self.y
        z = self.z[self.mask] if self.mask is not None else self.z
        colors = self.colors[self.mask] if self.mask is not None else self.colors
        self.data_buffer = VBOProvider(x, y, z, colors)
        self.update()
        self.setFocus()

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        glu.gluLookAt(self.location[0], self.location[1], self.location[2],
                      self.focus[0], self.focus[1], self.focus[2],
                      self.up[0], self.up[1], self.up[2])
        if self.data_buffer is not None:
            self.draw_points()
            diff = self.focus - self.location
            d = np.sqrt(diff.dot(diff))
            self.renderText(10, 10, "Position: %.2f,%.2f,%.2f, dist: %.2f" % (
                self.real_pos[0], self.real_pos[1], self.real_pos[2], d))

    def resizeGL(self, w, h):
        ratio = w if h == 0 else float(w) / h
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glViewport(0, 0, w, h)
        gl.glLoadIdentity()
        glu.gluPerspective(90, float(ratio), self.near_z, self.far_z)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def initializeGL(self):
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClearDepth(1.0)

    def draw_points(self):
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)
        self.data_buffer.draw()
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def mouseMoveEvent(self, mouseEvent):
        if int(mouseEvent.buttons()) != QtCore.Qt.NoButton:
            # user is dragging
            delta_x = mouseEvent.x() - self.old_mouse_x
            delta_y = self.old_mouse_y - mouseEvent.y()
            if int(mouseEvent.buttons()) & QtCore.Qt.LeftButton:
                self.camera_yaw_pitch(delta_x * 0.03, delta_y * 0.03)
                self.update()
            elif int(mouseEvent.buttons()) & QtCore.Qt.RightButton:
                self.camera_roll((delta_x) * 0.05)
                self.update()
        self.old_mouse_x = mouseEvent.x()
        self.old_mouse_y = mouseEvent.y()

    def rotate_vector(self, vec_rot, vec_about, theta):
        d = np.sqrt(vec_about.dot(vec_about))

        L = np.array((0, vec_about[2], -vec_about[1],
                      -vec_about[2], 0, vec_about[0],
                      vec_about[1], -vec_about[0], 0))
        L.shape = (3, 3)

        try:
            R = (np.identity(3) + np.sin(theta) / d * L +
                 (1 - np.cos(theta)) / (d * d) * (L.dot(L)))
        except Exception:
            print("Error in rotation.")
            return()
        return(vec_rot.dot(R))

    def camera_reset(self):
        self.location = np.array([0.0, 0.0, self.initial_z])
        self.focus = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])
        self.real_pos = self.location + self.center
        self.update()

    def reset_all(self):
        self.point_size = 1
        gl.glPointSize(self.point_size)
        self.camera_reset()
        self.setFocus()  # will loose keyboard tracking else :-/

    def camera_move(self, ammount, axis=1):

        if axis == 1:
            pointing = self.focus - self.location
            pnorm = np.sqrt(pointing.dot(pointing))
            pointing /= pnorm
            self.location = self.location + ammount * pointing
            self.focus = self.location + pnorm * pointing
        elif axis == 2:
            pointing = self.focus - self.location
            direction = np.cross(self.up, pointing)
            direction /= np.sqrt(direction.dot(direction))
            self.location = self.location + ammount * direction
            self.focus = self.location + pointing
        elif axis == 3:
            pointing = self.focus - self.location
            direction = self.up
            self.location = self.location + ammount * direction
            self.focus = self.location + pointing
        self.real_pos = self.location + self.center

    def camera_yaw(self, theta):
        pointing = self.focus - self.location
        newpointing = self.rotate_vector(pointing, self.up, theta)
        self.focus = newpointing + self.location

    def camera_yaw_pitch(self, theta1, theta2):
        pointing = self.focus - self.location
        d1 = np.sqrt(pointing.dot(pointing))
        newpointing1 = self.rotate_vector(pointing, self.up, theta1)
        axis = np.cross(self.up, pointing)
        newpointing2 = self.rotate_vector(pointing, axis, theta2)
        self.up = np.cross(newpointing2, axis)
        self.up /= np.sqrt(self.up.dot(self.up))
        total = newpointing1 + newpointing2
        d2 = np.sqrt(total.dot(total))
        self.focus = total * d1 / d2 + self.location

    def camera_roll(self, theta):
        self.up = self.rotate_vector(
            self.up, self.focus - self.location, theta)

    def camera_pitch(self, theta):
        pointing = self.focus - self.location
        axis = np.cross(self.up, pointing)
        newpointing = self.rotate_vector(pointing, axis, theta)
        self.focus = newpointing + self.location
        self.up = np.cross(newpointing, axis)
        self.up /= np.sqrt(self.up.dot(self.up))

    def wheelEvent(self, event):
        if self.data_buffer is not None:
            self.camera_move(event.delta() * self.movement_granularity * 0.03)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if self.data_buffer is not None:
            self.camera_move(self.movement_granularity)
            self.update()

    # for this to work - we seemingly need to give focus to this widget from
    # time to time...
    def keyPressEvent(self, event):
        if self.data_buffer is not None:
            if event.key() == QtCore.Qt.Key_A:
                self.camera_move(self.movement_granularity, 2)
                self.update()
            elif event.key() == QtCore.Qt.Key_D:
                self.camera_move(-self.movement_granularity, 2)
                self.update()
            elif event.key() == QtCore.Qt.Key_W:
                self.camera_move(self.movement_granularity, 3)
                self.update()
            elif event.key() == QtCore.Qt.Key_S:
                self.camera_move(-self.movement_granularity, 3)
                self.update()
            event.accept()
        else:
            event.ignore()


class ViewerContainer(QtGui.QWidget):
    """
    A handy container widget for the PointcloudViewerWidget.
    Not LAS specific.
    We can increase point size, decrease point size and
    reset view
    """

    def __init__(self, parent):
        QtGui.QWidget.__init__(self, parent)
        # self.setupUi(self)
        self.viewer = PointcloudViewerWidget(self)
        v_layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        v_layout.addLayout(top_layout)
        self.bt_ps_plus = QPushButton("+", self)
        self.bt_ps_minus = QPushButton("-", self)
        self.bt_reset_view = QPushButton("reset", self)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.controls = [self.bt_ps_plus, self.bt_ps_minus, self.bt_reset_view]
        for bt, handler in zip(self.controls,
                               ("on_ps_plus", "on_ps_minus", "on_reset")):
            bt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            top_layout.addWidget(bt)
            bt.clicked.connect(getattr(self, handler))
        v_layout.addWidget(self.viewer)
        self.set_loaded_state(False)

    def set_loaded_state(self, is_loaded):
        for widget in self.controls:
            widget.setEnabled(is_loaded)

    def on_reset(self):
        self.viewer.reset_all()

    def on_ps_plus(self):
        self.viewer.increase_point_size()

    def on_ps_minus(self):
        self.viewer.decrease_point_size()

    def set_points(self, x, y, z):
        self.viewer.set_points(x, y, z)

    def set_colors(self, colors):
        self.viewer.set_colors(colors)

    def update_view(self):
        self.viewer.update_view()
