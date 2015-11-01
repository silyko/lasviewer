from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import *
import numpy as np
import traceback
import threading
import sys
import os
import json
import laspy.file as lasf
import qt_glviewer

ABOUT = "A pointcloud viewer based on laspy"

CLS_MAP = {1: (0.9, 0.9, .9), 6: (0.8, 0, 0), 9: (0, 0, 0.9), 2: (0.6, 0.5, 0), 3: (
    0, 0.8, 0), 4: (0, 0.6, 0), 5: (0.1, 0.9, 0), 17: (0, 0.35, 0.3), 18: (0.1, 0.5, 0.5)}

COLOR_LIST = ((0.9, 0, 0), (0, 0.9, 0), (0, 0, 0.9),
              (0.7, 0.8, 0), (0, 0.8, 0.7))


# Some colormap methods
def class_to_color(cls, cls_map):
    colors = np.ones((cls.shape[0], 3), dtype=np.float32) * 0.5
    for c in cls_map:
        colors[cls == c] = cls_map[c]
    return colors


def discrete_dimension_to_color(all_vals, color_list):
    colors = np.zeros((all_vals.shape[0], 3), dtype=np.float32)
    vals = np.unique(all_vals)
    for i, val in enumerate(vals):
        M = (all_vals == val)
        colors[M] = color_list[i % len(color_list)]
    return colors


def linear_colormap(all_vals, color_low, color_high):
    c1 = np.ones((all_vals.shape[0], 3), dtype=np.float32) * color_low
    c2 = np.ones((all_vals.shape[0], 3), dtype=np.float32) * color_high
    m1 = np.percentile(all_vals, 5)
    m2 = np.percentile(all_vals, 95)
    dv = ((all_vals - m1) / (m2 - m1)).reshape((all_vals.shape[0], 1))
    dv[dv < 0] = 0
    dv[dv > 1] = 1
    colors = c1 + c2 * dv
    return colors


class RedirectOutput(object):

    def __init__(self, win, signal):
        self.win = win
        self.signal = signal
        self.buffer = ""

    def write(self, text):
        self.buffer += text
        if self.buffer[-1] == "\n":
            self.flush()

    def flush(self):
        if len(self.buffer) == 0:
            return
        if self.buffer[-1] == "\n":
            self.win.emit(self.signal, self.buffer[:-1])
        else:
            self.win.emit(self.signal, self.buffer)
        self.buffer = ""


class TextViewer(QDialog):
    """Class to display text output"""

    def __init__(self, parent):
        QDialog.__init__(self, parent)
        self.setWindowTitle("Log")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.txt_field = QTextEdit(self)
        self.txt_field.setCurrentFont(QFont("Courier", 9))
        self.txt_field.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.txt_field.setReadOnly(True)
        self.txt_field.setMinimumSize(600, 200)
        layout = QVBoxLayout(self)
        layout.addWidget(self.txt_field)

    def log(self, text, color):
        self.txt_field.setTextColor(QColor(color))
        self.txt_field.append(text)
        self.txt_field.ensureCursorVisible()


menu_model = (
    {"name": "&File",
     "items": (("Open", "onOpenFile"),
               ("About", "onAbout"),
               ("Exit", "onClose"))
     },
    {"name": "&Color",
     "items": (("By class", "colorByClass"),
               ("By z", "colorByZ"),
               ("By return", "colorByRetNum"),
               ("By source id", "colorByPid"),
               ("By rgb", "colorByRGB"),
               ("By intensity", "colorByIntensity"))
     },
    {"name": "&Display",
     "items": (("Increase point size", "increasePointSize"),
               ("Decrease point size", "decreasePointSize"),
               ("Filtering", "setFilter"),
               ("Clear mask", "clearMask"),
               ("Reset", "resetView"))}
)


class LasViewer(QtGui.QMainWindow):
    """Container for PointCloudViewerWidget"""

    def __init__(self, fname=None):
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle("LasViewer")
        self.viewer = qt_glviewer.PointcloudViewerWidget(self)
        self.setCentralWidget(self.viewer)
        menubar = self.menuBar()
        for menu_def in menu_model:
            menu = menubar.addMenu(menu_def["name"])
            for item, handler in menu_def["items"]:
                action = QAction(item, self)
                menu.addAction(action)
                if handler:
                    action.triggered.connect(getattr(self, handler))

        self.dir = "/"
        self.logWindow = TextViewer(self)
        self.filtering_expression = '{"x": [xmin,xmax], "y": [ymin,ymax],...}'
        # threading stuff
        self.background_task_signal = QtCore.SIGNAL("__my_backround_task")
        self.log_stdout_signal = QtCore.SIGNAL("__stdout_signal")
        self.log_stderr_signal = QtCore.SIGNAL("__stderr_signal")
        QtCore.QObject.connect(
            self, self.background_task_signal, self.finishBackgroundTask)
        QtCore.QObject.connect(self, self.log_stdout_signal, self.logStdout)
        QtCore.QObject.connect(self, self.log_stderr_signal, self.logStderr)
        self.filename = None
        self.display_dimension = "raw_classification"
        self.lasf_object = None
        self.err_msg = None
        # redirect textual output
        if "debug" not in sys.argv:
            sys.stdout = RedirectOutput(self, self.log_stdout_signal)
            sys.stderr = RedirectOutput(self, self.log_stderr_signal)
        self.show()
        if fname is not None:
            self.openFile(fname)
        else:
            QMessageBox.information(
                self, "Movement", "Use first and second mouse button as well as 'asdw' to move around.")

    # Menu event handlers here
    def onClose(self):
        self.close()

    def onAbout(self):
        msg = ABOUT
        QMessageBox.about(self, "About", msg)

    def onOpenFile(self):
        my_file = unicode(QFileDialog.getOpenFileName(
            self, "Select a vector-data input file", self.dir))
        if len(my_file) > 0:
            self.openFile(my_file)

    def colorByClass(self):
        self.display_dimension = "raw_classification"
        self.onChangeColorMode()

    def colorByZ(self):
        self.display_dimension = "z"
        self.onChangeColorMode()

    def colorByRGB(self):
        self.display_dimension = "rgb"
        self.onChangeColorMode()

    def colorByPid(self):
        self.display_dimension = "pt_src_id"
        self.onChangeColorMode()

    def colorByRetNum(self):
        self.display_dimension = "return_num"
        self.onChangeColorMode()

    def colorByIntensity(self):
        self.display_dimension = "intensity"
        self.onChangeColorMode()

    def increasePointSize(self):
        self.viewer.increase_point_size()

    def decreasePointSize(self):
        self.viewer.decrease_point_size()

    def resetView(self):
        self.viewer.reset_all()
    
    def setFilter(self):
        if self.lasf_object is not None:
            expression, ok = QInputDialog.getText(self,
                                    "Filtering",
                                    "JSON expression:",
                                    text=self.filtering_expression)
            if ok:
                self.filtering_expression = str(expression)
                conditions = json.loads(self.filtering_expression)
                M = np.ones((len(self.lasf_object),), dtype = np.bool)
                for key in conditions:
                    val_min, val_max = conditions[key]
                    vals = getattr(self.lasf_object, key)
                    M &= vals <= val_max
                    M &= vals >= val_min
                self.viewer.set_mask(M)
                self.log("Updating view..")
                self.viewer.update_view()
    
    def clearMask(self):
        self.viewer.clear_mask()
        self.log("Updating view..")
        self.viewer.update_view()
                
    # Other methods
    def onChangeColorMode(self):
        if self.lasf_object is not None:
            self.runInBackground(self._setColorsInBackground)

    def openFile(self, my_file):
        self.dir = os.path.dirname(my_file)
        self.log("Opening " + my_file + "...")
        self.filename = my_file
        if self.lasf_object is not None:  # hmm check destructor
            self.lasf_object.close()
        self.runInBackground(self._loadInBackground)

    def runInBackground(self, run_method):
        """
        Start a background task.
        finishBackgroundTask should be triggered by an event when done.
        """
        self.setEnabled(False)
        self.err_msg = None  # Nothing bad - yet!
        thread = threading.Thread(target=run_method)
        # probably exceptions in the run method should be handled
        # there in order to avoid a freeze...
        # Avoud calling GUI methods in thread!
        thread.start()

    def finishBackgroundTask(self):
        # This is called from an emmitted event -
        # the last execution from the run method...
        self.setEnabled(True)
        if self.err_msg is not None:
            # Something went wrong!
            raise Exception(self.err_msg)
        else:
            # Everything should have been transfered to the viewer
            # and we're good to go
            self.statusBar().showMessage("Source: %s, display dimension: %s" %
                                         (self.filename, self.display_dimension))
            self.viewer.update_view()

    def _setColorsInBackground(self):
        self.setColors()
        self.emit(self.background_task_signal)

    def setColors(self):
        """
        This can happen in a background thread,
        so beware not to call any GUI methods.
        """
        dim = self.display_dimension

        try:
            if dim == "rgb":
                self.log("Getting rgb")
                colors = np.array([getattr(self.lasf_object, color) for color
                                   in ("red", "green", "blue")]).T.astype(np.float32)
                _max = np.max(colors)
                _min = np.min(colors)
                diff = _max - _min
                colors -= _min
                colors /= diff
            else:
                self.log("Getting dimension " + dim)
                data = getattr(self.lasf_object, dim)
                self.log("Generating colors...")
                if dim == "intensity":  # normalise
                    data = data.astype(np.float32)
                    data /= data.max()
                if dim == "raw_classification":
                    colors = class_to_color(data, CLS_MAP)
                elif data.dtype == np.float32 or data.dtype == np.float64:
                    colors = linear_colormap(
                        data, (0.1, 0.1, 0.1), (0.9, 0.9, 0.9))
                else:
                    colors = discrete_dimension_to_color(data, COLOR_LIST)
            self.viewer.set_colors(colors)
        except Exception as e:
            self.err_msg = traceback.format_exc()
            self.err_msg += "\n" + str(e)

    def _loadInBackground(self):
        self.load()
        self.emit(self.background_task_signal)

    def load(self):
        """
        This can happen in a background thread,
        so beware not to call any GUI methods.
        """
        try:
            self.lasf_object = lasf.File(self.filename)
        except Exception as e:
            self.lasf_object = None
            self.err_msg = str(e)
            return
        else:
            # Transfer x, y, z (will reset position)
            self.viewer.set_points(self.lasf_object.x,
                                   self.lasf_object.y,
                                   self.lasf_object.z)
            self.setColors()

    def log(self, text):
        self.emit(self.log_stdout_signal, text)

    def logDebug(self, text, color="blue"):
        # For logging to logWindow (debug etc...)
        # TODO: use event propagation to capture
        #       logging from background threads...
        self.logWindow.log(text, color)

    @pyqtSlot(str)
    def logStderr(self, text):
        # Show traceback etc...
        self.logWindow.show()
        self.logWindow.log(text, "red")

    @pyqtSlot(str)
    def logStdout(self, text):
        # Just for small temporary messages
        self.statusBar().showMessage(text)

if __name__ == '__main__':
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        fname = sys.argv[1]
    else:
        fname = None
    app = QtGui.QApplication(sys.argv)
    window = LasViewer(fname)
    sys.exit(app.exec_())
