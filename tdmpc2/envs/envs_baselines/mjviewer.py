from threading import Lock
import glfw
import mujoco
import mujoco.viewer
import time
import copy
from multiprocessing import Process, Queue
import numpy as np
import imageio


class MjViewerBasic:
    """
    A simple display GUI showing the scene of a MuJoCo model with a mouse-movable camera.
    MjViewer extends this class to provide more sophisticated playback and interaction controls.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.viewer = None
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self.window = None
        self._scale = 1.0

    def render(self):
        """
        Render the current simulation state to the screen.
        Call this in your main loop.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        with self._gui_lock:
            self.viewer.sync()

    def close(self):
        """Close the viewer"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def key_callback(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE and key == glfw.KEY_ESCAPE:
            print("Pressed ESC")
            print("Quitting.")
            self.close()
            exit(0)


class MjViewer(MjViewerBasic):
    """
    Extends MjViewerBasic to add video recording and interaction controls.
    The key bindings are simplified for the new MuJoCo API.
    """

    def __init__(self, model, data):
        super().__init__(model, data)

        self._ncam = model.ncam
        self._paused = False
        self._advance_by_one_step = False

        # Vars for recording video
        self._record_video = False
        self._video_queue = Queue()
        self._video_idx = 0
        self._video_path = "/tmp/video_%07d.mp4"

        # vars for capturing screen
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"

        self._run_speed = 1.0
        self._loop_count = 0
        self._render_every_frame = False

        self._show_mocap = True
        self._transparent = False

        self._time_per_render = 1 / 60.0
        self._hide_overlay = False
        self._user_overlay = {}

        self.video_fps = 30
        self.frame_skip = 1
        self.sim_time = 0
        self.custom_key_callback = None

    def render(self):
        """
        Render the current simulation state to the screen.
        Call this in your main loop.
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        render_start = time.time()
        
        with self._gui_lock:
            self.viewer.sync()
            
        if self._record_video:
            # For video recording, we would need to implement screen capture
            # This is simplified for the new API
            pass
        else:
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

    def key_callback(self, window, key, scancode, action, mods):
        if self.custom_key_callback is not None:
            res = self.custom_key_callback(key, action, mods)
            if res:
                return

        if action != glfw.RELEASE:
            return
        elif key == glfw.KEY_TAB:  # Switches cameras - simplified
            pass  # Camera switching would be handled differently in new API
        elif key == glfw.KEY_H:  # hides all overlay
            self._hide_overlay = not self._hide_overlay
        elif key == glfw.KEY_SPACE and self._paused is not None:  # stops simulation
            self._paused = not self._paused
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        elif key == glfw.KEY_V:  # Records video
            self._record_video = not self._record_video
            if self._record_video:
                fps = self.video_fps
                self._video_process = Process(target=save_video,
                                  args=(self._video_queue, self._video_path % self._video_idx, fps))
                self._video_process.start()
            if not self._record_video:
                self._video_queue.put(None)
                self._video_process.join()
                self._video_idx += 1
        elif key == glfw.KEY_T:  # capture screenshot - simplified
            # Screenshot functionality would need to be reimplemented
            pass
        elif key == glfw.KEY_I:  # drops in debugger
            print('You can access the model and data by self.model and self.data')
            try:
                import ipdb
                ipdb.set_trace()
            except ImportError:
                import pdb
                pdb.set_trace()
        elif key == glfw.KEY_S:  # Slows down simulation
            self._run_speed /= 2.0
        elif key == glfw.KEY_F:  # Speeds up simulation
            self._run_speed *= 2.0
        elif key == glfw.KEY_C:  # Contact forces - simplified
            pass  # Contact force visualization would be handled differently
        elif key == glfw.KEY_D:  # render every frame
            self._render_every_frame = not self._render_every_frame
        elif key == glfw.KEY_R:  # transparency - simplified
            self._transparent = not self._transparent
        elif key == glfw.KEY_M:  # mocap bodies - simplified
            self._show_mocap = not self._show_mocap

        super().key_callback(window, key, scancode, action, mods)

# Separate Process to save video. This way visualization is
# less slowed down.


def save_video(queue, filename, fps):
    writer = imageio.get_writer(filename, fps=fps)
    while True:
        frame = queue.get()
        if frame is None:
            break
        writer.append_data(frame)
    writer.close()
