import kivy
from kivy.config import Config
import colour as clr
from kivy.metrics import dp

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', 1000)
Config.set('graphics', 'height', 600)
from kivy.app import App
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.graphics.vertex_instructions import Rectangle, Ellipse
from kivy.graphics import Color, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.properties import *
import kivy.properties as props
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.behaviors import ButtonBehavior
from PIL import Image, ImageDraw, ImageFilter
import io
from kivy.metrics import dp, sp
import numpy as np
import os
import moviepy.editor as mpy
import time
import datetime

WORKER_COLORS = [
    '#FF1654',
    '#5026a7',
    '#247BA0',
    '#8d448b',
    '#70C1B3',
    '#B2DBBF',
    '#F3FFBD',

]

kivy.require('1.11.1')

KV_STRING = """
<MaterialWidget>
    color: (1, 0, 0, 1)
    canvas.before:
        Color:
            rgba: 1,1,1,1
        Rectangle:
            size: self.shadow_size1
            pos: self.shadow_pos1
            texture: self.shadow_texture1
        Rectangle:
            size: self.shadow_size2
            pos: self.shadow_pos2
            texture: self.shadow_texture2
        Color:
            rgba: 1,1,1,1
        Rectangle:
            size: self.size
            pos: self.pos

<ColoredCircle>:
    size_hint: None, None
    size: self.size
    pos: self.pos

    canvas.before:
        Color:
            rgb: root.color
        Ellipse:
            size: root.size
            pos: root.pos

<CircleBorder>:
    size_hint: None, None
    size: self.size
    pos: self.pos
    canvas:
        Color:
            rgb: root.color
        Line:
            width: root.border_width
            circle:
                (self.center_x, self.center_y, min(self.width, self.height) * 0.5, 0, root.theta)

<LabelTextClass>:
    size_hint: None, None
    size: self.size
    pos: self.pos

    canvas.before:
        Color:
            rgba: [0, 0, 0, 0]
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        pos: root.pos
        size: root.size
        text: root.text
        color: root.text_color


<RectangleBorder>:
    size_hint: None, None
    size: self.size
    pos: self.pos

    canvas:
        Color:
            rgb: root.color
        Line:
            width: root.border_width
            rectangle: (root.x, root.y, root.width, root.height)


<ColoredRectangle>:
    size_hint: None, None
    size: self.size
    pos: self.pos

    canvas.before:
        Color:
            rgb: root.color
        Rectangle:
            size: root.size
            pos: root.pos


<LabelWidget>:
    size_hint: None, None
    size: self.size
    pos: self.pos

    Label:
        pos: root.pos
        size: self.size
        text_size: self.size
        text: root.label_text
        color: root.text_color
        halign: 'left'
"""


def get_rgba_color(val, a=1.):
    temp = clr.Color(val).get_rgb()
    temp = list(temp) + [a]
    return tuple(temp)


CONFIG = {
    # 'main_bg_color': clr.Color('#ececec').get_rgb(),
    'map_bg_color': get_rgba_color('#B8B8B8'),
    'main_bg_color': get_rgba_color('white'),
    'window_padding_x': dp(25),
    'window_padding_y': dp(25),
    'cell_border_width': dp(1),
}


def hide_widget(wid: Widget):
    if wid.hidden == 1:
        return
    setattr(wid, 'size_attrs_buffer', [wid.height, wid.size_hint_y, wid.opacity, wid.disabled])
    wid.height, wid.size_hint_y, wid.opacity, wid.disabled = 0, None, 0, True
    wid.hidden = 1


def show_widget(wid):
    if wid.hidden == 0:
        return
    wid.height, wid.size_hint_y, wid.opacity, wid.disabled = getattr(wid, 'size_attrs_buffer')
    wid.hidden = 0


def in_interval(val, interval: list):
    out = False
    if interval[0] <= val <= interval[1]:
        out = True
    return out


class PanelWidget(Widget):
    size_hint = (None, None)
    handler = props.ObjectProperty(None)
    hidden = props.NumericProperty(0)
    bg_color = props.ListProperty(None)
    is_rectangle = props.BooleanProperty(True)
    with_shadow = props.BooleanProperty(False)
    with_border = props.BooleanProperty(True)
    text = props.StringProperty(None)
    font_size = props.NumericProperty(sp(10))

    # ------------------------------------------------------------------------------------------------
    # shadow props
    shadow_texture_1 = props.ObjectProperty(None)
    shadow_pos_1 = props.ListProperty([0, 0])
    shadow_size_1 = props.ListProperty([0, 0])

    shadow_texture_2 = props.ObjectProperty(None)
    shadow_pos_2 = props.ListProperty([0, 0])
    shadow_size_2 = props.ListProperty([0, 0])

    elevation = props.NumericProperty(1)
    _shadow_clock = None

    _shadows = {
        1: (1, 3, 0.12, 1, 2, 0.24),
        2: (3, 6, 0.16, 3, 6, 0.23),
        3: (10, 20, 0.19, 6, 6, 0.23),
        4: (14, 28, 0.25, 10, 10, 0.22),
        5: (19, 38, 0.30, 15, 12, 0.22)
    }

    # ------------------------------------------------------------------------------------------------

    def __init__(self, **kwargs):
        super(PanelWidget, self).__init__(**kwargs)
        self.rad_mult = 0.25  # PIL GBlur seems to be stronger than Chrome's so I lower the radius

        self.canvas.add(Color(*self.bg_color))
        if self.with_shadow:
            self._create_shadow()
            self.canvas.add(Rectangle(size=self.shadow_size_1, pos=self.shadow_pos_1, texture=self.shadow_texture_1))
            self.canvas.add(Rectangle(size=self.shadow_size_2, pos=self.shadow_pos_2, texture=self.shadow_texture_2))
            self.canvas.add(Color(*self.bg_color))
            self.canvas.add(Rectangle(size=self.size, pos=self.pos))

        if self.with_border:
            self.canvas.add( Line(width=CONFIG['cell_border_width'],rectangle=[self.x, self.y, self.width, self.height]))

        if self.text is not None:
            label_text_size = self.width, self.height
            label_text_pos = self.x, self.y
            self.label_text = Label(pos=label_text_pos,
                                    size=label_text_size,
                                    text=self.text,
                                    color=get_rgba_color('black'),
                                    font_size=self.font_size
                                    )
            self.add_widget(self.label_text)

    def _create_shadow(self, *args):
        ow, oh = self.size[0], self.size[1]

        offset_x = 0

        # Shadow 1
        shadow_data = self._shadows[self.elevation]
        offset_y = shadow_data[0]
        radius = shadow_data[1]
        w, h = ow + radius * 6.0, oh + radius * 6.0
        texture_1 = self._create_boxshadow(ow, oh, radius, shadow_data[2])
        self.shadow_texture_1 = texture_1
        self.shadow_size_1 = w, h
        self.shadow_pos_1 = self.x - (w - ow) / 2. + offset_x, self.y - (h - oh) / 2. - offset_y

        # Shadow 2
        shadow_data = self._shadows[self.elevation]
        offset_y = shadow_data[3]
        radius = shadow_data[4]
        w, h = ow + radius * 6.0, oh + radius * 6.0
        texture_2 = self._create_boxshadow(ow, oh, radius, shadow_data[5])
        self.shadow_texture_2 = texture_2
        self.shadow_size_2 = w, h
        self.shadow_pos_2 = self.x - (w - ow) / 2. + offset_x, self.y - (h - oh) / 2. - offset_y

    def _create_boxshadow(self, ow, oh, radius, alpha):
        # We need a bigger texture to correctly blur the edges
        w = ow + radius * 6.0
        h = oh + radius * 6.0
        w = int(w)
        h = int(h)
        texture = Texture.create(size=(w, h), colorfmt='rgba')
        im = Image.new('RGBA', (w, h), color=(1, 1, 1, 0))

        draw = ImageDraw.Draw(im)
        # the rectangle to be rendered needs to be centered on the texture
        x0, y0 = (w - ow) / 2., (h - oh) / 2.
        x1, y1 = x0 + ow - 1, y0 + oh - 1
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0, int(255 * alpha)))
        im = im.filter(ImageFilter.GaussianBlur(radius * self.rad_mult))
        texture.blit_buffer(im.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        return texture


class EnvHandler:
    def __init__(self, env_data_path):
        self.demand_map_buffer, self.altitude_map_buffer, self.worker_trace_buffer, self.visited_cells_buffer = np.load(
            env_data_path, allow_pickle=True)
        self.demand_map_buffer = np.concatenate(self.demand_map_buffer).reshape(-1, 2)
        self.altitude_map_buffer = np.concatenate(self.altitude_map_buffer).reshape(-1, 2)
        self.worker_trace_buffer = np.concatenate(self.worker_trace_buffer).reshape(-1, 2)
        self.env_ticks_max = self.worker_trace_buffer[:, 0].max()
        self.tick_snapshots = self.create_tick_snapshots()
        self.num_rows, self.num_cols = self.demand_map_buffer[0, 1].shape
        self.num_workers = self.worker_trace_buffer[0, 1].shape[0]
        self.altitude_map_buffer[0, 1][0, 0] = 2
        max_altitude, min_altitude = self.altitude_map_buffer[0, 1].max(), self.altitude_map_buffer[-1, 1].min()
        self.altitude_range = altitude_range = np.arange(min_altitude, max_altitude + 1).astype(int)
        colors_list = self.get_colors_list(altitude_range)
        self.cell_color_map = dict((altitude, c) for altitude, c in zip(altitude_range, colors_list))

    def create_tick_snapshots(self):
        tick_snapshots = []

        for tick in range(self.env_ticks_max + 1):
            tick_snapshots.append(
                dict(env_tick=tick,
                     demand_map=None,
                     altitude_map=None,
                     worker_coords=None)
            )
        for t, item in self.demand_map_buffer:
            tick_snapshots[t]['demand_map'] = item
        for t, item in self.altitude_map_buffer:
            tick_snapshots[t]['altitude_map'] = item
        for t, item in self.worker_trace_buffer:
            tick_snapshots[t]['worker_coords'] = item
        return tick_snapshots

    @staticmethod
    def get_colors_list(altitude_range):
        above_color_from = clr.Color('#57E86B')  # green
        above_color_to = clr.Color('#FEFE69')  # light yellow

        below_color_from = clr.Color('#4c3100')  # dark red
        below_color_to = clr.Color('#ffa500')  # light orange

        above_ground_levels = altitude_range[altitude_range > 0]
        under_ground_levels = altitude_range[altitude_range < 0]

        colors_range = list(above_color_from.range_to(above_color_to, len(above_ground_levels)))
        colors_above = [c.get_rgb() for c in colors_range]

        colors_range = list(below_color_from.range_to(below_color_to, len(under_ground_levels)))
        colors_below = [c.get_rgb() for c in colors_range]

        color_zero = [clr.Color('#ececec').get_rgb()]

        colors_list = colors_below + color_zero + colors_above
        return colors_list


class CustomButton(Button):
    hidden = NumericProperty(0)
    font_size = NumericProperty(dp(15))

    def __init__(self, **kwargs):
        super(CustomButton, self).__init__(**kwargs)


class Cell(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)
    cell_altitude = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Cell, self).__init__(**kwargs)

        cell_color = self.handler.cell_color_map[int(self.cell_altitude)]
        with self.canvas.before:
            self.fill_color = Color(rgb=cell_color)
            self.rectangle = Rectangle(size=self.size, pos=self.pos)

        with self.canvas:
            self.border_color = Color(rgba=get_rgba_color('black'))
            self.border = Line(
                width=CONFIG['cell_border_width'],
                rectangle=[self.x, self.y, self.width, self.height],

            )
        self.bind(cell_altitude=self.update_cell_color)

    def update_cell_color(self, *args):
        cell_color = self.handler.cell_color_map[int(self.cell_altitude)]
        self.fill_color.rgb = cell_color


class Worker(Widget):
    worker_id = NumericProperty(None)
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Worker, self).__init__(**kwargs)

        # worker_color = self.handler.worker_color_map[int(self.worker_id)]
        # worker_color = self.handler.worker_color_map[int(self.worker_id)]
        with self.canvas.before:
            self.fill_color = Color(rgba=get_rgba_color(WORKER_COLORS[self.worker_id]))
            self.circle = Ellipse(size=self.size, pos=self.pos)

        self.bind(center=self.update_center)

    def update_center(self, *args):
        self.circle.pos = self.pos


class LabelWidget(Widget):
    hidden = NumericProperty(0)
    label_text = StringProperty('None')
    text_color = ListProperty(list(clr.Color('white').get_rgb()) + [1])
    halign = StringProperty('left')

    def __init__(self, **kwargs):
        super(LabelWidget, self).__init__(**kwargs)


class ColoredRectangle(Widget):
    hidden = NumericProperty(0)
    color = ListProperty(None)  # background color of the rectangle

    def __init__(self, **kwargs):
        super(ColoredRectangle, self).__init__(**kwargs)


class VolumeColorBarWidget(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)

    def __init__(self, **kwargs):
        super(VolumeColorBarWidget, self).__init__(**kwargs)

        panel = PanelWidget(size=self.size,
                            pos=self.pos,
                            bg_color=get_rgba_color('black', a=1),
                            with_shadow=True,
                            elevation=3)
        self.add_widget(panel)

        num_cells = len(self.handler.altitude_range)
        cell_width = self.width
        cell_height = self.height / num_cells
        rect_y = self.y

        interval_label_pos_x = self.x + 1.3 * cell_width
        cell_size = cell_width, cell_height
        interval_label_size = cell_width, cell_height
        interval_label_pos_y = self.y

        for i, altitude in enumerate(self.handler.altitude_range):
            colored_rectangle = ColoredRectangle(
                color=self.handler.cell_color_map[altitude],
                x=self.x,
                y=rect_y,
                size=cell_size,
            )

            self.add_widget(colored_rectangle)

            itv_label = LabelWidget(label_text=str(altitude),
                                    pos=[interval_label_pos_x, interval_label_pos_y],
                                    size=interval_label_size,
                                    text_color=list(clr.Color('black').get_rgb()) + [1],
                                    )
            self.add_widget(itv_label)
            rect_y += cell_height
            interval_label_pos_y += cell_height


class AltitudeMap(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)
    color = ListProperty(None)

    def __init__(self, **kwargs):
        super(AltitudeMap, self).__init__(**kwargs)

        with self.canvas.before:
            Color(rgba=CONFIG['map_bg_color'])
            Rectangle(pos=self.pos, size=self.size)

        panel = PanelWidget(size=self.size,
                            pos=self.pos,
                            bg_color=get_rgba_color('black', a=1),
                            with_shadow=True,
                            elevation=3)

        self.add_widget(panel)
        self.cells_map = None
        self.cell_size = None
        self.workers = []
        self.num_rows, self.num_cols = None, None
        self.num_workers = None

        self.initialize_map()

    def initialize_map(self):
        altitude_map_init = self.handler.altitude_map_buffer[0][1]
        worker_coords_init = self.handler.worker_trace_buffer[0][1]
        self.num_workers = self.handler.num_workers

        self.num_rows, self.num_cols = altitude_map_init.shape
        max_dim = max(self.num_rows, self.num_cols)

        cell_height = self.height / max_dim
        self.cell_size = cell_size = cell_height, cell_height
        self.cells_map = np.zeros(shape=altitude_map_init.shape, dtype='object')

        # CREATE ALTITUDE MAP CELLS
        for r in np.arange(self.num_rows):
            for c in np.arange(self.num_cols):
                cell_pos = int(self.x + cell_height * c), int(self.y + self.height - cell_height * (r + 1))
                cell_altitude = altitude_map_init[r, c].astype(int)
                cell = Cell(handler=self.handler,
                            pos=cell_pos,
                            size=cell_size,
                            cell_altitude=cell_altitude,
                            )
                self.add_widget(cell)
                self.cells_map[r, c] = cell

        for w_id, worker_coords in enumerate(worker_coords_init):
            worker_center = self.get_worker_center(worker_coords)
            worker = Worker(handler=self.handler,
                            size=(cell_size[0] * 0.75, cell_size[0] * 0.75),
                            center=worker_center,
                            worker_id=w_id,
                            )
            self.add_widget(worker)
            self.workers.append(worker)

        print('----')

    def update_map(self, altitude_map, worker_coords):
        if altitude_map is not None:
            for r in np.arange(self.num_rows):
                for c in np.arange(self.num_cols):
                    new_altitude = altitude_map[r, c]
                    cell = self.cells_map[r, c]
                    cell.cell_altitude = new_altitude

        for w_id, worker_coord in enumerate(worker_coords):
            worker = self.workers[w_id]
            new_center = self.get_worker_center(worker_coords[w_id])
            worker.center = new_center

    def get_worker_center(self, worker_coords):
        w_x = int(self.x + worker_coords[0] * self.cell_size[0] + self.cell_size[0] / 2)
        w_y = int(self.y + worker_coords[1] * self.cell_size[1] + self.cell_size[1] / 2)
        return w_x, w_y


class MainUI(Widget):
    handler = ObjectProperty(None)  # type: EnvHandler
    hidden = NumericProperty(0)
    color = ListProperty(None)

    def __init__(self, **kwargs):
        super(MainUI, self).__init__(**kwargs)

        self.images = []
        self.env_ticks = 0
        with self.canvas.before:
            Color(rgb=self.color)
            Rectangle(pos=self.pos, size=self.size)

        # padding_x = CONFIG['window_padding_x']
        # padding_y = CONFIG['window_padding_y']
        # pos = padding_x, padding_y
        # size = self.width - 2 * padding_x, self.height - 2 * padding_y

        # ------------------------------------------------------------------------------------------------------------
        # ALTITUDE MAP
        altitude_map_pos = (100, 100)
        altitude_map_size = (300, 300)

        self.altitude_map = AltitudeMap(pos=altitude_map_pos, size=altitude_map_size, handler=self.handler)
        self.add_widget(self.altitude_map)

        # ------------------------------------------------------------------------------------------------------------
        # TICK DEBUG BUTTON
        btn_pos = 700, 100
        btn_size = 100, 100
        self.tick_btn = CustomButton(pos=btn_pos, size=btn_size, text='Tick')
        self.add_widget(self.tick_btn)
        # self.tick_btn.bind(on_press=self.write_video)

        # ------------------------------------------------------------------------------------------------------------
        colorbar_pos = (450, 100)
        colorbar_size = (50, 150)

        self.colorbar_widget = VolumeColorBarWidget(size=colorbar_size,
                                                    pos=colorbar_pos,
                                                    handler=self.handler)
        self.add_widget(self.colorbar_widget)

    def write_video(self, obj, fps=8):
        for i in range(self.handler.env_ticks_max + 1):
            self.tick_test(None)

        start_duration_sec = 2
        end_duration_sec = 2

        n_frames = start_duration_sec * fps
        start_list = [self.images[0] for _ in range(int(start_duration_sec * fps))]
        end_list = [self.images[-1] for _ in range(int(end_duration_sec * fps))]
        images = start_list + self.images + end_list

        clip = mpy.ImageSequenceClip(images, fps=fps, with_mask=True)
        now = datetime.datetime.now()
        filename = now.strftime("%m%d_%H%M") + "_" + "animation.mp4"
        clip.write_videofile(os.path.join(filename), fps=fps)
        exit()

    def tick_test(self, obj):
        if self.env_ticks == 0:
            img = self.export_as_image()
            img_numpy = np.frombuffer(img.texture.pixels, dtype=np.ubyte).reshape(self.height, self.width, 4)
            self.images.append(img_numpy)
        elif self.env_ticks >= len(self.handler.tick_snapshots):
            print("FINISHED")
            hide_widget(self.tick_btn)
        else:
            tick_snapshot = self.handler.tick_snapshots[self.env_ticks]
            altitude_map = tick_snapshot['altitude_map']
            worker_coords = tick_snapshot['worker_coords']
            self.altitude_map.update_map(altitude_map=altitude_map, worker_coords=worker_coords)

            img = self.export_as_image()
            img_numpy = np.frombuffer(img.texture.pixels, dtype=np.ubyte).reshape(self.height, self.width, 4)
            self.images.append(img_numpy)

        self.env_ticks += 1

    def update_and_export(self, obj):

        filename = 'snapshot_' + str(self.env_ticks).zfill(2) + '.png'
        filename = os.path.join("./snapshots", filename)
        if self.env_ticks == 0:
            self.images.append(self.export_as_image())
            self.export_to_png(filename)
            return

        elif self.env_ticks >= len(self.handler.tick_snapshots):
            print("FINISHED")
            hide_widget(self.tick_btn)

        else:
            tick_snapshot = self.handler.tick_snapshots[self.env_ticks]
            altitude_map = tick_snapshot['altitude_map']
            worker_coords = tick_snapshot['worker_coords']

            if altitude_map is not None:
                self.altitude_map.update_map(altitude_map=altitude_map, worker_coords=worker_coords)
            self.export_to_png(filename)

        self.env_ticks += 1


class SCKivyApp(App):
    def __init__(self, **kwargs):
        super(SCKivyApp, self).__init__(**kwargs)
        self.main_layout = None

    def build(self):
        # data_path = os.path.join("./3x3_test.npy")
        data_path = os.path.join("./export_data_big.npy")
        handler = EnvHandler(data_path)
        self.main_layout = MainUI(handler=handler,
                                  size_hint=(None, None),
                                  size=Window.size,
                                  color=CONFIG['main_bg_color']
                                  )
        self.main_layout.tick_btn.bind(on_press=self.main_layout.tick_test)
        return self.main_layout


if __name__ == '__main__':
    Builder.load_string(KV_STRING)
    SCKivyApp().run()
