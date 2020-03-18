import kivy
from kivy.config import Config

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', 1000)
Config.set('graphics', 'height', 600)

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.graphics.vertex_instructions import Rectangle, Ellipse
from kivy.graphics import Color, Line
from kivy.properties import *
import kivy.properties as props
from kivy.metrics import dp, sp
from PIL import Image, ImageDraw, ImageFilter
import colour as clr
from copy import copy, deepcopy
from kivy.clock import Clock

WORKER_COLORS = [
    '#5026a7',
    '#247BA0',
    '#8d448b',
    '#70C1B3',
    "#ffb947",
    "#ff7256",
    "#aaaaaa",
    "#880088",
    '#FF1654',
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


# <ColoredRectangle>:
#     size_hint: None, None
#     size: self.size
#     pos: self.pos
# 
#     canvas.after:
#         Color:
#             rgb: root.color
#         Rectangle:
#             size: root.size
#             pos: root.pos
"""


# <LabelWidget>:
#     size_hint: None, None
#     size: self.size
#     pos: self.pos
#
#     Label:
#         pos: root.pos
#         size: self.size
#         text_size: self.size
#         text: root.label_text
#         color: root.text_color
#         halign: 'left'

def get_rgba_color(val,
                   a: float = 1.0):
    temp = clr.Color(val).get_rgb()
    temp = list(temp) + [a]
    return tuple(temp)


CONFIG = {
    'map_bg_color': get_rgba_color('#B8B8B8'),
    'main_bg_color': get_rgba_color('white'),
    'window_padding_x': dp(25),
    'window_padding_y': dp(25),
    'cell_border_width': dp(0),
    'cell_border_color': get_rgba_color('black', a=0.3),
    'label_kwargs': dict(text='None',
                         color=get_rgba_color('black', a=0.9),
                         halign='left',
                         valign='middle',
                         ),

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

        self.canvas.before.add(Color(*self.bg_color))
        if self.with_shadow:
            self._create_shadow()
            self.canvas.before.add(Rectangle(size=self.shadow_size_1, pos=self.shadow_pos_1, texture=self.shadow_texture_1))
            self.canvas.before.add(Rectangle(size=self.shadow_size_2, pos=self.shadow_pos_2, texture=self.shadow_texture_2))
            self.canvas.before.add(Color(*self.bg_color))
            self.canvas.before.add(Rectangle(size=self.size, pos=self.pos))

        if self.with_border:
            self.canvas.add(
                Line(width=CONFIG['cell_border_width'], rectangle=[self.x, self.y, self.width, self.height]))

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


class LabelWidget(Widget):
    hidden = NumericProperty(0)
    bg_color = ListProperty([])

    def __init__(self, label_kwargs, **kwargs):
        super(LabelWidget, self).__init__(**kwargs)
        if len(self.bg_color) > 0:
            with self.canvas.before:
                self.fill_color = Color(rgba=self.bg_color)
                self.rectangle = Rectangle(size=self.size, pos=self.pos)

        self.label = Label(pos=self.pos,
                           size=self.size,
                           # text_size=self.size,
                           **label_kwargs)
        self.add_widget(self.label)

    def update_label_text(self, new_val):
        self.label.text = str(new_val)


class ColoredRectangle(Widget):
    hidden = NumericProperty(0)
    color = ListProperty(None)  # background color of the rectangle

    def __init__(self, **kwargs):
        super(ColoredRectangle, self).__init__(**kwargs)

        with self.canvas:
            self.fill_color = Color(rgb=self.color)
            self.rect = Rectangle(size=self.size, pos=self.pos)


class CustomButton(Button):
    hidden = NumericProperty(0)
    font_size = NumericProperty(dp(15))

    def __init__(self, **kwargs):
        super(CustomButton, self).__init__(**kwargs)
