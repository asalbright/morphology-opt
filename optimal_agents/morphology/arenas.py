import random
import numpy as np
import scipy
from dm_control import mjcf

from dm_control import composer
from dm_control.composer import variation
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
from dm_control.mujoco.wrapper import mjbindings
from scipy import ndimage
from scipy import stats

mjlib = mjbindings.mjlib

_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1

# Constants related to terrain generation.
_TERRAIN_SMOOTHNESS = .5  # 0.0: maximally bumpy; 1.0: completely smooth.
_TERRAIN_BUMP_SCALE = .2  # Spatial scale of terrain bumps (in meters).

_SIDE_WALLS_GEOM_GROUP = 3
_CORRIDOR_X_PADDING = 2.0
_WALL_THICKNESS = 0.16
_SIDE_WALL_HEIGHT = 4.0
_DEFAULT_ALPHA = 0.5

class EmptyCorridor(object):
    def __init__(self, corridor_width=4, corridor_length=40, visible_side_planes=False):
        self._corridor_width = corridor_width
        self._corridor_length = corridor_length
        self.visible_side_planes = visible_side_planes
        
    
    def construct(self, random_state=np.random.RandomState(12345)):
        self.arena = mjcf.RootElement(model="morphology")

        self._walls_body = self.arena.worldbody.add('body', name='walls')

        self.arena.visual.map.znear = 0.0005
        self.arena.asset.add('texture', type='skybox', builtin='gradient', rgb1=[0.4, 0.6, 0.8], rgb2=[0, 0, 0], width=100, height=600)
        self.arena.visual.headlight.set_attributes(ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8], specular=[0.1, 0.1, 0.1])

        alpha = _DEFAULT_ALPHA if self.visible_side_planes else 0.0
        self._ground_plane = self.arena.worldbody.add('geom', type='plane', rgba=[0.5, 0.5, 0.5, 1], size=[self._corridor_length, self._corridor_width, 1])
        self._left_plane = self.arena.worldbody.add('geom', type='plane', xyaxes=[1, 0, 0, 0, 0, 1], size=[self._corridor_length, self._corridor_width, 1], rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
        self._right_plane = self.arena.worldbody.add('geom', type='plane', xyaxes=[-1, 0, 0, 0, 0, 1], size=[self._corridor_length, self._corridor_width, 1], rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
        self._near_plane = self.arena.worldbody.add('geom', type='plane', xyaxes=[0, 1, 0, 0, 0, 1], size=[self._corridor_length, self._corridor_width, 1], rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
        self._far_plane = self.arena.worldbody.add('geom', type='plane', xyaxes=[0, -1, 0, 0, 0, 1], size=[self._corridor_length, self._corridor_width, 1], rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)

        self._walls_body.geom.clear()

        corridor_width = variation.evaluate(self._corridor_width,
                                        random_state=random_state)
        corridor_length = variation.evaluate(self._corridor_length,
                                            random_state=random_state)
        self._current_corridor_length = corridor_length
        self._current_corridor_width = corridor_width

        self._ground_plane.pos = [corridor_length / 2, 0, 0]
        self._ground_plane.size = [corridor_length / 2 + _CORRIDOR_X_PADDING, corridor_width / 2, 1]

        self._left_plane.pos = [corridor_length / 2, corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
        self._left_plane.size = [corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

        self._right_plane.pos = [corridor_length / 2, -corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
        self._right_plane.size = [corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

        self._near_plane.pos = [-_CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
        self._near_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

        self._far_plane.pos = [corridor_length + _CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
        self._far_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

        return self.arena

class GapsCorridor(EmptyCorridor):

    def __init__(self,
                platform_length=1.,
                gap_length=2.5,
                corridor_width=4,
                corridor_length=40,
                ground_rgba=(0.5, 0.5, 0.5, 1),
                visible_side_planes=False):

        super().__init__(
            corridor_width=corridor_width,
            corridor_length=corridor_length,
            visible_side_planes=visible_side_planes)

        self._platform_length = platform_length
        self._gap_length = gap_length
        self._ground_rgba = ground_rgba

    def construct(self, random_state=np.random.RandomState(12345)):
        super().construct(random_state)

        self._ground_body = self.arena.worldbody.add('body', name='ground')

        # Move the ground plane down and make it invisible.
        self._ground_plane.pos = [self._corridor_length / 2, 0, -10]
        self._ground_plane.rgba = [0, 0, 0, 0]

        # Clear the existing platform pieces.
        self._ground_body.geom.clear()

        # Make the first platform larger.
        platform_length = 3 * _CORRIDOR_X_PADDING
        platform_pos = [platform_length / 2, 0, -_WALL_THICKNESS]
        platform_size = [platform_length / 2, self._corridor_width / 2, _WALL_THICKNESS]

        self._ground_body.add('geom', type='box', rgba=variation.evaluate(self._ground_rgba, random_state), name='start_floor', pos=platform_pos, size=platform_size)

        current_x = platform_length
        platform_id = 0
        while current_x < self._corridor_length:
            platform_length = variation.evaluate(self._platform_length, random_state=random_state)
            platform_pos = [current_x + platform_length / 2., 0, -_WALL_THICKNESS]
            platform_size = [platform_length / 2, self._corridor_width / 2, _WALL_THICKNESS]
            
            self._ground_body.add('geom', type='box', rgba=variation.evaluate(self._ground_rgba, random_state), name='floor_{}'.format(platform_id), pos=platform_pos, size=platform_size)

            platform_id += 1

            # Move x to start of the next platform.
            current_x += platform_length + variation.evaluate(
                self._gap_length, random_state=random_state)

        return self.arena

class WallsCorridor(EmptyCorridor):
  """A corridor obstructed by multiple walls aligned against the two sides."""

  def __init__(self,
             wall_gap=2.5,
             wall_width=2.5,
             wall_height=2.0,
             swap_wall_side=True,
             wall_rgba=(1, 1, 1, 1),
             corridor_width=4,
             corridor_length=40,
             visible_side_planes=False,
             include_initial_padding=True):
    super().__init__(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes)

    self._wall_height = wall_height
    self._wall_rgba = wall_rgba
    self._wall_gap = wall_gap
    self._wall_width = wall_width
    self._swap_wall_side = swap_wall_side
    self._include_initial_padding = include_initial_padding

  def construct(self, random_state=np.random.RandomState(12345)):
    super().construct(random_state)

    wall_x = variation.evaluate(
        self._wall_gap, random_state=random_state) - _CORRIDOR_X_PADDING
    if self._include_initial_padding:
      wall_x += 2*_CORRIDOR_X_PADDING
    wall_side = 0
    wall_id = 0
    while wall_x < self._current_corridor_length:
      wall_width = variation.evaluate(self._wall_width, random_state=random_state)
      wall_height = variation.evaluate(self._wall_height, random_state=random_state)
      wall_rgba = variation.evaluate(self._wall_rgba, random_state=random_state)
      if variation.evaluate(self._swap_wall_side, random_state=random_state):
        wall_side = 1 - wall_side

      wall_pos = [wall_x, (2 * wall_side - 1) * (self._current_corridor_width - wall_width) / 2, wall_height / 2]
      wall_size = [_WALL_THICKNESS / 2, wall_width / 2, wall_height / 2]
      self._walls_body.add('geom', type='box', name='wall_{}'.format(wall_id), pos=wall_pos, size=wall_size, rgba=wall_rgba)

      wall_id += 1
      wall_x += variation.evaluate(self._wall_gap, random_state=random_state)

    return self.arena

class HurdlesCorridor(EmptyCorridor):
    def __init__(self,
                hurdle_height=0.25,
                gap_length=2.5,
                corridor_width=4,
                corridor_length=40,
                hurdle_rgba=(0.4, 0.4, 0.4, 1),
                visible_side_planes=False):
        
        self._hurdle_height = hurdle_height
        self._gap_length = gap_length
        self._hurdle_rgba = hurdle_rgba

        super().__init__(corridor_width=corridor_width,
                         corridor_length=corridor_length,
                         visible_side_planes=visible_side_planes)

    def construct(self, random_state=np.random.RandomState(12345)):
        super().construct(random_state)

        self._hurdles_body = self.arena.worldbody.add('body', name='hurdle')

        # Move the ground plane down and make it invisible.
        # self._ground_plane.pos = [self._corridor_length / 2, 0, -10]
        # self._ground_plane.rgba = [0, 0, 0, 0]

        # Clear the existing platform pieces.
        self._hurdles_body.geom.clear()

        current_x = 1
        hurdle_id = 0
        while current_x < self._corridor_length:
            gap_length = variation.evaluate(self._gap_length, random_state=random_state)
            # vary the height of the hurdles between 0.25 and 0.75 of self._hurdle_height
            hurdle_height = self._hurdle_height*np.random.uniform(0.25, 0.75)
            hurdle_pos = [current_x + gap_length / 2., 0, hurdle_height / 2.]
            hurdle_size = [_WALL_THICKNESS, self._corridor_width / 2, hurdle_height]
            
            self._hurdles_body.add('geom', type='box', rgba=variation.evaluate(self._hurdle_rgba, random_state), name='hurdle_{}'.format(hurdle_id), pos=hurdle_pos, size=hurdle_size)

            hurdle_id += 1

            # Move x to start of the next platform.
            current_x += gap_length + variation.evaluate(
                self._gap_length, random_state=random_state)

        return self.arena


class GM_Terrain(object):

    def __init__(self, n=8, max_height=0.2):
        self.n = n
        self.max_height = max_height
        self.morphology_height = 0.16

    def construct(self):
        max_spread = 0.5
        min_var, max_var = 0.2, 0.8
        min_pos, max_pos = -4, 4
        samples = 60
        pdfs = [stats.norm(random.uniform(min_pos, max_pos), random.uniform(min_var, max_var)) for _ in range(self.n)]
        x = np.linspace(min_pos, max_pos, num=samples)
        y = np.zeros(x.shape)
        for pdf in pdfs:
            y += pdf.pdf(x)
        y /= np.max(y)
        y *= self.max_height

        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")

        width = (max_pos - min_pos)/(2*samples)
        for i, xval, yval in zip(range(samples), x, y):
            geom_name = "terrain" + str(i)
            size = [width, 0.25, yval/2]
            pos = [xval, 0, yval/2]
            arena.worldbody.add('geom', name=geom_name, type='box', size=size, pos=pos, rgba=(.4, .4, .4, 1))

        return arena

class GM_Terrain3D(object):

    def __init__(self, n=10, max_height=0.25):
        self.n = n
        self.max_height = max_height
        self.morphology_height = 0.18

    def construct(self):
        max_spread = 0.5
        min_var, max_var = 0.3, 1.2
        min_pos, max_pos = -4, 4
        samples = 30
        
        pdfs = []
        for _ in range(self.n):
            mean = np.array([random.uniform(min_pos, max_pos), random.uniform(min_pos, max_pos)])
            v1, v2, cov = 0, 0, 100
            while abs(cov) > v1 or abs(cov) > v2:
                v1, v2, cov = random.uniform(min_var, max_var), random.uniform(min_var, max_var), (random.random()-0.5) * random.uniform(min_var, max_var)
            
            cov_mat = np.array([[v1, cov], [cov, v2]])
            pdfs.append(stats.multivariate_normal(mean, cov_mat))
        
        x = np.linspace(min_pos, max_pos, num=samples)
        y = np.linspace(min_pos, max_pos, num=samples)
        xx, yy = np.meshgrid(x, y, sparse=False)
        z = np.zeros((x.shape[0], y.shape[0]))
        pos = np.dstack((xx, yy))
        for pdf in pdfs:
            z += pdf.pdf(pos)
        z /= np.max(z)
        z *= self.max_height
        width = (max_pos - min_pos)/(2*samples)

        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")

        for i, xval, yval, zval in zip(range(samples**2), xx.flatten(), yy.flatten(), z.flatten()):
            geom_name = "terrain" + str(i)
            size = [width, width, zval/2]
            pos = [xval, yval, zval/2]
            arena.worldbody.add('geom', name=geom_name, type='box', size=size, pos=pos, rgba=(.4, .4, .4, 1))
        
        return arena

class ReachTarget(object):

    def construct(self):
        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")
        arena.worldbody.add('geom', name='target', type='sphere', size=(0.08,), pos=[0, 0, 0.04], rgba=(.8, .4, .4, .6), conaffinity=0, contype=0)
        return arena

class ReachBox(object):

    def construct(self):
        arena = mjcf.RootElement(model="morphology")
        arena.asset.add('texture', name="texplane", builtin='checker', height=300, rgb1=[0.1, 0.2, 0.3], rgb2=[0.2, 0.3, 0.4], type="2d", width=300)
        arena.asset.add('material', name="MatPlane", reflectance=0.5, shininess=1, specular=1, texrepeat=[20, 20], texture="texplane")
        arena.worldbody.add('light', cutoff=100, diffuse=[1, 1, 1], dir=[-0, 0, -1.3], directional=True, exponent=1, pos=[0,0,1.5], specular=[.1,.1,.1])
        arena.worldbody.add('geom', name='ground', type='plane', size=[15, 15, .5], rgba=(1, 1, 1, 1), material="MatPlane")
        
        block1 = arena.worldbody.add('body', name='block1', pos=[0.9, 0.65, 0.0])
        block1.add('geom', name='block1', type='box', size=[0.1, 0.1, 0.1], pos=[0, 0, 0.1], rgba=(.8, .3, .3, 1), contype=1, conaffinity=1, condim=3)
        block1.add('joint', name='block1x', type='slide', axis=[1, 0, 0], limited=False, damping=0, armature=0, stiffness=0)
        block1.add('joint', name='block1y', type='slide', axis=[0, 1, 0], limited=False, damping=0, armature=0, stiffness=0)
        block1.add('joint', name='block1z', type='slide', axis=[0, 0, 1], limited=True, damping=0, armature=0, stiffness=0, range=(-0.02, 0.02))

        block2 = arena.worldbody.add('body', name='block2', pos=[0.9, -0.65, 0])
        block2.add('geom', name='block2', type='box', size=[0.1, 0.1, 0.1], pos=[0, 0, 0.1], rgba=(.8, .3, .3, 1), contype=1, conaffinity=1, condim=3)
        block2.add('joint', name='block2x', type='slide', axis=[1, 0, 0], limited=False, damping=0, armature=0, stiffness=0)
        block2.add('joint', name='block2y', type='slide', axis=[0, 1, 0], limited=False, damping=0, armature=0, stiffness=0)
        block2.add('joint', name='block2z', type='slide', axis=[0, 0, 1], limited=True, damping=0, armature=0, stiffness=0, range=(-0.02, 0.02))

        return arena
