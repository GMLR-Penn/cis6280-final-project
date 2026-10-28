"""Volumetric elastic cuboids and ellipsoids; preserves the v0.1 cube environment."""
from dataclasses import dataclass, replace
from itertools import permutations, product
import json
import xml.etree.ElementTree as ET

import numpy as np

from .sharpa import ROOT
from .sharpa_policy import SharpaCollector
from .soft_cube import SoftSceneConfig, SharpaSoftCubeEnv


# Full outer dimensions in world XYZ, meters; the grasp span is a scripted guess.
SHAPES = {
    'cube': ('cuboid', (.060,.060,.060), .060),
    'long': ('cuboid', (.045,.090,.045), .050),
    'short': ('cuboid', (.045,.045,.045), .045),
    'flat': ('cuboid', (.070,.060,.025), .060),
    'ellipsoid': ('ellipsoid', (.055,.085,.045), .045),
}


@dataclass
class ShapeSceneConfig(SoftSceneConfig):
    physics_dt: float = .00025
    shape: str = 'cuboid'
    dimensions: tuple = (.060,.060,.060)
    grasp_span: float = .060


def shape_config(label, seed=0, young=100000., physics_dt=.00025):
    shape,dimensions,span = SHAPES[label]
    return replace(ShapeSceneConfig.from_seed(seed),cube_edge=dimensions[2],cube_yaw=0.,
                   shape=shape,dimensions=dimensions,grasp_span=span,young=young,physics_dt=physics_dt,
                   grid_count=5 if shape=='ellipsoid' else 4)


def solid_mesh(c):
    """Conforming six-tet subdivision of each grid cell; ellipsoid is a mapped solid grid.

    The smooth cube-to-sphere map sends each outer face to the unit sphere.
    Odd resolution includes axis extrema. All interior vertices remain free.
    """
    dims = np.asarray(c.dimensions,dtype=float)
    n = c.grid_count
    points = np.array(list(product(np.linspace(-1,1,n),repeat=3)))
    if c.shape=='ellipsoid':
        squared = points**2
        points *= np.sqrt(1-(squared[:,[1,2,0]]+squared[:,[2,0,1]])/2
                          +squared[:,[1,2,0]]*squared[:,[2,0,1]]/3)
    points *= (dims-2*c.contact_radius)/2
    elements = []
    for origin in product(range(n-1),repeat=3):
        for axes in permutations(range(3)):
            corner = np.array(origin); ids = [np.ravel_multi_index(corner,(n,n,n))]
            for axis in axes:
                corner = corner.copy();corner[axis]+=1
                ids.append(np.ravel_multi_index(corner,(n,n,n)))
            elements.append(ids)
    elements = np.asarray(elements)
    tetra = points[elements]
    determinants = np.linalg.det(np.transpose(tetra[:,1:]-tetra[:,:1],(0,2,1)))
    swap = determinants<0
    elements[swap,1],elements[swap,2] = elements[swap,2].copy(),elements[swap,1].copy()
    assert np.min(np.abs(determinants))>1e-12
    return points,elements


class SharpaSoftShapeEnv(SharpaSoftCubeEnv):
    def __init__(self,seed=0,resolution=192,config=None,*,shape='cube',young=100000.,dimensions=None,grasp_span=None):
        c = config or shape_config(shape,seed,young)
        if dimensions is not None:
            if len(dimensions)!=3:raise ValueError('dimensions must be full XYZ lengths in meters')
            c = replace(c,dimensions=tuple(dimensions),cube_edge=dimensions[2])
        if grasp_span is not None:c = replace(c,grasp_span=grasp_span)
        dims = np.asarray(c.dimensions,dtype=float)
        if dims.shape!=(3,) or not np.isfinite(dims).all() or np.any(dims<=2*c.contact_radius):
            raise ValueError('dimensions must contain three positive full lengths larger than the contact diameter')
        if c.shape not in ('cuboid','ellipsoid') or not np.isclose(c.cube_edge,dims[2]):
            raise ValueError('Use cuboid or ellipsoid, with cube_edge equal to object height for the inherited controller')
        if c.shape=='ellipsoid' and c.grid_count%2!=1:
            raise ValueError('Use an odd ellipsoid grid_count to include all six axis extrema')
        if not np.isfinite(c.grasp_span) or not .045<=c.grasp_span<=.070:
            raise ValueError('The example grasp span must lie in the existing 45–70 mm profile range')
        super().__init__(resolution=resolution,config=c)

    @staticmethod
    def build_xml(c):
        root = ET.fromstring(SharpaSoftCubeEnv.build_xml(c))
        root.set('model','sharpa_elastic_shapes_v0')
        flex = root.find("worldbody/flexcomp[@name='soft_cube']")
        points,elements = solid_mesh(c)
        flex.set('type','direct')
        flex.set('flatskin','true' if c.shape=='cuboid' else 'false')
        for attr in ('count','spacing'):flex.attrib.pop(attr,None)
        flex.set('point',' '.join(format(x,'.17g') for x in points.ravel()))
        flex.set('element',' '.join(map(str,elements.ravel())))
        return ET.tostring(root,encoding='unicode')


class SoftShapeCollector(SharpaCollector):
    """Whole-hand scripted profile, with grasp height and span supplied by geometry.

    fixed60 retains the 60 mm cube's target stream as a diagnostic baseline.
    geometry chooses a known span profile and sets the vertical target from height.
    Neither variant is learned, visual, or a tuned shape-specific grasp planner.
    """
    pose_path = 'assets/sharpa-whole-hand-grasp.json'

    def __init__(self,env,mode,seed,grip_scale=1.,policy='geometry'):
        if policy not in ('geometry','fixed60') or not 0<=grip_scale<=1:
            raise ValueError('Use geometry/fixed60 policy and grip_scale in [0,1]')
        self.policy = policy; self.grip_scale = grip_scale
        self.target_height = env.config.cube_edge if policy=='geometry' else .060
        span = env.config.grasp_span if policy=='geometry' else .060
        calibration = json.loads((ROOT/'assets/sharpa-size-grasps.json').read_text())
        assert calibration['joint_names']==env.joint_names[3:]
        profiles = calibration['profiles']; sizes = np.array([r['edge_mm'] for r in profiles])/1000
        self.profile = {key:np.array([np.interp(span,sizes,column)
                                     for column in np.asarray([r[key] for r in profiles]).T])
                        for key in ('grasp_center_local','open','contact','closed')}
        env.grasp_center = self.profile['grasp_center_local'].copy()
        super().__init__(env,mode,seed)
        for key in ('open','contact','closed'):setattr(self,key,self.profile[key].copy())
        self.closed = self.contact+grip_scale*(self.closed-self.contact)
        self.stages = [(name,30 if name=='settle' else length) for name,length in self.stages]
        self.horizon = sum(n for _,n in self.stages)
        env.reset(fingers=self.open)

    def action(self,env,t):
        # The base action method uses cube_edge only for grasp/release height.
        original = env.config
        try:
            env.config = replace(original,cube_edge=self.target_height)
            return super().action(env,t)
        finally:env.config = original
