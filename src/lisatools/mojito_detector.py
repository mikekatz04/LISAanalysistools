"""Mojito-specific Orbits implementation with JAX."""

import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d as np_gaussian_filter1d
from typing import Optional
from copy import deepcopy
import jax 
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
except (ModuleNotFoundError, ImportError):
    import numpy as cp
    cp_gaussian_filter1d = np_gaussian_filter1d

from cudakima import AkimaInterpolant1D

from lisaconstants.indexing import link2sc
from .detector import Orbits, LINEAR_INTERP_TIMESTEP
from .utils.parallelbase import LISAToolsParallelModule
from .domains import DomainSettingsBase
from .sensitivity import SensitivityMatrixBase

NUM_SPLINE_THREADS = 256


# @jax.jit
# def interpolate_pos(query_t, sc_idx, t_grid, pos_grid):
#     """
#     Interpolate spacecraft position using JAX.
    
#     Args:
#         query_t: shape (N,)
#         sc_idx: shape (N,) or (1,) - 0-based index (0, 1, 2)
#         t_grid: shape (T_dense,)
#         pos_grid: shape (T_dense, 3_sc, 3_coords)
#     """

#     def _single_point(t, sc):
#         # t: scalar float
#         # sc: scalar int (0, 1, 2)
        
#         # pos_grid is (Times, SC, 3)
#         # We want to interpolate along axis 0.
        
#         # Helper for 1D interp
#         def interp_1d(vals):
#             return jnp.interp(t, t_grid, vals)
            
#         # Select the trajectory for this spacecraft
#         # pos_grid[:, sc, 0] gives x(t) for spacecraft sc
#         # We use simple indexing. Since 'sc' is a tracer in JIT, simple indexing might need dynamic_slice 
#         # or just work if shape is static. JAX numpy indexing usually works fine.
        
#         val_x = interp_1d(pos_grid[:, sc, 0])
#         val_y = interp_1d(pos_grid[:, sc, 1])
#         val_z = interp_1d(pos_grid[:, sc, 2])
#         return jnp.array([val_x, val_y, val_z])

#     # If inputs are arrays, we vmap. 
#     # query_t is (N,), sc_idx is (3,) or (1,)
#     return jax.vmap(jax.vmap(_single_point, in_axes=(None, 0)), in_axes=(0, None))(query_t, sc_idx)

# @jax.jit
# def interpolate_ltt(query_t, link_idx, t_grid, ltt_grid):
#     """
#     Interpolate Light Travel Time using JAX.
    
#     Args:
#         query_t: shape (N,)
#         link_idx: shape (N_links,) or (1,) - 0-based index
#         t_grid: shape (T_dense,)
#         ltt_grid: shape (T_dense, N_links)
#     """
#     def _single_point(t, l):
#         vals = ltt_grid[:, l]
#         return jnp.interp(t, t_grid, vals)
    
#     return jax.vmap(jax.vmap(_single_point, in_axes=(None, 0)), in_axes=(0, None))(query_t, link_idx)

# def icrs_to_ecliptic(positions_icrs):
#     """
#     Convert cartesian positions from ICRS to ecliptic coordinates.

#     Args:
#         positions_icrs: Array of shape (n_times, 3, 3) representing positions
#                         in ICRS frame for 3 spacecraft over n_times.
    
#     Returns:
#         positions_ecliptic: Array of shape (n_times, 3, 3) in ecliptic frame.
#     """

#     import astropy
#     positions_ecliptic = np.zeros_like(positions_icrs)
    
#     for sc in range(3):

#         c_icrs = astropy.coordinates.SkyCoord(positions_icrs[:,sc,0], positions_icrs[:,sc,1], positions_icrs[:,sc,2], frame='icrs', unit='m', representation_type='cartesian')
#         c_ecliptic = c_icrs.transform_to(astropy.coordinates.BarycentricMeanEcliptic)
#         c_ecliptic.representation_type='cartesian'
#         positions_ecliptic[:, sc, :] = np.array([c_ecliptic.x.value, c_ecliptic.y.value, c_ecliptic.z.value]).T

#     return positions_ecliptic

# class L1Orbits(Orbits):
#     """Base class for LISA Orbits from Mojito L1 File structure.
    
#     This class handles orbit data from MojitoL1File where:
#     - Light travel times and positions have different time arrays
#     - Both time arrays may start at t0 != 0
    
#     Args:
#         armlength: Armlength of detector (default 2.5e9 m)
#     """

#     def __init__(
#         self, 
#         filename: str,
#         armlength: float = 2.5e9,
#         force_backend: Optional[str] = None,
#         **kwargs
#     ):
#         # Store the Mojito file path
#         self.filename = filename
        
#         # Don't call super().__init__ - we need to override _setup
#         # Instead, manually initialize the minimal required attributes
#         self._armlength = armlength
#         self._filename = filename  # For compatibility
#         self.configured = False
        
#         # Load data from Mojito file
#         self._load_mojito_data()
        
#         # Initialize backend
#         from .utils.parallelbase import LISAToolsParallelModule
#         LISAToolsParallelModule.__init__(self, force_backend=force_backend)
        
#     def open(self):
#         """Override base class open method."""
#         try:
#             from mojito import MojitoL1File
#         except ImportError:
#             raise ImportError("mojito package required for L1Orbits. Follow instructions at: https://mojito-e66317.io.esa.int/content/installation.html")
#         f = MojitoL1File(self.filename)
#         return f            
    
#     def _load_mojito_data(self):
#         """Load orbit and LTT data from Mojito file."""
        
#         with self.open() as f:
#             # Load light travel times and their time array
#             self.ltt = f.ltts.ltts[:]  # Shape: (N_ltt_times, 6)
#             self.ltt_t = f.ltts.time_sampling.t()  # Shape: (N_ltt_times,)
            
#             # Load spacecraft positions and their time array  
#             pos_icrs = f.orbits.positions[:]  # Shape: (N_pos_times, 3, 3)
#             self.x_base = icrs_to_ecliptic(pos_icrs)  # Convert to ecliptic frame
#             self.v_base = f.orbits.velocities[:] # Shape: (N_pos_times, 3, 3)
#             self.sc_t_base = f.orbits.time_sampling.t()  # Shape: (N_pos_times,)
            
#             # Store dt from each dataset (they may differ)
#             self.ltt_dt = f.ltts.time_sampling.dt
#             self.sc_dt = f.orbits.time_sampling.dt
            
#             # Store t0 values
#             self.ltt_t0 = float(self.ltt_t[0])
#             self.sc_t0 = float(self.sc_t_base[0])
    
#     @property
#     def ltt_t(self):
#         """LTT file time."""
#         return self._ltt_t
    
#     @ltt_t.setter
#     def ltt_t(self, x):
#         self._ltt_t = x
    
#     @property
#     def ltt(self):
#         """Light travel times from Mojito file."""
#         return self._ltt

#     @ltt.setter
#     def ltt(self, x):
#         self._ltt = x
    
#     @property
#     def x_base(self):
#         """Spacecraft positions from Mojito file."""
#         return self._x_base
#     @x_base.setter
#     def x_base(self, x):
#         self._x_base = x

#     @property
#     def v_base(self):
#         """Velocities from Mojito file."""
#         return self._v_base
#     @v_base.setter
#     def v_base(self, x):
#         self._v_base = x

#     @property
#     def sc_t_base(self):
#         """Spacecraft position file time."""
#         return self._sc_t_base
#     @sc_t_base.setter
#     def sc_t_base(self, x):
#         self._sc_t_base = x
    
#     @property
#     def ltt_dt(self):
#         """Time step of LTT data."""
#         return self._ltt_dt
#     @ltt_dt.setter
#     def ltt_dt(self, x):
#         self._ltt_dt = x
    
#     @property
#     def sc_dt_base(self):    
#         """Time step of spacecraft position data."""
#         return self._sc_dt_base
#     @sc_dt_base.setter
#     def sc_dt_base(self, x):
#         self._sc_dt_base = x
    
#     @property
#     def ltt_t0(self):
#         """Start time of LTT data."""
#         return self._ltt_t0
#     @ltt_t0.setter
#     def ltt_t0(self, x):
#         self._ltt_t0 = x    

#     @property
#     def sc_t0(self):
#         """Start time of spacecraft position data."""
#         return self._sc_t0
#     @sc_t0.setter
#     def sc_t0(self, x):
#         self._sc_t0 = x

#     @property
#     def sc_dt(self):    
#         """Time step of spacecraft position data."""
#         return self._sc_dt
#     @sc_dt.setter
#     def sc_dt(self, x):
#         self._sc_dt = x

#     @property
#     def sc_t(self):
#         """Configured spacecraft time array."""
#         return self._sc_t
#     @sc_t.setter
#     def sc_t(self, x):
#         self._sc_t = x
    
#     @property
#     def t(self):
#         """Configured time array (spacecraft positions)."""
#         return self._sc_t

#     @property
#     def x(self):
#         """Configured spacecraft positions."""
#         self._check_configured()
#         return self._x
#     @x.setter
#     def x(self, x):
#         self._x = x
    
#     @property
#     def v(self):
#         """Configured spacecraft velocities."""
#         self._check_configured()
#         return self._v
#     @v.setter
#     def v(self, x):
#         self._v = x
    
#     @property
#     def n(self):
#         return self._n
#     @n.setter
#     def n(self, x):
#         self._n = x

#     # @property
#     # def LINKS(self):
#     #     """Link IDs in Mojito convention."""
#     #     return lisaconstants.indexing.LINKS

#     # @property
#     # def SC(self):
#     #     """Spacecraft IDs in Mojito convention."""
#     #     return lisaconstants.indexing.SPACECRAFT

    
#     @property
#     def pycppdetector(self) -> object:
#         """C++ class"""
#         if self._pycppdetector_args is None:
#             raise ValueError(
#                 "Asking for c++ class. Need to set linear_interp_setup = True when configuring."
#             )
#         self._pycppdetector = self.backend.OrbitsWrap(*self._pycppdetector_args)

#         return self._pycppdetector
    
#     def configure(
#         self,
#         t_arr=None,
#         dt=None, 
#         linear_interp_setup=False
#     ):
#         """Configure orbits with interpolation to a target time grid.
        
#         This handles the fact that ltts and positions come from different
#         time arrays in the Mojito file.
        
#         Args:
#             t_arr: Target time array (if None, will be constructed)
#             dt: Target time step
#             linear_interp_setup: If True, create dense grid for fast linear interpolation
#         """
        
#         # Determine target time array
#         if linear_interp_setup:
#             make_cpp = True
#             dt = dt if dt is not None else LINEAR_INTERP_TIMESTEP
#             # interpolate only the orbit quantities, the ltts are already dense enough
#             t0 = self.sc_t0
#             t_end = float(self._sc_t_base[-1])
                        
#             t_arr = np.arange(t0, t_end + dt, dt)
#             t_arr = t_arr[t_arr <= t_end]
            
#         elif t_arr is None:
#             if dt is not None:
#                 # Use position time range as reference
#                 t0 = self.t0_pos
#                 t_end = float(self._pos_times[-1])
#                 t_arr = np.arange(t0, t_end + dt, dt)
#                 t_arr = t_arr[t_arr <= t_end]
#                 make_cpp = True
#             else:
#                 # Use position time array as default
#                 t_arr = self._sc_t_base.copy()
#                 make_cpp = False
#         else:
#             make_cpp = True
                
#         # Interpolate positions from their native time grid to target grid
#         # _pos_data is (N_pos, 3_sc, 3_xyz)
#         pos_interp_shape = (len(t_arr), 3, 3)
#         pos_interpolated = np.zeros(pos_interp_shape)
#         vel_interpolated = np.zeros(pos_interp_shape)
        
#         for isc in range(3):  # 3 spacecraft
#             for icoord in range(3):  # x, y, z
#                 # Cubic spline interpolation
#                 cs = interpolate.CubicSpline(
#                     self.sc_t_base, 
#                     self.x_base[:, isc, icoord]
#                 )
#                 pos_interpolated[:, isc, icoord] = cs(t_arr)

#                 #interpolate velocities as well
#                 cs = interpolate.CubicSpline(
#                     self.sc_t_base,
#                     self.v_base[:, isc, icoord]
#                 )
#                 vel_interpolated[:, isc, icoord] = cs(t_arr)
        
       
#         # Store interpolated data
#         self.sc_dt = dt
#         self.sc_t = self.xp.asarray(t_arr)
#         self.x = self.xp.asarray(pos_interpolated)
#         self.v = self.xp.asarray(vel_interpolated)
#         self.n = self.xp.zeros((len(t_arr), 18))

#         # make sure base spacecraft and link inormation is ready
#         lsr = np.asarray(self.link_space_craft_r).copy().astype(np.int32)
#         lse = np.asarray(self.link_space_craft_e).copy().astype(np.int32)
#         ll = np.asarray(self.LINKS).copy().astype(np.int32)
                
#         # Mark as configured
#         self.configured = True

#         if make_cpp:
#             self.pycppdetector_args = [
#                 self.sc_t0, # spacecraft time start
#                 self.sc_dt,   # spacecraft time step
#                 len(self.sc_t), # number of spacecraft time points
#                 self.ltt_t0,    # ltt time start
#                 self.ltt_dt,    # ltt time step
#                 len(self.ltt_t), # number of ltt time points
#                 self.xp.asarray(self.n.flatten().copy()),
#                 self.xp.asarray(self.ltt.flatten().copy()),
#                 self.xp.asarray(self.x.flatten().copy()),
#                 self.xp.asarray(ll),
#                 self.xp.asarray(lsr),
#                 self.xp.asarray(lse),
#                 self.armlength,
#             ]
#         else:
#             self.pycppdetector_args = None


#     def _setup(self):
#         """Override base class _setup - we load data in `_load_mojito_data` instead."""
#         pass
    

# @jax.tree_util.register_pytree_node_class
# class JAXL1Orbits(L1Orbits):
#     """LISA Orbits from Mojito L1 File structure with JAX support.
    
#     This class handles orbit data from MojitoL1File where:
#     - Light travel times and positions have different time arrays
#     - Both time arrays may start at t0 != 0
    
#     Uses Numba CUDA kernels for fast GPU interpolation, with automatic
#     fallback to CPU if CUDA is not available.
    
#     Args:
#         filename: Path to Mojito L1 HDF5 file
#         armlength: Armlength of detector (default 2.5e9 m)
#         force_backend: If 'gpu' or 'cuda', use GPU; if 'cpu', use CPU
#     """
    
    
#     def configure(self, t_arr=None, dt=None, linear_interp_setup=False):
#         super().configure(t_arr, dt, linear_interp_setup)
#         # No C++ backend for this implementation, ovverride attribute
#         self._pycppdetector_args = None

#         # move arrays to JAX
#         self.sc_t = jnp.asarray(self.sc_t)
#         self.x = jnp.asarray(self.x)
#         self.v = jnp.asarray(self.v)
#         self.n = jnp.asarray(self.n)

#         self.ltt_t = jnp.asarray(self.ltt_t)
#         self.ltt = jnp.asarray(self.ltt)
    
#     @property
#     def dt(self):
#         """Time step of configured grid."""
#         if not self.configured:
#             return self._dt
#         return self._dt
    
#     @dt.setter
#     def dt(self, dt):
#         self._dt = dt
    
#     def get_pos(self, t, sc):
#         """
#         Compute spacecraft position using Numba CUDA interpolation.
        
#         Args:
#             t: time (scalar, array, or list)
#             sc: spacecraft index (1, 2, or 3) or array of indices
#         """
#         if not self.configured:
#             raise RuntimeError("Must call configure() before get_pos()")

#         squeeze_t = jnp.isscalar(t)
#         squeeze_sc = jnp.isscalar(sc)

#         t_arr = jnp.atleast_1d(t)
#         sc_arr = (jnp.atleast_1d(sc) - 1).astype(int)

#         output = interpolate_pos(t_arr, sc_arr, self.sc_t, self.x)
        
#         if squeeze_sc:
#             output = output.squeeze(axis=1)
#         if squeeze_t:
#             output = output.squeeze(axis=0)
        
#         return output.block_until_ready()

    
#     def get_light_travel_times(self, t, link):
#         """
#         Compute light travel times using Numba CUDA interpolation.
        
#         Args:
#             t: time (scalar, array, or list)
#             link: link index (12, 23, 31, 13, 32, 21) or array of indices
#         """
#         if not self.configured:
#             raise RuntimeError("Must call configure() before get_light_travel_times()")
        
#         squeeze_t = jnp.isscalar(t)
#         squeeze_link = jnp.isscalar(link)

#         t_arr = jnp.atleast_1d(t)
#         link_arr = jnp.atleast_1d(link)
        
#         link_map_keys = jnp.array(self.LINKS)
#         link_map_vals = jnp.arange(len(self.LINKS))
        
#         def map_link(l):
#             # Find index where link_map_keys == l
#             # jnp.where returns (array_of_indices,)
#             # We take the first match. 
#             # Note: This might be slow inside vmap if not optimized, but for small list constant (6) it's fine.
#             # Faster: use a direct lookup array if link IDs were small integers, but they are 12, 23 etc.
#             # We can use a boolean mask.
#             return jnp.sum(link_map_vals * (link_map_keys == l))

#         link_idx = jax.vmap(map_link)(link_arr)
        
#         output = interpolate_ltt(t_arr, link_idx, self.ltt_t, self.ltt)

#         if squeeze_link:
#             output = output.squeeze(axis=1)
#         if squeeze_t:
#             output = output.squeeze(axis=0)
        
#         return output.block_until_ready()

#     def tree_flatten(self):
#         # Collect children (JAX arrays)
#         children = (
#             self.ltt,
#             self.ltt_t,
#             self.x_base,
#             self.v_base,
#             self.sc_t_base,
#         )
        
#         # If configured, add interpolated arrays
#         if self.configured:
#             children += (
#                 self.sc_t,
#                 self.x,
#                 self.v,
#                 self.n
#             )
        
#         # Collect aux_data (static configuration)
#         aux_data = {
#             'filename': self.filename,
#             'armlength': self._armlength,
#             'configured': self.configured,
#             'ltt_dt': self.ltt_dt,
#             'sc_dt': self.sc_dt,
#             'ltt_t0': self.ltt_t0,
#             'sc_t0': self.sc_t0,
#             # Capture other potential attributes
#             '_dt': getattr(self, '_dt', None),
#             '_t0': getattr(self, '_t0', None),
#             # Preserve backend info if needed (though usually static)
#             'use_gpu': getattr(self, 'use_gpu', False)
#         }
        
#         return (children, aux_data)

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         # Create empty instance without calling __init__
#         obj = object.__new__(cls)
        
#         # Restore aux_data
#         obj.filename = aux_data['filename']
#         obj._armlength = aux_data['armlength']
#         obj.configured = aux_data['configured']
#         obj.ltt_dt = aux_data['ltt_dt']
#         obj.sc_dt = aux_data['sc_dt']
#         obj.ltt_t0 = aux_data['ltt_t0']
#         obj.sc_t0 = aux_data['sc_t0']
#         obj.use_gpu = aux_data['use_gpu']
#         obj._filename = aux_data['filename']
        
#         if aux_data.get('_dt') is not None:
#             obj._dt = aux_data['_dt']
#         if aux_data.get('_t0') is not None:
#             obj._t0 = aux_data['_t0']

#         # Restore children
#         # Base arrays always present (first 5)
#         obj.ltt = children[0]
#         obj.ltt_t = children[1]
#         obj.x_base = children[2]
#         obj.v_base = children[3]
#         obj.sc_t_base = children[4]
        
#         if obj.configured:
#             # Configured arrays (next 4)
#             obj.sc_t = children[5]
#             obj.x = children[6]
#             obj.v = children[7]
#             obj.n = children[8]
        
#         # Initialize other attributes needed for method calls
#         obj._pycppdetector_args = None
        
#         return obj

# class XYZSensitivityBackend:
#     pass

# class XYZSensitivityBackend(LISAToolsParallelModule, SensitivityMatrixBase):
#     """Helper class for sensitivity matrix with c++ backend."""
    
#     def __init__(self, 
#                  orbits: L1Orbits,
#                  settings: DomainSettingsBase,
#                  tdi_generation: int = 2,
#                  use_splines: bool = False,
#                  force_backend: Optional[str] = 'cpu',
#                  mask_percentage: Optional[float] = None,
#                  ):
        
#         LISAToolsParallelModule.__init__(self, force_backend=force_backend)
#         SensitivityMatrixBase.__init__(self, settings)

#         assert self.backend.xp == orbits.xp, "Orbits and Sensitivity backend mismatch."
        
#         self.orbits = orbits
#         if not self.orbits.configured:
#             self.orbits.configure(linear_interp_setup=True)

#         self.tdi_generation = tdi_generation
#         self.channel_shape = (3, 3) 

#         _use_gpu = force_backend != 'cpu'

#         self.use_splines = use_splines
#         self.spline_interpolant = AkimaInterpolant1D(use_gpu=_use_gpu, threadsperblock=NUM_SPLINE_THREADS, order='cubic')

#         self.mask_percentage = mask_percentage if mask_percentage is not None else 0.05

#         self._setup()

#     @property
#     def xp(self):
#         """Array module."""
#         return self.backend.xp
    
#     @property
#     def time_indices(self):
#         return self._time_indices
#     @time_indices.setter
#     def time_indices(self, x):
#         self._time_indices = x
    
#     def get_averaged_ltts(self):
#         # first, compute the average ltts and their differences. 
#         # check if we need multiple time points
#         if hasattr(self.basis_settings, 't_arr'):
#             t_arr = self.xp.asarray(self.basis_settings.t_arr)
#             tiled_times = self.xp.tile(
#                 t_arr[:, self.xp.newaxis], (1, 6)
#             ).flatten()  # compute ltts at these times with orbits

#             links = self.xp.tile(self.xp.asarray(self.orbits.LINKS), (t_arr.shape[0],))

#             ltts = self.orbits.get_light_travel_times(
#                 tiled_times, links
#             ).reshape(len(t_arr), 6)

#             self.time_indices = self.xp.arange(len(t_arr), dtype=self.xp.int32)
        
#         else:
#             ltts = self.xp.mean(self.orbits.ltt, axis=0)[self.xp.newaxis, :]
#             self.time_indices = self.xp.array([0], dtype=self.xp.int32)

#         # with orbits.LINKS order: 12, 23, 31, 13, 32, 21, we need averages between pairs
#         # pairs: (12,21), (23,32), (31,13)
#         # Use direct indexing to avoid assignment issues with shape (1, 6) arrays
#         indices = [0, 1, 2, 3, 4, 5]
#         opposite_indices = indices[::-1]

#         avg_ltts = 0.5 * (ltts[:, indices] + ltts[:, opposite_indices])
#         delta_ltts = ltts[:, indices] - ltts[:, opposite_indices]

#         return avg_ltts, delta_ltts

#     def _setup(self):
#         """Setup the arguments for the c++ backend."""
        
#         avg_ltts, delta_ltts = self.get_averaged_ltts()
        
#         self.pycppsensmat_args = [
#             self.xp.asarray(avg_ltts.flatten().copy()),
#             self.xp.asarray(delta_ltts.flatten().copy()),
#             avg_ltts.shape[0],  # n_times
#             self.orbits.armlength,
#             self.tdi_generation,
#             self.use_splines,
#         ]

#         self.pycpp_sensitivity_matrix = self.backend.SensitivityMatrixWrap(*self.pycppsensmat_args)

#         self._init_basis_settings()

#     def __deepcopy__(self, memo):
#         """Custom deepcopy to handle unpicklable backend objects."""
#         from copy import copy
        
#         # Create a new instance without calling __init__
#         cls = self.__class__
#         new_obj = cls.__new__(cls)
        
#         # Copy the memo to avoid infinite recursion
#         memo[id(self)] = new_obj
        
#         # Manually copy attributes
#         for key, value in self.__dict__.items():
#             if key in ('_backend', 'pycpp_sensitivity_matrix'):
#                 # Don't deepcopy backend objects - just reference
#                 setattr(new_obj, key, value)
#             elif key == 'orbits':
#                 # Shallow copy orbits (share the same backend)
#                 setattr(new_obj, key, copy(value))
#             elif key == 'spline_interpolant':
#                 # Shallow copy spline interpolant
#                 setattr(new_obj, key, copy(value))
#             else:
#                 # Deepcopy everything else
#                 setattr(new_obj, key, deepcopy(value, memo))
        
#         return new_obj

#     def _init_basis_settings(self):
#         """Initialize basis settings from domain settings."""
#         self.f_arr = self.xp.asarray(self.basis_settings.f_arr)
        
#         if hasattr(self.basis_settings, 't_arr'):
#             self.t_arr = self.xp.asarray(self.basis_settings.t_arr)

#         self.num_times = len(self.t_arr) if hasattr(self, 't_arr') else 1
#         self.num_freqs = len(self.f_arr)

#         dips_indices = self._get_dips_indices()

#         dips_mask = self.xp.zeros((self.num_times, self.num_freqs), dtype=bool)
#         for t_idx in range(self.num_times):
#             dips_mask[t_idx, dips_indices[t_idx]] = True
        
#         self.dips_mask = dips_mask.flatten()

#     def _find_dips_with_percentage(self, tf, mask_percentage=0.05):

#         if hasattr(self.f_arr, 'get'):
#             f_arr = self.f_arr.get()
#             tf = tf.get()

#         peaks = find_peaks(-tf)[0]
        
#         all_indices = set()
#         for peak in peaks:
#             freq = self.f_arr[peak]
#             df = self.f_arr[1] - self.f_arr[0]
            
#             lower_freq = freq - mask_percentage * freq
#             upper_freq = freq + mask_percentage * freq

#             lower_idx = int(self.xp.searchsorted(self.f_arr, lower_freq - df/2))
#             upper_idx = int(self.xp.searchsorted(self.f_arr, upper_freq + df/2))
            
#             all_indices.update(range(lower_idx, upper_idx))
        
#         return self.xp.array(sorted(all_indices), dtype=self.xp.int32)

#     def _get_dips_indices(self,):

#         transfer_functions = self.compute_transfer_functions(self.f_arr)

#         tf = transfer_functions[0]

#         dips_indices = [
#             self._find_dips_with_percentage(tf[t_idx], mask_percentage=self.mask_percentage)
#             for t_idx in range(self.num_times)
#         ]

#         return dips_indices


#     def _compute_matrix_elements(self, 
#                                  freqs, 
#                                  Soms_d_in=15e-12, 
#                                  Sa_a_in=3e-15, 
#                                  Amp=0, 
#                                  alpha=0, 
#                                  sl1=0, 
#                                  kn=0, 
#                                  sl2=0, 
#                                  knots_position_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                                  knots_amplitude_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                                  ):
#         """Compute the 6 sensitivity matrix terms using the c++ backend."""
        
#         xp = self.xp
#         total_terms = self.basis_settings.total_terms
        
#         c00 = xp.empty(total_terms, dtype=xp.float64)
#         c11 = xp.empty(total_terms, dtype=xp.float64)
#         c22 = xp.empty(total_terms, dtype=xp.float64)
#         c01 = xp.empty(total_terms, dtype=xp.complex128)
#         c02 = xp.empty(total_terms, dtype=xp.complex128)
#         c12 = xp.empty(total_terms, dtype=xp.complex128)

#         if self.use_splines:
#             assert knots_position_all is not None and knots_amplitude_all is not None
#             splines_out = self.spline_interpolant(freqs, knots_position_all, knots_amplitude_all)
#             splines_in_isi_oms = splines_out[0]
#             spline_in_testmass = splines_out[1]
#         else:
#             splines_in_isi_oms = xp.zeros(len(freqs), dtype=xp.float64)
#             spline_in_testmass = xp.zeros(len(freqs), dtype=xp.float64)

#         self.pycpp_sensitivity_matrix.get_noise_covariance_wrap(
#             xp.asarray(freqs),
#             self.time_indices,
#             float(Soms_d_in),
#             float(Sa_a_in),
#             float(Amp),
#             float(alpha),
#             float(sl1),
#             float(kn),
#             float(sl2),
#             splines_in_isi_oms,
#             spline_in_testmass,
#             c00, c01, c02, c11, c12, c22,
#             len(freqs),
#             len(self.time_indices)
#         )

#         return c00, c11, c22, c01, c02, c12
    
#     def _fill_matrix(self, c00, c11, c22, c01, c02, c12):
#         """Fill the full 3x3 sensitivity matrix from its 6 unique elements."""
#         xp = self.xp    
#         shape = self.basis_settings.basis_shape

#         # Reshape views (no copy)
#         c00 = c00.reshape(shape)
#         c11 = c11.reshape(shape)
#         c22 = c22.reshape(shape)
#         c01 = c01.reshape(shape)
#         c02 = c02.reshape(shape)
#         c12 = c12.reshape(shape)

#         # Direct assignment is faster than stack (no intermediate copies)
#         matrix = xp.empty(self.channel_shape + shape, dtype=xp.complex128)
#         matrix[0, 0] = c00
#         matrix[1, 1] = c11
#         matrix[2, 2] = c22
#         matrix[0, 1] = c01
#         matrix[1, 0] = c01.conj()
#         matrix[0, 2] = c02
#         matrix[2, 0] = c02.conj()
#         matrix[1, 2] = c12
#         matrix[2, 1] = c12.conj()
        
#         return matrix
    
#     def _extract_matrix_elements(self, matrix_in, flatten=False):
#         """Extract the 6 unique sensitivity matrix elements from the full 3x3 matrix."""

#         c00 = matrix_in[0, 0].real
#         c11 = matrix_in[1, 1].real
#         c22 = matrix_in[2, 2].real
#         c01 = matrix_in[0, 1]
#         c02 = matrix_in[0, 2]
#         c12 = matrix_in[1, 2]

#         if flatten:
#            return c00.flatten(), c11.flatten(), c22.flatten(), c01.flatten(), c02.flatten(), c12.flatten()

#         return c00, c11, c22, c01, c02, c12

    
#     def compute_sensitivity_matrix(self, freqs, Soms_d_in=15e-12, Sa_a_in=3e-15, Amp=0, alpha=0, sl1=0, kn=0, sl2=0, knots_position_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                                    knots_amplitude_all: np.ndarray | cp.ndarray | jnp.ndarray = None,):
#         """Compute the full 3x3 sensitivity matrix using the c++ backend."""
#         c00, c11, c22, c01, c02, c12 = self._compute_matrix_elements(
#             freqs, Soms_d_in, Sa_a_in, Amp, alpha, sl1, kn, sl2, knots_position_all, knots_amplitude_all
#         )
#         matrix = self._fill_matrix(c00, c11, c22, c01, c02, c12)
#         return matrix

#     def set_sensitivity_matrix(self, 
#                                Soms_d_in: float = 15e-12, 
#                                Sa_a_in: float = 3e-15, 
#                                knots_position_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                                knots_amplitude_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                                Amp: float = 0., 
#                                alpha: float = 0., 
#                                sl1: float = 0., 
#                                kn: float = 0., 
#                                sl2: float = 0., 
#                                ):
#         """Internally store the sensitivity matrix computed at the basis frequencies."""

        
        
#         c00, c11, c22, c01, c02, c12 = self._compute_matrix_elements(
#             self.f_arr, Soms_d_in, Sa_a_in, Amp, alpha, sl1, kn, sl2, knots_position_all, knots_amplitude_all
#         )

#         sens_mat = self._fill_matrix(c00, c11, c22, c01, c02, c12)

#         self.sens_mat = self.smooth_sensitivity_matrix(sens_mat, sigma=5)

    
#     def _setup_det_and_inv(self):
#         """use the c++ backend to compute the log-determinant and inverse of the sensitivity matrix."""
#         c00, c11, c22, c01, c02, c12 = self._extract_matrix_elements(self.sens_mat, flatten=True)
#         self.invC, self.detC = self._inverse_det_wrapper(c00, c11, c22, c01, c02, c12)

#     def _inverse_det_wrapper(self, 
#                              c00: np.ndarray | cp.ndarray | jnp.ndarray, 
#                              c11: np.ndarray | cp.ndarray | jnp.ndarray, 
#                              c22: np.ndarray | cp.ndarray | jnp.ndarray, 
#                              c01: np.ndarray | cp.ndarray | jnp.ndarray, 
#                              c02: np.ndarray | cp.ndarray | jnp.ndarray, 
#                              c12: np.ndarray | cp.ndarray | jnp.ndarray
#                              ) -> tuple:
        
#         """Wrapper to call c++ backend for inverse log-determinant computation."""
        
#         xp = self.xp
#         total_terms = self.basis_settings.total_terms

#         i00 = xp.empty(total_terms, dtype=xp.float64)
#         i11 = xp.empty(total_terms, dtype=xp.float64)
#         i22 = xp.empty(total_terms, dtype=xp.float64)
#         i01 = xp.empty(total_terms, dtype=xp.complex128)
#         i02 = xp.empty(total_terms, dtype=xp.complex128)
#         i12 = xp.empty(total_terms, dtype=xp.complex128)

#         det = xp.empty(total_terms, dtype=xp.float64)

#         self.pycpp_sensitivity_matrix.get_inverse_det_wrap(
#             c00, c01, c02, c11, c12, c22,
#             i00, i01, i02, i11, i12, i22,
#             det,
#             total_terms
#         )
        
#         inverse_matrix = self._fill_matrix(i00, i11, i22, i01, i02, i12)

#         return inverse_matrix, det.reshape(self.basis_settings.basis_shape)

#     def compute_inverse_det(self, 
#                             matrix_in: np.ndarray | cp.ndarray | jnp.ndarray
#                             ) -> tuple:
#         """
#         Invert the 3x3 sensitivity matrix and compute its log-determinant with the c++ backend.

#         Args:
#             matrix_in: Input sensitivity matrix. Shape (3, 3, ...)
        
#         Returns:
#             inverse_matrix: Inverted sensitivity matrix. Shape (3, 3, ...)
#             det: Determinant of the sensitivity matrix. Shape (...)
#         """
#         c00, c11, c22, c01, c02, c12 = self._extract_matrix_elements(matrix_in, flatten=True)
#         inverse_matrix, det = self._inverse_det_wrapper(c00, c11, c22, c01, c02, c12)
#         return inverse_matrix, det

#     def compute_transfer_functions(self, 
#                                    freqs: np.ndarray | cp.ndarray | jnp.ndarray
#                                    ) -> tuple:
        
#         """Compute transfer functions using the c++ backend."""

#         xp = self.xp
#         num_freqs = len(freqs)

#         total_shape = self.num_times * num_freqs

#         oms_xx = xp.empty(shape=(total_shape,), dtype=xp.float64)
#         oms_yy = xp.empty(shape=(total_shape,), dtype=xp.float64)
#         oms_zz = xp.empty(shape=(total_shape,), dtype=xp.float64)
#         oms_xy = xp.empty(shape=(total_shape,), dtype=xp.complex128)
#         oms_xz = xp.empty(shape=(total_shape,), dtype=xp.complex128)
#         oms_yz = xp.empty(shape=(total_shape,), dtype=xp.complex128)

#         tm_xx = xp.empty(shape=(total_shape,), dtype=xp.float64)
#         tm_yy = xp.empty(shape=(total_shape,), dtype=xp.float64)
#         tm_zz = xp.empty(shape=(total_shape,), dtype=xp.float64)
#         tm_xy = xp.empty(shape=(total_shape,), dtype=xp.complex128)
#         tm_xz = xp.empty(shape=(total_shape,), dtype=xp.complex128)
#         tm_yz = xp.empty(shape=(total_shape,), dtype=xp.complex128)

#         self.pycpp_sensitivity_matrix.get_noise_tfs_wrap(
#             xp.asarray(freqs),
#             oms_xx, oms_xy, oms_xz, oms_yy, oms_yz, oms_zz,
#             tm_xx, tm_xy, tm_xz, tm_yy, tm_yz, tm_zz,
#             num_freqs, self.num_times,
#             self._time_indices
#         )

#         return (oms_xx.reshape(self.num_times, num_freqs), 
#                 oms_xy.reshape(self.num_times, num_freqs),
#                 oms_xz.reshape(self.num_times, num_freqs),
#                 oms_yy.reshape(self.num_times, num_freqs), 
#                 oms_yz.reshape(self.num_times, num_freqs),
#                 oms_zz.reshape(self.num_times, num_freqs), 
#                 tm_xx.reshape(self.num_times, num_freqs), 
#                 tm_xy.reshape(self.num_times, num_freqs),
#                 tm_xz.reshape(self.num_times, num_freqs),
#                 tm_yy.reshape(self.num_times, num_freqs), 
#                 tm_yz.reshape(self.num_times, num_freqs),
#                 tm_zz.reshape(self.num_times, num_freqs)
#                 )

#     def compute_log_like(self,
#                          data_in_all: np.ndarray | cp.ndarray | jnp.ndarray, 
#                          data_index_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          Soms_in_all: np.ndarray | cp.ndarray | jnp.ndarray, 
#                          Sa_in_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          Amp_in_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          alpha_in_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          sl1_in_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          kn_in_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          sl2_in_all: np.ndarray | cp.ndarray | jnp.ndarray,
#                          knots_position_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                          knots_amplitude_all: np.ndarray | cp.ndarray | jnp.ndarray = None,
#                          ) -> np.ndarray | cp.ndarray | jnp.ndarray:
#         """
#         Compute log-likelihood using the c++ backend.

#         Args:
#             data_in_all: Input data array. Shape (num_psds, num_freqs * num_times)
#             data_index_all: Data indices array to keep track of which data corresponds to which PSD. Shape (num_psds)
#             Soms_in_all: Displacement noise levels for each walker. Shape (num_psds)
#             Sa_in_all: Acceleration noise levels for each walker. Shape (num_psds)
#             Amp_in_all: Galactic foreground amplitude for each walker. Shape (num_psds)
#             alpha_in_all: Galactic foreground alpha for each walker. Shape (num_psds)
#             sl1_in_all: First galactic foreground slope parameter for each walker. Shape (num_psds)
#             kn_in_all: Galactic foreground knee frequency parameter for each walker. Shape (num_psds)
#             sl2_in_all: Second galactic foreground slope parameter for each walker. Shape (num_psds)
#             knots_position_all: Positions of spline knots for noise modeling. Shape (2 * num_psds, num_knots)
#             knots_amplitude_all: Amplitudes of spline knots for noise modeling. Shape (2 * num_psds, num_knots)

#         Returns:
#             log_like_out: Computed log-likelihoods for each PSD. Shape (num_psds,)
#         """

#         xp = self.xp

#         num_psds = len(Soms_in_all)

#         log_like_out = xp.zeros(shape=(num_psds,), dtype=xp.float64)

#         if self.use_splines:
#             splines_weights = self.spline_interpolant(xp.log10(self.f_arr), knots_position_all, knots_amplitude_all)

#             splines_weights_isi_oms = splines_weights[:num_psds].flatten()
#             splines_weights_testmass = splines_weights[num_psds:].flatten()

#         else:
#             splines_weights_isi_oms = xp.zeros(shape=(num_psds * self.num_freqs))
#             splines_weights_testmass = xp.zeros(shape=(num_psds * self.num_freqs))
    
#         self.pycpp_sensitivity_matrix.psd_likelihood_wrap(
#             log_like_out,
#             self.f_arr,
#             xp.asarray(data_in_all.flatten()),
#             xp.asarray(data_index_all.flatten()),
#             xp.asarray(self.time_indices),
#             xp.asarray(Soms_in_all),
#             xp.asarray(Sa_in_all),
#             xp.asarray(Amp_in_all),
#             xp.asarray(alpha_in_all),
#             xp.asarray(sl1_in_all),
#             xp.asarray(kn_in_all),
#             xp.asarray(sl2_in_all),
#             xp.asarray(splines_weights_isi_oms),
#             xp.asarray(splines_weights_testmass), 
#             self.basis_settings.differential_component,
#             self.num_freqs,
#             self.num_times,
#             self.dips_mask,
#             num_psds
#         )

#         return log_like_out
    
#     def smooth_sensitivity_matrix(self,
#                                   matrix_in: np.ndarray | cp.ndarray | jnp.ndarray,
#                                   sigma: float = 5.0,
#                                   ) -> np.ndarray | cp.ndarray | jnp.ndarray:
        
#         """
#         Perform log-frequency smoothing of the sensitivity matrix to get rid of the very sharp dips.

#         Args:
#             matrix_in: Input sensitivity matrix. Shape (3, 3, num_times, num_freqs)
#             sigma: Width of the Gaussian smoothing kernel in frequency bins.
#         """
#         filter_func = np_gaussian_filter1d if self.xp == np else cp_gaussian_filter1d if self.xp == cp else jax_gaussian_filter1d
        
#         smoothed_matrix = matrix_in.copy()
#         mask = self.dips_mask.reshape(self.num_times, self.num_freqs)
#         _smoothed = filter_func(matrix_in, sigma=sigma, axis=-1)

#         smoothed_matrix[..., mask] = _smoothed[..., mask]

#         return smoothed_matrix
        


#     def __call__(self, 
#                 name: str,
#                 psd_params: np.ndarray, 
#                 galfor_params: np.ndarray=None
#                 ) -> XYZSensitivityBackend:
#         """
#         Update the internal sensitivity matrix with new noise parameters and return to be used in a AnalysisContainer.

#         Args:
#             psd_params: Array of PSD parameters in order [Soms_d, Sa_a, (optional spline params...)]
#             galfor_params: Array of galactic foreground parameters in order [Amp, alpha, sl1, kn, sl2].
        
#         Returns:
#             self: a configured copy of the sensitivity matrix backend.
#         """
#         self.name = name

#         Soms_d = psd_params[0]
#         Sa_a = psd_params[1]

#         if self.use_splines:
#             # todo add a container for the noise
#             spline_params = psd_params[2:]
#             spline_knots_position = spline_params[::2]
#             spline_knots_amplitude = spline_params[1::2]

#         else:
#             spline_knots_position = None
#             spline_knots_amplitude = None
        
#         if galfor_params is None:
#             galfor_params = np.zeros(5)
    
#         self.set_sensitivity_matrix(
#             Soms_d,
#             Sa_a,
#             spline_knots_position,
#             spline_knots_amplitude,
#             *galfor_params
#         )

#         return deepcopy(self) #todo self, or deepcopy(self)?