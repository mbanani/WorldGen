# This file makes src/worldgen a Python package 

from .worldgen import WorldGen

# NOTE: This creates a default WorldGen instance upon import worldgen.
# This instance uses the default parameters of WorldGen.__init__ (e.g., device='cuda').
# If you need more control (e.g., different device), import the class directly:
# from worldgen import WorldGen
# wg = WorldGen(device='cpu')
# then use wg.generate_world(...)
_default_instance = WorldGen()
generate_world = _default_instance.generate_world

# Expose both the class and the direct function for convenience
__all__ = ['WorldGen', 'generate_world'] 