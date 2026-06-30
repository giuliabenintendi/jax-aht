"""Overcooked v2 environment (copied from JaxMARL 0.1.0; see PROVENANCE.md).

The raw v2 env files (overcooked.py, common.py, layouts.py, settings.py, utils.py,
grid_rendering_v2.py) are copied from upstream. Repo-specific wrappers (image
observation, BaseEnv adapter) are added in this package as they are built.
"""
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.layouts import Layout, overcooked_v2_layouts

__all__ = ["OvercookedV2", "Layout", "overcooked_v2_layouts"]
