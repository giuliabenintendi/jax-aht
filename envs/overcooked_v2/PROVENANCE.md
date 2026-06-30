# Provenance

These files are copied from JaxMARL's Overcooked v2 environment.

- Source: https://github.com/FLAIROx/JaxMARL
- Tag: `v0.1.0`
- Commit: `66f41e5a36131d86bf5791d6bbe501275ed2cd30`
- License: Apache-2.0 (see `LICENSE`)

## Why this is copied in instead of imported

Overcooked v2 first ships in `jaxmarl==0.1.0`. This repo pins `jaxmarl==0.0.7`
(Overcooked v1 only). Copying the v2 package in keeps v2 usable without bumping the
global `jaxmarl` dependency, so the existing card-game / LBF / overcooked-v1 / hanabi
environments stay byte-stable.

## Files

Copied from `jaxmarl/environments/overcooked_v2/`: `__init__.py`, `overcooked.py`,
`common.py`, `layouts.py`, `settings.py`, `utils.py`. Copied from `jaxmarl/viz/`:
`grid_rendering_v2.py`.

## Modifications

The only edits are import-path rewrites so the package resolves at its new location:

- `from jaxmarl.environments.overcooked_v2.X import ...` -> `from envs.overcooked_v2.X import ...`
- relative `from .X import ...` -> `from envs.overcooked_v2.X import ...`

`from jaxmarl.environments import MultiAgentEnv` and `from jaxmarl.environments import spaces`
are left untouched: they resolve against the installed `jaxmarl==0.0.7`, which exposes both.
No environment logic was changed.
