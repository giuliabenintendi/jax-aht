import sys
sys.path.insert(0, ".")
from envs.overcooked.overcooked_image_wrapper import OvercookedImageWrapper
from envs.overcooked.augmented_layouts import augmented_layouts
env = OvercookedImageWrapper(layout=augmented_layouts["cramped_room"], max_steps=400)
print("agent_view_size:", env.agent_view_size)
print("grid_height:", env.grid_height, "grid_width:", env.grid_width)
print("image obs padding:", 4)
print("eval video padding:", env.agent_view_size - 2)
print("interior_wall_mask shape:", env.interior_wall_mask.shape)
print("interior_wall_mask sum:", int(env.interior_wall_mask.sum()))
