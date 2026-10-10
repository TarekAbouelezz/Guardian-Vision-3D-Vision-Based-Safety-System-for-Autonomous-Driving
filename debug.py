import carla
import random

# Connect
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print("Connected to world:", world.get_map().name)

# Test spawning
bp_lib = world.get_blueprint_library()
tm = client.get_trafficmanager(8000)

# Sync mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
print("Sync mode ON")

# Spawn ego
bp = bp_lib.filter('vehicle.tesla.model3')[0]
spawn = random.choice(world.get_map().get_spawn_points())
ego = world.spawn_actor(bp, spawn)
print("Ego spawned:", ego.id)

# Tick once
world.tick()
print("Tick OK")

# Cleanup
ego.destroy()
settings.synchronous_mode = False
world.apply_settings(settings)
print("All good — ready to collect data")
