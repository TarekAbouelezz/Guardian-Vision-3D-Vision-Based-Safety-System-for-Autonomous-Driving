"""
CARLA scenario-based dataset collector (KITTI format: image + LiDAR + labels + calib).

Each "run" = one scripted safety event (a pedestrian crossing in front of the ego car),
repeated for several weather conditions. Ground truth is projected into the camera,
filtered by visibility, and written in KITTI format for SMOKE-style training.

Start CARLA first (e.g. .\\CarlaUE4.exe -quality-level=Low), then run this script.
Needs CARLA >= 0.9.13 for the instance segmentation camera (set USE_INSTANCE_SEG=False otherwise).
"""
import csv
import math
import os
import queue
import random
import traceback

import carla
import numpy as np

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════
HOST, PORT, TM_PORT = 'localhost', 2000, 8000
MAP_NAME = 'Town03'
SEED = 42
BASE = 'dataset'

IMG_W, IMG_H, FOV = 1242, 375, 90.0
FIXED_DT = 0.05              # 20 FPS simulation
SAVE_EVERY = 2               # save every 2nd tick -> 10 Hz
MAX_RANGE = 50.0             # metres, matches LiDAR range
MIN_BOX_H = 10               # px, skip tiny 2D boxes
MIN_VISIBLE_PIXELS = 25      # skip objects (almost) fully hidden in the image
USE_INSTANCE_SEG = True      # used to drop labels of occluded objects

RUNS_PER_COMBO = 3           # runs per (weather x scenario)
MAX_RUN_TICKS = 600          # 30 s safety timeout per run
RECORD_START_DIST = 60.0     # start saving when ego is this close to the crossing point
CROSS_DIST_AHEAD = (80.0, 100.0)   # crossing point distance from ego spawn
EGO_SPEEDS_KMH = (20, 30, 40)
N_BG_VEHICLES = 20
N_BG_WALKERS = 30
VAL_EVERY = 5                # every 5th run goes to the validation split (split by run, not by frame)
SIDEWALK_MARGIN = 1.5        # metres beyond the driving-lane edge where pedestrians start

WEATHERS = {
    'clear_day': carla.WeatherParameters.ClearNoon,
    'rain':      carla.WeatherParameters.HardRainNoon,
    'fog':       carla.WeatherParameters(cloudiness=80, precipitation=0, fog_density=60,
                                         fog_distance=10, sun_altitude_angle=45),
    'night':     carla.WeatherParameters(cloudiness=10, precipitation=0, sun_altitude_angle=-30),
}
NIGHT = {'night'}

# ttc = time-to-arrival of the ego car (s) at which the pedestrian starts walking.
# Small ttc = more dangerous. ped_speed in m/s.
SCENARIOS = [
    dict(name='walk_normal',  n_peds=1, ped_speed=(1.2, 1.6), ttc=(3.5, 5.0), child=False, occluder=False),
    dict(name='jaywalk_fast', n_peds=1, ped_speed=(1.8, 2.6), ttc=(2.0, 3.0), child=False, occluder=False),
    dict(name='occluded',     n_peds=1, ped_speed=(1.2, 2.0), ttc=(2.2, 3.2), child=False, occluder=True),
    dict(name='child_run',    n_peds=1, ped_speed=(2.5, 3.5), ttc=(2.2, 3.2), child=True,  occluder=True),
    dict(name='group',        n_peds=3, ped_speed=(1.2, 1.6), ttc=(3.5, 5.0), child=False, occluder=False),
]

# ═════════════════════════════════════════════════════════════
# SENSOR GEOMETRY + CALIBRATION (computed once)
# ═════════════════════════════════════════════════════════════
CAM_TF = carla.Transform(carla.Location(x=1.5, z=1.65))
LIDAR_TF = carla.Transform(carla.Location(x=1.0, z=1.8))

F = IMG_W / (2.0 * math.tan(math.radians(FOV / 2.0)))
CX, CY = IMG_W / 2.0, IMG_H / 2.0
P2 = np.array([[F, 0, CX, 0], [0, F, CY, 0], [0, 0, 1, 0]], dtype=float)

# CARLA sensor frame (x fwd, y right, z up) -> KITTI camera frame (x right, y down, z fwd)
M_CAM2KITTI = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=float)
# saved LiDAR uses y flipped (x fwd, y left, z up) = KITTI velodyne convention
M_VELO2CARLA = np.diag([1.0, -1.0, 1.0, 1.0])
_rel = np.linalg.inv(np.array(CAM_TF.get_matrix())) @ np.array(LIDAR_TF.get_matrix())
TR_VELO_TO_CAM = (M_CAM2KITTI @ _rel @ M_VELO2CARLA)[:3, :]


def calib_text():
    p = ' '.join(f'{v:.6e}' for v in P2.flatten())
    tr = ' '.join(f'{v:.9e}' for v in TR_VELO_TO_CAM.flatten())
    return '\n'.join([
        f'P0: {p}', f'P1: {p}', f'P2: {p}', f'P3: {p}',
        'R0_rect: 1 0 0 0 1 0 0 0 1',
        f'Tr_velo_to_cam: {tr}',
        'Tr_imu_to_velo: 1 0 0 0 0 1 0 0 0 0 1 0',
    ]) + '\n'


CALIB_TEXT = calib_text()

# ═════════════════════════════════════════════════════════════
# SETUP
# ═════════════════════════════════════════════════════════════
random.seed(SEED)
for sub in ['images', 'velodyne', 'labels', 'calib', 'ImageSets']:
    os.makedirs(f'{BASE}/{sub}', exist_ok=True)

client = carla.Client(HOST, PORT)
client.set_timeout(30.0)
world = client.get_world()
if MAP_NAME not in world.get_map().name:
    print('Loading map', MAP_NAME)
    world = client.load_world(MAP_NAME)
bp_lib = world.get_blueprint_library()
tm = client.get_trafficmanager(TM_PORT)
print('Connected to:', world.get_map().name)

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = FIXED_DT
world.apply_settings(settings)
tm.set_synchronous_mode(True)
tm.set_random_device_seed(SEED)
world.set_pedestrians_seed(SEED)
world.set_pedestrians_cross_factor(0.3)   # background walkers sometimes cross roads
print('Sync mode ON')

frame_id = 0
split_frames = {'train': [], 'val': []}


# ═════════════════════════════════════════════════════════════
# SENSORS
# ═════════════════════════════════════════════════════════════
def cam_bp(kind):
    bp = bp_lib.find(kind)
    bp.set_attribute('image_size_x', str(IMG_W))
    bp.set_attribute('image_size_y', str(IMG_H))
    bp.set_attribute('fov', str(FOV))
    return bp


def lidar_bp():
    bp = bp_lib.find('sensor.lidar.ray_cast')
    bp.set_attribute('channels', '64')
    bp.set_attribute('points_per_second', '2000000')    # lower this if the PC struggles
    bp.set_attribute('rotation_frequency', str(1.0 / FIXED_DT))   # one full sweep per tick
    bp.set_attribute('range', str(MAX_RANGE))
    bp.set_attribute('upper_fov', '2.0')
    bp.set_attribute('lower_fov', '-24.8')
    return bp


def listen_queue(sensor):
    q = queue.Queue()
    sensor.listen(q.put)
    return q


def get_frame(q, frame, timeout=10.0):
    """Return the sensor measurement that belongs to simulation frame `frame`."""
    while True:
        d = q.get(timeout=timeout)
        if d.frame == frame:
            return d
        if d.frame > frame:
            raise RuntimeError(f'Sensor frame {d.frame} is ahead of world frame {frame}')


def drain(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ═════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═════════════════════════════════════════════════════════════
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def to_cam(world_pts, cam_inv):
    """Nx3 world points -> Nx3 KITTI camera coordinates (x right, y down, z forward)."""
    pts = np.c_[world_pts, np.ones(len(world_pts))]
    c = (cam_inv @ pts.T).T[:, :3]
    return np.c_[c[:, 1], -c[:, 2], c[:, 0]]


def decode_ids(seg_img):
    a = np.frombuffer(seg_img.raw_data, dtype=np.uint8).reshape(seg_img.height, seg_img.width, 4)
    return a[:, :, 1].astype(np.int32) + a[:, :, 0].astype(np.int32) * 256   # G + 256*B = actor id


def make_label(actor, cam_inv, ids):
    """Return (KITTI label line, actor id) or None if the object is not usable."""
    tf = actor.get_transform()
    bb = actor.bounding_box
    verts = np.array([[v.x, v.y, v.z] for v in bb.get_world_vertices(tf)])
    vc = to_cam(verts, cam_inv)
    if (vc[:, 2] > 0.1).sum() < 8:          # not fully in front of the camera
        return None

    ctr = tf.transform(bb.location)
    fwd = tf.get_forward_vector()
    pc = to_cam(np.array([
        [ctr.x, ctr.y, ctr.z - bb.extent.z],                  # bottom centre (KITTI location)
        [ctr.x, ctr.y, ctr.z],
        [ctr.x + fwd.x, ctr.y + fwd.y, ctr.z + fwd.z],
    ]), cam_inv)
    x, y, z = pc[0]
    if not (1.0 < z < MAX_RANGE):
        return None
    d = pc[2] - pc[1]
    ry = math.atan2(-d[2], d[0])
    alpha = wrap(ry - math.atan2(x, z))

    u = F * vc[:, 0] / vc[:, 2] + CX
    v = F * vc[:, 1] / vc[:, 2] + CY
    x1, x2, y1, y2 = u.min(), u.max(), v.min(), v.max()
    raw = (x2 - x1) * (y2 - y1)
    cx1, cx2 = max(x1, 0.0), min(x2, IMG_W - 1.0)
    cy1, cy2 = max(y1, 0.0), min(y2, IMG_H - 1.0)
    if cx2 <= cx1 or cy2 <= cy1 or raw <= 0:
        return None
    trunc = 1.0 - ((cx2 - cx1) * (cy2 - cy1)) / raw
    if trunc > 0.8 or (cy2 - cy1) < MIN_BOX_H:
        return None

    is_walker = 'walker' in actor.type_id
    occ = 0
    if ids is not None:
        region = ids[int(cy1):int(cy2) + 1, int(cx1):int(cx2) + 1]
        vis = int((region == actor.id).sum())
        if vis < MIN_VISIBLE_PIXELS:
            return None                     # hidden behind something: no label
        fill = 0.45 if is_walker else 0.6   # typical silhouette/box ratio (approximate)
        r = vis / max(region.size, 1) / fill
        occ = 0 if r > 0.75 else (1 if r > 0.4 else 2)

    if is_walker:
        cls = 'Pedestrian'
    elif actor.attributes.get('number_of_wheels') == '2':
        cls = 'Cyclist'
    else:
        cls = 'Car'

    h, w, l = 2 * bb.extent.z, 2 * bb.extent.y, 2 * bb.extent.x
    line = (f'{cls} {max(trunc, 0.0):.2f} {occ} {alpha:.2f} '
            f'{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} '
            f'{h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}')
    return line, actor.id


# ═════════════════════════════════════════════════════════════
# ROAD / SCENARIO HELPERS
# ═════════════════════════════════════════════════════════════
def lane_edges(wp):
    """Lateral distance (m) from the lane centre to the right and left edge of the driving lanes."""
    def walk(getter):
        off, cur = wp.lane_width / 2.0, wp
        for _ in range(8):
            nxt = getter(cur)
            if nxt is None or nxt.lane_type != carla.LaneType.Driving:
                break
            off += nxt.lane_width
            cur = nxt
        return off
    return walk(lambda w: w.get_right_lane()), walk(lambda w: w.get_left_lane())


def find_straight_spawn(min_len):
    """Spawn point followed by `min_len` m of road without a junction (so the ego reaches the crossing)."""
    cmap = world.get_map()
    sps = cmap.get_spawn_points()
    random.shuffle(sps)
    for sp in sps:
        wp = cmap.get_waypoint(sp.location)
        if wp.is_junction:
            continue
        cur, dist, ok = wp, 0.0, True
        while dist < min_len:
            nxt = cur.next(5.0)
            if not nxt or nxt[0].is_junction:
                ok = False
                break
            cur, dist = nxt[0], dist + 5.0
        if ok:
            return sp
    return None


def pick_walker_bp(child):
    bps = list(bp_lib.filter('walker.pedestrian.*'))
    want = 'child' if child else 'adult'
    sel = [b for b in bps if b.has_attribute('age') and b.get_attribute('age').as_str() == want]
    return random.choice(sel or bps)


def spawn_scenario_peds(scn, wp_c, r_edge):
    tf = wp_c.transform
    rv, fv, base = tf.get_right_vector(), tf.get_forward_vector(), tf.location
    cd = (-rv.x, -rv.y)                                   # crossing direction: towards the left
    yaw = math.degrees(math.atan2(cd[1], cd[0]))
    peds = []
    for i in range(scn['n_peds']):
        lat = r_edge + SIDEWALK_MARGIN + random.uniform(-0.2, 0.4) + 0.6 * i
        lon = random.uniform(-1.0, 1.0) + 1.0 * i
        for dz in (0.6, 2.0):
            loc = carla.Location(base.x + rv.x * lat + fv.x * lon,
                                 base.y + rv.y * lat + fv.y * lon, base.z + dz)
            a = world.try_spawn_actor(pick_walker_bp(scn['child'] and i == 0),
                                      carla.Transform(loc, carla.Rotation(yaw=yaw)))
            if a is not None:
                peds.append(dict(actor=a, speed=random.uniform(*scn['ped_speed'])))
                break
    return peds, cd


def spawn_occluder(wp_c, r_edge):
    """Parked vehicle on the roadside just before the crossing point, hiding the pedestrian."""
    prev = wp_c.previous(5.0)
    if not prev:
        return None
    tf = prev[0].transform
    rv = tf.get_right_vector()
    lat = r_edge + 0.9
    loc = carla.Location(tf.location.x + rv.x * lat, tf.location.y + rv.y * lat, tf.location.z + 0.3)
    bps = list(bp_lib.filter('vehicle.volkswagen.t2*')) or \
        [b for b in bp_lib.filter('vehicle.*') if b.get_attribute('number_of_wheels').as_int() == 4]
    occ = world.try_spawn_actor(random.choice(bps), carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw)))
    if occ is not None:
        occ.apply_control(carla.VehicleControl(hand_brake=True))
    return occ


def spawn_background(ego_sp):
    vehicles, walkers, controllers = [], [], []
    v_bps = [b for b in bp_lib.filter('vehicle.*') if b.get_attribute('number_of_wheels').as_int() == 4]
    spawns = [s for s in world.get_map().get_spawn_points() if s.location.distance(ego_sp.location) > 40.0]
    random.shuffle(spawns)
    for s in spawns[:N_BG_VEHICLES]:
        v = world.try_spawn_actor(random.choice(v_bps), s)
        if v is not None:
            v.set_autopilot(True, TM_PORT)
            vehicles.append(v)

    w_bps = list(bp_lib.filter('walker.pedestrian.*'))
    batch, tries = [], 0
    while len(batch) < N_BG_WALKERS and tries < N_BG_WALKERS * 20:
        tries += 1
        loc = world.get_random_location_from_navigation()
        if loc is None or loc.distance(ego_sp.location) > 100.0:
            continue
        batch.append(carla.command.SpawnActor(random.choice(w_bps), carla.Transform(loc)))
    for res in client.apply_batch_sync(batch, True):
        if not res.error:
            walkers.append(world.get_actor(res.actor_id))

    ctrl_bp = bp_lib.find('controller.ai.walker')
    for res in client.apply_batch_sync(
            [carla.command.SpawnActor(ctrl_bp, carla.Transform(), w) for w in walkers], True):
        if not res.error:
            controllers.append(world.get_actor(res.actor_id))
    world.tick()
    for c in controllers:
        c.start()
        dest = world.get_random_location_from_navigation()
        if dest:
            c.go_to_location(dest)
        c.set_max_speed(random.uniform(1.0, 2.0))
    return vehicles, walkers, controllers


def cleanup(sensors, controllers, actors):
    for s in sensors + controllers:
        try:
            s.stop()
        except RuntimeError:
            pass
    ids = [a.id for a in sensors + controllers + actors if a is not None]
    client.apply_batch([carla.command.DestroyActor(i) for i in ids])
    world.tick()


# ═════════════════════════════════════════════════════════════
# ONE RUN = ONE SCRIPTED SCENARIO
# ═════════════════════════════════════════════════════════════
def run_episode(run_id, weather_name, scn, writer):
    global frame_id
    sp = find_straight_spawn(CROSS_DIST_AHEAD[1] + 30.0)
    if sp is None:
        print('  no straight road found, skipping')
        return 0

    world.set_weather(WEATHERS[weather_name])
    sensors, controllers, actors = [], [], []
    split = 'val' if run_id % VAL_EVERY == VAL_EVERY - 1 else 'train'
    saved = 0
    try:
        ego = world.try_spawn_actor(bp_lib.filter('vehicle.tesla.model3')[0], sp)
        if ego is None:
            print('  ego spawn failed, skipping')
            return 0
        actors.append(ego)
        ego.set_autopilot(True, TM_PORT)
        tm.ignore_walkers_percentage(ego, 0)      # ego reacts to pedestrians (set 100 to ignore them)
        tm.auto_lane_change(ego, False)
        tm.distance_to_leading_vehicle(ego, 5.0)
        if weather_name in NIGHT:
            try:
                tm.update_vehicle_lights(ego, True)
            except AttributeError:
                ego.set_light_state(carla.VehicleLightState(
                    carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))

        # sensors
        cam = world.spawn_actor(cam_bp('sensor.camera.rgb'), CAM_TF, attach_to=ego)
        lid = world.spawn_actor(lidar_bp(), LIDAR_TF, attach_to=ego)
        col = world.spawn_actor(bp_lib.find('sensor.other.collision'), carla.Transform(), attach_to=ego)
        sensors += [cam, lid, col]
        cam_q, lid_q = listen_queue(cam), listen_queue(lid)
        seg_q = None
        if USE_INSTANCE_SEG:
            seg = world.spawn_actor(cam_bp('sensor.camera.instance_segmentation'), CAM_TF, attach_to=ego)
            sensors.append(seg)
            seg_q = listen_queue(seg)
        collisions = []
        col.listen(lambda e: collisions.append(e.frame))

        # scenario geometry
        world.tick()
        wp0 = world.get_map().get_waypoint(ego.get_location())
        wp_c = wp0.next(random.uniform(*CROSS_DIST_AHEAD))[0]
        cross_loc = wp_c.transform.location
        fv_c = wp_c.transform.get_forward_vector()
        rv_c = wp_c.transform.get_right_vector()
        r_edge, l_edge = lane_edges(wp_c)

        peds, cd = spawn_scenario_peds(scn, wp_c, r_edge)
        if not peds:
            print('  pedestrian spawn failed, skipping')
            return 0
        actors += [p['actor'] for p in peds]
        if scn['occluder']:
            actors.append(spawn_occluder(wp_c, r_edge))
        bg_v, bg_w, bg_c = spawn_background(sp)
        actors += bg_v + bg_w
        controllers += bg_c
        if weather_name in NIGHT:
            for v in bg_v:
                try:
                    tm.update_vehicle_lights(v, True)
                except AttributeError:
                    pass

        # ego target speed
        limit = ego.get_speed_limit() or 50.0
        target = random.choice(EGO_SPEEDS_KMH)
        tm.vehicle_percentage_speed_difference(ego, (1.0 - target / limit) * 100.0)
        ttc_trigger = random.uniform(*scn['ttc'])
        for _ in range(20):
            world.tick()
        drain(cam_q), drain(lid_q)
        if seg_q:
            drain(seg_q)

        triggered, recording = False, False
        done_tick = hit_tick = None
        cam_inv = None

        for tick in range(MAX_RUN_TICKS):
            frame = world.tick()
            ego_loc, vel = ego.get_location(), ego.get_velocity()
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            dist_cross = ego_loc.distance(cross_loc)

            if not triggered and dist_cross / max(speed, 2.0) < ttc_trigger:
                triggered = True
            if triggered:
                all_done = True
                for p in peds:
                    loc = p['actor'].get_location()
                    lat = (loc.x - cross_loc.x) * rv_c.x + (loc.y - cross_loc.y) * rv_c.y
                    if lat > -(l_edge + 1.0):
                        p['actor'].apply_control(carla.WalkerControl(
                            direction=carla.Vector3D(cd[0], cd[1], 0.0), speed=p['speed']))
                        all_done = False
                    else:
                        p['actor'].apply_control(carla.WalkerControl(speed=0.0))
                if all_done and done_tick is None:
                    done_tick = tick
            if collisions and hit_tick is None:
                hit_tick = tick
            if dist_cross < RECORD_START_DIST:
                recording = True

            passed = ((ego_loc.x - cross_loc.x) * fv_c.x + (ego_loc.y - cross_loc.y) * fv_c.y) > 10.0
            if recording and tick % SAVE_EVERY == 0:
                img = get_frame(cam_q, frame)
                pts = get_frame(lid_q, frame)
                ids = decode_ids(get_frame(seg_q, frame)) if seg_q else None
                cam_inv = np.array(img.transform.get_inverse_matrix())

                lines, labeled = [], set()
                cands = list(world.get_actors().filter('vehicle.*')) + \
                    list(world.get_actors().filter('walker.pedestrian.*'))
                for a in cands:
                    if a.id == ego.id:
                        continue
                    res = make_label(a, cam_inv, ids)
                    if res:
                        lines.append(res[0])
                        labeled.add(res[1])

                fid = frame_id
                img.save_to_disk(f'{BASE}/images/{fid:06d}.png')
                arr = np.frombuffer(pts.raw_data, dtype=np.float32).reshape(-1, 4).copy()
                arr[:, 1] *= -1.0                                     # CARLA y-right -> KITTI y-left
                arr.tofile(f'{BASE}/velodyne/{fid:06d}.bin')
                with open(f'{BASE}/labels/{fid:06d}.txt', 'w') as f:
                    f.write('\n'.join(lines))
                with open(f'{BASE}/calib/{fid:06d}.txt', 'w') as f:
                    f.write(CALIB_TEXT)

                dmin = min(ego_loc.distance(p['actor'].get_location()) for p in peds)
                writer.writerow([fid, run_id, split, world.get_map().name.split('/')[-1], weather_name,
                                 scn['name'], f'{speed * 3.6:.1f}', f'{dmin:.1f}',
                                 int(any(p['actor'].id in labeled for p in peds)),
                                 len(lines), int(bool(collisions))])
                split_frames[split].append(fid)
                frame_id += 1
                saved += 1
            else:
                drain(cam_q), drain(lid_q)
                if seg_q:
                    drain(seg_q)

            if passed or (done_tick is not None and tick - done_tick > 60) or \
                    (hit_tick is not None and tick - hit_tick > 20):
                break
    finally:
        cleanup(sensors, controllers, actors)
    return saved


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
try:
    with open(f'{BASE}/metadata.csv', 'w', newline='') as mf:
        writer = csv.writer(mf)
        writer.writerow(['frame_id', 'run_id', 'split', 'map', 'weather', 'scenario', 'ego_speed_kmh',
                         'dist_to_nearest_scenario_ped_m', 'scenario_ped_labeled', 'n_labels', 'collision'])
        run_id = 0
        for weather_name in WEATHERS:
            for scn in SCENARIOS:
                for rep in range(RUNS_PER_COMBO):
                    print(f'>>> run {run_id}: {weather_name} / {scn["name"]} / rep {rep}')
                    n = run_episode(run_id, weather_name, scn, writer)
                    mf.flush()
                    print(f'    saved {n} frames (total {frame_id})')
                    run_id += 1

    for split, ids_ in split_frames.items():
        with open(f'{BASE}/ImageSets/{split}.txt', 'w') as f:
            f.write('\n'.join(f'{i:06d}' for i in ids_))
    print(f'\nDone: {frame_id} frames in {BASE}/')

except Exception:
    traceback.print_exc()

finally:
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)
    print('Simulator restored to normal mode.')