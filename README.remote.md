# AIC Remote Runner for Mac

This setup runs the AIC evaluation image in Docker and exposes:

- `http://localhost:8080` - web UI for starting/stopping experiments and reading logs.
- `http://localhost:6080/vnc.html?autoconnect=1&resize=scale` - noVNC display for Gazebo/RViz.

The image is based on `ghcr.io/intrinsic-dev/aic/aic_eval:latest`, which is the official Ubuntu 24.04 / ROS 2 Kilted / Gazebo environment used by the AIC toolkit.

## Run

```bash
docker compose up --build
```

Open `http://localhost:8080` on the Mac.

Recommended first flow:

1. Click `Start headless` for the faster non-visual simulator, or `Start Gazebo/RViz only` when you need the visual interface.
2. Open noVNC directly in a separate tab for better performance.
3. Click `Run WaveArm`.
4. Watch logs in the UI and generated files under `./aic_results`.

## Gazebo/RViz Simulation

For the visual simulator like the screenshot, run the Docker service and start the AIC simulator from the web UI:

```bash
cd /Users/fidlobabovich/Desktop/robot_infra
docker compose up --build
```

Open:

```text
http://localhost:8080
```

Then click `Start Gazebo/RViz + engine`.

For best performance, open noVNC directly instead of embedding it inside the UI:

```text
http://localhost:6080/vnc.html?autoconnect=1&resize=scale
```

After Gazebo/RViz are visible, click `Run WaveArm` to start the example ROS policy against the simulator.

If you only want to verify the visual interface first, click `Start Gazebo/RViz only`. That opens Gazebo/RViz without starting the AIC engine loop that waits for `aic_model`.

If you need a mobile warehouse robot in a room, click `Start visual warehouse robot`. This opens a separate Gazebo world with:

- a warehouse room with walls, racks, storage zones, and dispatch zones;
- a small two-wheel-style mobile robot with cargo bed and lidar/camera marker;
- a Gazebo camera initially placed above the robot/work area;
- an automatic route that drives the robot from dispatch to storage and back.

If you need a separate policy run instead of route replay, click `Run warehouse policy`. That uses a dedicated observation-action loop for the same warehouse world:

- observation: current robot pose, goal heading error, target yaw error, distance to waypoint;
- action: linear and angular velocity commands;
- update: pose integration and goal switching when the current waypoint is reached.

If you need the map-based task mode, click `Start Nav2 map task mode`, then open:

```text
http://localhost:8080/nav2
```

The map window lets you:

- choose `ProductR`, `ProductG`, or `ProductB`;
- click a drop point on the map;
- draw keepout/avoid zones as rectangles;
- send a `TaskGoal` so the robot plans to the shelf pickup pose, logs `pick_up`, then drives to the selected drop point and logs `drop_off`.

This mode generates Nav2-compatible artifacts in:

```text
./aic_results/nav2_warehouse_map/
```

Generated files include `warehouse_map.yaml/.pgm`, `keepout_mask.yaml/.pgm`, `nav2_params.yaml`, and `nav_to_pose_and_pause_near_goal_obstacle.xml`. The current container mode also includes a built-in A* fallback controller so the task can run even if the base AIC image does not include the full Nav2 stack.

If you do not need RViz/Gazebo windows, click `Start headless`. The equivalent command is:

```bash
docker compose exec web /entrypoint.sh gazebo_gui:=false launch_rviz:=false ground_truth:=false start_aic_engine:=true shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=600
```

The web runner automatically stops older simulator/policy runs before starting a new simulator run. You can also reset the container manually:

```bash
docker compose restart web
```

Equivalent commands inside the container are:

```bash
/entrypoint.sh gazebo_gui:=true launch_rviz:=true ground_truth:=false start_aic_engine:=true shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=600
```

and, in a second run, the example policy:

```bash
source /ws_aic/install/setup.bash && ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

## Text Warehouse AMR Experiment

The web UI also includes `Run text warehouse demo`. This starts a container-local warehouse picking simulation inspired by:

- `rodriguesrenato/warehouse_robot_simulation`: order queue, storages, products, robot cargo, dispatch areas.
- `shrikrishnarb/amr-ros`: AMR-style task FSM and pickup-to-dropoff flow.

The experiment runs without extra ROS packages so it can execute inside the same AIC container while the full Gazebo image is still downloading or when you need a fast deterministic pipeline test.

It models:

- A warehouse grid with aisles and blocked rack cells.
- Storage zones for `ProductR`, `ProductG`, and `ProductB`.
- Dispatch zones `DispatchA` and `DispatchB`.
- A single AMR with orientation, battery drain, A* route planning, cargo loading, unloading, and order completion.

Artifacts are written to `./aic_results/warehouse_<timestamp>/`:

- `warehouse_summary.json`
- `warehouse_trace.json`
- `warehouse_final_map.txt`

## Apple Silicon

The compose file defaults to `linux/amd64` because the official ROS/Gazebo image may not be available for ARM. On Apple Silicon this runs through Docker Desktop emulation and can be slow:

```bash
AIC_PLATFORM=linux/amd64 docker compose up --build
```

For better performance in Docker Desktop, set resources before running:

- CPUs: 6-8 if available.
- Memory: 10-16 GB if available.
- Swap: 4-8 GB.
- Disk image size: at least 80-120 GB.

After changing Docker Desktop resources, restart Docker Desktop and then run:

```bash
docker compose down
docker compose up --build
```

## Custom Experiments

Use the custom command field for commands inside the container. For example:

```bash
source /ws_aic/install/setup.bash && ros2 topic list
```

or run a different policy:

```bash
source /ws_aic/install/setup.bash && ros2 run aic_model aic_model --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.WaveArm
```

Results are persisted in `./aic_results`; run metadata and logs are persisted in `./aic_runs`.
