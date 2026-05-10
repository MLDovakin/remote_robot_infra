FROM ghcr.io/intrinsic-dev/aic/aic_eval:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      fluxbox \
      novnc \
      python3 \
	      websockify \
	      wmctrl \
	      x11vnc \
	      xdotool \
	      xvfb && \
	    NAV2_PACKAGES="" && \
	    if [ -n "${ROS_DISTRO:-}" ]; then \
	      for pkg in \
	        ros-${ROS_DISTRO}-navigation2 \
	        ros-${ROS_DISTRO}-nav2-bringup \
	        ros-${ROS_DISTRO}-nav2-simple-commander \
	        ros-${ROS_DISTRO}-nav2-smac-planner \
	        ros-${ROS_DISTRO}-slam-toolbox \
	        ros-${ROS_DISTRO}-robot-localization; do \
	        if apt-cache show "$pkg" >/dev/null 2>&1; then NAV2_PACKAGES="$NAV2_PACKAGES $pkg"; fi; \
	      done; \
	    fi && \
	    if [ -n "$NAV2_PACKAGES" ]; then \
	      apt-get install -y --no-install-recommends $NAV2_PACKAGES; \
	    else \
	      echo "Nav2 apt packages were not found for ROS_DISTRO=${ROS_DISTRO:-unset}; using bundled fallback planner"; \
	    fi && \
	    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aic_web
COPY --chmod=755 docker/aic_web/remote_entrypoint.sh /opt/aic_web/remote_entrypoint.sh
COPY --chmod=644 docker/aic_web/server.py /opt/aic_web/server.py
COPY --chmod=755 docker/aic_web/warehouse_sim.py /opt/aic_web/warehouse_sim.py
COPY --chmod=755 docker/aic_web/warehouse_visual.py /opt/aic_web/warehouse_visual.py
COPY --chmod=755 docker/aic_web/warehouse_policy.py /opt/aic_web/warehouse_policy.py
COPY --chmod=755 docker/aic_web/warehouse_nav2_mode.py /opt/aic_web/warehouse_nav2_mode.py
COPY --chmod=755 docker/aic_web/warehouse_lidar_random_mode.py /opt/aic_web/warehouse_lidar_random_mode.py
COPY --chmod=644 docker/aic_web/warehouse_visual.sdf /opt/aic_web/warehouse_visual.sdf

EXPOSE 8080 6080

ENTRYPOINT ["/opt/aic_web/remote_entrypoint.sh"]
