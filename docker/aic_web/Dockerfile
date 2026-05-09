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
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/aic_web
COPY --chmod=755 docker/aic_web/remote_entrypoint.sh /opt/aic_web/remote_entrypoint.sh
COPY --chmod=644 docker/aic_web/server.py /opt/aic_web/server.py
COPY --chmod=755 docker/aic_web/warehouse_sim.py /opt/aic_web/warehouse_sim.py
COPY --chmod=755 docker/aic_web/warehouse_visual.py /opt/aic_web/warehouse_visual.py
COPY --chmod=755 docker/aic_web/warehouse_policy.py /opt/aic_web/warehouse_policy.py
COPY --chmod=644 docker/aic_web/warehouse_visual.sdf /opt/aic_web/warehouse_visual.sdf

EXPOSE 8080 6080

ENTRYPOINT ["/opt/aic_web/remote_entrypoint.sh"]
