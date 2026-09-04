# syntax=docker/dockerfile:1
#
# Container for running the superdex-scenarios headless, against the released
# SuperDex wheels from PyPI (no C++ toolchain required).
#
# Build from the directory that contains BOTH checkouts side by side
# (project_superdex/ and superdex-scenarios/):
#
#   cd /home/relari/sim
#   docker build -f superdex-scenarios/Dockerfile -t superdex-scenarios .
#
# Targets:
#   minimal (default) - physics simulation, telemetry, and replay export only.
#                       Runs `--headless --skip-video` by default.
#   video             - adds headless Firefox + geckodriver so the PBR
#                       desk/wrist-camera MP4s are rendered too.
#
#   docker build -f superdex-scenarios/Dockerfile --target video -t superdex-scenarios:video .
#
# Run (mount a host directory to keep the export bundles):
#
#   docker run --rm -v "$PWD/exports:/opt/scenarios/scenarios/ball_bowl/exports" superdex-scenarios
#   docker run --rm superdex-scenarios --dry-run
#   docker run --rm superdex-scenarios --plan-only --seed 1234
#
# Viewer (`no option`) and `--debugger` modes need a display and are not
# supported in the container.

ARG SUPERDEX_VERSION=1.0.0
ARG GECKODRIVER_VERSION=0.36.0

# ---------------------------------------------------------------------------
# Shared runtime base: Python 3.12 (SuperDex pins >=3.12,<3.13), SuperDex
# wheels, assets, and the scenarios checkout. The native superdex modules link
# only libc/libstdc++, so no extra system libraries are needed for headless
# physics.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS base
ARG SUPERDEX_VERSION

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir "superdex-lab==${SUPERDEX_VERSION}" \
    # The scenario's video exporter looks for `ffmpeg` on PATH; imageio-ffmpeg
    # (a superdex-physics dependency) already bundles a static ffmpeg build, so
    # expose it instead of installing the ~300 MB Debian ffmpeg stack.
    && ln -s "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
        /usr/local/bin/ffmpeg

# SuperDex assets (robot prefabs, meshes); resolved via SUPERDEX_ASSETS_PATH.
COPY project_superdex/assets /opt/superdex/assets

# The scenarios repo (exports/, studio/, __pycache__ are excluded by
# Dockerfile.dockerignore; pbr/node_modules stays in for the video target).
COPY superdex-scenarios /opt/scenarios

ENV SUPERDEX_ASSETS_PATH=/opt/superdex/assets

WORKDIR /opt/scenarios
ENTRYPOINT ["python", "/opt/scenarios/scenarios/ball_bowl/runner.py"]

# ---------------------------------------------------------------------------
# video: adds headless Firefox + geckodriver, which the PBR exporter drives
# (via WebDriver, software WebGL) to render the desk/wrist-camera MP4s.
# ---------------------------------------------------------------------------
FROM base AS video
ARG GECKODRIVER_VERSION

RUN apt-get update \
    && apt-get install -y --no-install-recommends firefox-esr ca-certificates curl \
    && arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) gd_arch=linux64 ;; \
         arm64) gd_arch=linux-aarch64 ;; \
         *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
       esac \
    && curl -fsSL -o /tmp/geckodriver.tar.gz \
        "https://github.com/mozilla/geckodriver/releases/download/v${GECKODRIVER_VERSION}/geckodriver-v${GECKODRIVER_VERSION}-${gd_arch}.tar.gz" \
    && tar -xzf /tmp/geckodriver.tar.gz -C /usr/local/bin geckodriver \
    && rm /tmp/geckodriver.tar.gz \
    && apt-get purge -y curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Firefox needs a writable home for its headless profile.
ENV HOME=/tmp \
    MOZ_HEADLESS=1

CMD ["--headless"]

# ---------------------------------------------------------------------------
# minimal: last stage, so a plain `docker build` produces it. Skips the MP4
# renders; still writes the physics replay, telemetry, and contact data.
# ---------------------------------------------------------------------------
FROM base AS minimal
CMD ["--headless", "--skip-video"]
