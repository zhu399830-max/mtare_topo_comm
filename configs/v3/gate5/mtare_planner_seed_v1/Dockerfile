FROM mtare-semantic-runtime:local

USER root
COPY source_sha256.txt /tmp/mtare_planner_seed_source_sha256.txt
COPY apply_mtare_planner_seed_v1.py /tmp/apply_mtare_planner_seed_v1.py
RUN cd /home/docker-user/mtare/tare_system/src/tare_planner \
    && sha256sum -c /tmp/mtare_planner_seed_source_sha256.txt \
    && python3 /tmp/apply_mtare_planner_seed_v1.py "$PWD"
RUN /bin/bash -lc 'source /opt/ros/noetic/setup.bash && cd /home/docker-user/mtare/tare_system && catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5'

LABEL org.mtare.reproducibility.planner_seed="v1"
LABEL org.mtare.reproducibility.base_image_id="sha256:9819eea672b2001ed18f12ee09c67f0da53a5ba6ce69192e2f21c128e5e78f9a"
