# BigDataProj/Dockerfile  ← make sure the filename is exactly 'Dockerfile'

FROM bitnami/spark:3.5.5

RUN pip install --no-cache-dir numpy pandas && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

USER 0

RUN echo "spark:x:1001:1001::/home/spark:/bin/bash" >> /etc/passwd && \
    mkdir -p /home/spark && chown 1001:1001 /home/spark
    
USER 1001 