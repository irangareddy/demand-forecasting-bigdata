FROM bitnami/spark:3.4.1

USER root

RUN install_packages python3 python3-pip curl git

WORKDIR /app

# Install Python packages
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

# Copy your source code
COPY src ./src
COPY scripts ./scripts
COPY data ./data
