# Use the official image as a parent image
FROM mcr.microsoft.com/devcontainers/python:3

# Install Miniconda
RUN curl -sLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p /opt/miniconda \
    && rm /tmp/miniconda.sh

# Update PATH environment variable to include Conda
ENV PATH=/opt/miniconda/bin:$PATH

# Ensure conda is initialized for bash
RUN conda init bash

# Copy environment.yml to the working directory
COPY environment.yml /tmp/environment.yml

# Set the working directory
WORKDIR /workspace

# Activate the environment
CMD ["bash", "-c", "source activate recsys_env && bash"]
