FROM condaforge/mambaforge:latest

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from environment.yml
COPY environment.yml .
RUN mamba env create -f environment.yml && \
    mamba clean --all --force-pkgs-dirs && \
    rm -rf /root/.cache/pip/*

# Set default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--LabApp.trust_xheaders=True", "--LabApp.disable_check_xsrf=False", "--LabApp.allow_remote_access=True", "--LabApp.allow_origin='*'"]
EXPOSE 8888