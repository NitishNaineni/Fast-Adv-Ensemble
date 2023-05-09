FROM condaforge/mambaforge:latest

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
RUN mamba env update -f environment.yml -n base && \
    mamba clean --all --force-pkgs-dirs && \
    rm -rf /root/.cache/pip/*

# Set default command
EXPOSE 8888
CMD ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.trust_xheaders=True", "--ServerApp.disable_check_xsrf=False", "--ServerApp.allow_remote_access=True", "--ServerApp.allow_origin='*'", "--ServerApp.allow_credentials=True"]

