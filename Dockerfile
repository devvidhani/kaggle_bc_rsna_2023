#### START OF DOCKERFILE ####
# Use NVIDIA CUDA 12.1 base image
# FROM nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04
FROM nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04

# arguments
ARG USERNAME
ARG PASSWORD

# Set the environment variables
ENV NVIDIA_VISIBLE_DEVICES 1
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# updates
RUN apt-get update
RUN apt-get install -y curl gnupg zsh tmux less htop time vim wget
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y tmux

# Prepare for new users
USER root
RUN apt-get update && apt-get install -y sudo

# Add new user
RUN adduser --disabled-password --gecos '' ${USERNAME} && \
    echo "${USERNAME}:${PASSWORD}" | chpasswd && \
    usermod -aG sudo ${USERNAME}

USER ${USERNAME}

# Set up the work directory
WORKDIR /home/${USERNAME}

# Download and install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 && \
    $HOME/miniconda3/bin/conda create -y -n kg_rsna_bc2 numpy pandas scikit-learn pydicom opencv ipython jupyter ipywidgets jupyter_contrib_nbextensions webcolors uri-template isoduration fqdn jsonpointer -c conda-forge -y && \
    $HOME/miniconda3/bin/conda clean -ya

ENV PATH /home/${USERNAME}/miniconda3/bin:$PATH
####

# Download and install conda
# RUN curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
#     bash miniconda.sh -b -p /app/miniconda && \
#     rm miniconda.sh

# Change the default shell to zsh
SHELL ["/bin/zsh", "-c"]

# Run conda init for zsh
RUN /home/${USERNAME}/miniconda3/bin/conda init zsh

# Add miniconda to the path
# ENV PATH /app/miniconda/bin:$PATH
ENV PATH /home/${USERNAME}/miniconda3/bin:$PATH

# Create a conda environment with specific packages
# RUN conda create -n kg_rsna_bc2 python=3.10 numpy pandas scikit-learn pydicom opencv ipython jupyter ipywidgets jupyter_contrib_nbextensions webcolors uri-template isoduration fqdn jsonpointer -c conda-forge -y

# Install autogluon
ENV CONDA_ENV kg_rsna_bc2
RUN /bin/bash -c "source /home/${USERNAME}/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && jupyter contrib nbextension install --user"
RUN /bin/bash -c "source /home/${USERNAME}/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && pip install autogluon pylibjpeg pylibjpeg-libjpeg"

# Activate the conda environment
# ENV PATH /app/miniconda/envs/kg_rsna_bc2/bin:$PATH

# Activate the conda environment
RUN echo “source activate kg_rsna_bc2” > ~/.bashrc ENV PATH /home/${USERNAME}/miniconda3/envs/myenv/bin:$PATH

# Copy the app code to the container
# COPY ./autogluon_beginner_multimodal.py ./autogluon_beginner_multimodal.py
# COPY ./autogluon_custom_metric_serializable.py ./autogluon_custom_metric_serializable.py

#### END OF DOCKERFILE ####

