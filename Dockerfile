FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
LABEL maintainer="Zhuang Chi Sheng <chngdickson@gmail.com>"
ENV DEBIAN_FRONTEND noninteractive

# Install zsh and git
RUN apt update && apt install -y wget git zsh tmux vim g++ rsync
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p ssh-agent \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting


# Install python
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip

# Install from directory
WORKDIR /root
RUN git clone https://github.com/chngdickson/sdp_tph.git

# RUN pip install laspy
# ENTRYPOINT []
# CMD []
# USER 0
# WORKDIR /
# ENTRYPOINT ["python","./main.py"]
# CMD ["file_dir" "input_file_name" "file_type"]

# docker build -t test-new-image .