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
RUN git config --global url.https://.insteadOf git://

# Install python
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip

# Install CV2 V4.11.0
RUN mkdir -p /root/opencv_build 
WORKDIR /root/opencv_build
RUN apt-get update && apt-get install -y software-properties-common
RUN apt install build-essential cmake git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y
RUN git clone https://github.com/opencv/opencv.git --branch 4.11.0 --single-branch
RUN git clone https://github.com/opencv/opencv_contrib.git --branch 4.11.0 --single-branch
RUN mkdir -p /root/opencv_build/opencv/build 
WORKDIR /root/opencv_build/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D OPENCV_ENABLE_NONFREE=True \
-D BUILD_EXAMPLES=ON \
-D BUILD_opencv_java=OFF \
-D OPENCV_EXTRA_MODULES_PATH=/root/opencv_build/opencv_contrib/modules \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
..
RUN make
RUN make install

# Install from directory
WORKDIR /root
RUN git clone --recursive https://github.com/chngdickson/sdp_tph.git

# Install python Requirements
WORKDIR /root/sdp_tph
RUN ./install_scipt.sh
# Install CSF to python
WORKDIR /root/sdp_tph/submodules/CSF
RUN python3 setup.py build
RUN python3 setup.py install

ENTRYPOINT []
# RUN pip install laspy
# ENTRYPOINT []
# CMD []
# USER 0
# WORKDIR /
# ENTRYPOINT ["python","./main.py"]
# CMD ["file_dir" "input_file_name" "file_type"]

# docker build -t test-new-image .