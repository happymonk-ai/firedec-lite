FROM python:3.7.8
# This prevents going stdout statements to go into a buffer first
ENV PYTHONUNBUFFERED 1
# installing essential packages for building and running packages
RUN apt-get install -y \
    build-essential \
    cmake \
    qt5-default \
    libvtk6-dev \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libswscale-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libx264-dev \
    yasm \
    libopencore-amrnb-dev \
    libv4l-dev \
    libxine2-dev \
    libtbb-dev \
    libeigen3-dev \
    doxygen \
    pip install --upgrade pip &&\
    apt-get install zlib \
    zlib-dev \
    libjpeg8 \
    libjpeg8-dev
    apt-get install -y libsm6 libxext6 libxrender-dev &&\
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
## this command will run on container creation
ENTRYPOINT [ "python" ]
CMD [ "firenet.py" ]
