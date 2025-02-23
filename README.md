Build Docker image

```bash
docker build \
--ssh github_ssh_key=/home/ds1804/.ssh/id_ed25519 \
-t dschng/tph -f Dockerfile .
```

RUN
```bash
docker run -it dschng/tph /bin/bash

cd && docker run -it \
--net=host \
--gpus all \
--privileged \
--volume /dev:/dev \
--volume /tmp/.x11-unix:/tmp/.x11-unix \
--volume ~/.ssh/ssh_auth_sock:/ssh-agent \
--env SSH_AUTH_SOCK=/ssh-agent \
--env DISPLAY=$DISPLAY \
--env TERM=xterm-256color \
-v /home/ds1804/pcds:/root/pcds \
dschng/tph /bin/bash
```

```bash
python3 main.py /root/pcds/ p01 .las
```