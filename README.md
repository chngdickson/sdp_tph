Build Docker image

```bash
docker build \
--ssh github_ssh_key=/home/ds1804/.ssh/id_ed25519 \
-t dschng/tph -f Dockerfile .
```

RUN
```bash
docker run -it dschng/tph /bin/bash
```

```bash
test
```