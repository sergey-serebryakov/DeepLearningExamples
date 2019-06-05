# Docker container
1. Set a folder for storing docker images.
It is optional. As an alternative option, local docker hub can be used.
   ```bash
   export DOCKER_BACKUP_DIR=/storage/docker-images
   ```

2. Cloning and serializing docker image
   ```bash
   docker pull nvcr.io/nvidia/pytorch:19.05-py3
   docker save --output ${DOCKER_BACKUP_DIR}/nvcr.io_nvidia_pytorch:19.05-py3.tar.gz nvcr.io/nvidia/pytorch:19.05-py3
   ```

3. Loading docker image from a backup directory
   ```bash
   docker load < ${DOCKER_BACKUP_DIR}/nvcr.io_nvidia_pytorch:19.05-py3.tar.gz
   ```

# Dataset
Input dataset is the raw ImageNet dataset. The root dataset folder must contain two subfolders - `train` and `val` with standard train and validation datasets
```bash
export IMAGENET_JPEGS=/storage/imagenet/jpegs
```

# Project
It is assumed that the benchmark project (this fork or original) was cloned to:
```bash
export DL_EXAMPLES=${HOME}/projects/DeepLearningExamples
```

# Running docker containers:
```bash
nvidia-docker run --privileged --rm -it --ipc=host --net=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /dev/infiniband:/dev/infiniband -v ${DL_EXAMPLES}/PyTorch/Classification/RN50v1.5/:/workspace/resnet50 -v ${IMAGENET_JPEGS}:/mnt/data nvcr.io/nvidia/pytorch:19.05-py3
cd /workspace/resnet50
```

# Single node benchmarks
1. One GPU benchmarks
   ```bash
   batch=256 && python ./main.py --arch resnet50 --training-only -p 1 -b ${batch} --raport-file ./resnet50_01_${batch}.json --epochs 1 --prof 500 --fp16 --static-loss-scale 64 /mnt/data
   ```
 
2. Multi-GPU benchmarks
   ```bash
   ngpus=8 && batch=256  && python ./multiproc.py --nproc_per_node ${ngpus} ./main.py --arch resnet50 --training-only -p 1 -b ${batch} --raport-file ./resnet50_0${ngpus}_${batch}.json --fp16 --static-loss-scale 256 --epochs 1 --prof 500 /mnt/data
   ```
# Multi-GPU benchmarks
Connect to nodes, run containers on each node as described above. Example provided here is for two nodes:
1. On each node run:
   ```bash
   # We'll use Etherner, change to `ib` for Infiniband
   export NCCL_SOCKET_IFNAME=eno1
   # PyTorch processes need to know each other. They use rendezvous address (host and port). It should be one of the hosts. 
   export RENDEZVOUS_HOST=HOST_ID_ADRESS
   export RENDEZVOUS_PORT=29500
   ```

2. First node:
   ```bash
   ngpus=8 && batch=64 && nnodes=2 && rank=0 && python ./multiproc.py --nnodes ${nnodes} --node_rank ${rank} --nproc_per_node ${ngpus} --master_addr ${RENDEZVOUS_HOST} --master_port ${RENDEZVOUS_PORT} ./main.py --arch resnet50 --training-only -p 1 -b ${batch} --raport-file ./resnet50_nodes${nnodes}_0${ngpus}_rank${rank}_${batch}.json --fp16 --static-loss-scale 256 --epochs 1 --prof 500 /mnt/data
   ```

3. Second node:
   ```bash
   ngpus=8 && batch=64 && nnodes=2 && rank=1 && python ./multiproc.py --nnodes ${nnodes} --node_rank ${rank} --nproc_per_node ${ngpus} --master_addr ${RENDEZVOUS_HOST} --master_port ${RENDEZVOUS_PORT} ./main.py --arch resnet50 --training-only -p 1 -b ${batch} --raport-file ./resnet50_nodes${nnodes}_0${ngpus}_rank${rank}_${batch}.json --fp16 --static-loss-scale 256 --epochs 1 --prof 500 /mnt/data
   ```
