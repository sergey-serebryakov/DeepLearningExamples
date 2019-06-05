# Docker container
1. Set a folder for storing docker images.
It is optional. As an alternative option, local docker hub can be used.
   ```bash
   export DOCKER_BACKUP_DIR=/storage/docker-images
   ```

2. Cloning and serializing docker image
   ```bash
   docker pull nvcr.io/nvidia/mxnet:19.05-py3
   docker save --output ${DOCKER_BACKUP_DIR}/nvcr.io_nvidia_mxnet:19.05-py3.tar.gz nvcr.io/nvidia/mxnet:19.05-py3
   ```

3. Loading docker image from a backup directory
   ```bash
   docker load < ${DOCKER_BACKUP_DIR}/nvcr.io_nvidia_mxnet:19.05-py3.tar.gz
   ```

# Dataset

Input dataset is the ImageNet dataset in MXNET's RecordIO format. The folder must contain the following files: `train.idx`, `train.lst`, `train.rec`, `val.idx`, `val.lst`, `val.rec`.
Assuming your ImageNet dataset is located in `/storage/imagenet-jpegs` and mxnet's data location is `/storage/imagenet-recordio`:
```bash
nvidia-docker run --rm -it --ipc=host -u $(id -u ${USER}):$(id -g ${USER}) -v /storage/imagenet-jpegs:/mnt/imagenet -v /storage/imagenet-recordio:/mnt/mxnet nvcr.io/nvidia/mxnet:19.05-py3
cd /mnt/mxnet
python /opt/mxnet/tools/im2rec.py --list --recursive train /mnt/imagenet/train
python /opt/mxnet/tools/im2rec.py --list --recursive val /mnt/imagenet/val
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 train /mnt/imagenet/train
python /opt/mxnet/tools/im2rec.py --pass-through --num-thread 40 val /mnt/imagenet/val
```

# Project
It is assumed that the benchmark project (this fork or original) was cloned to:
```bash
export DL_EXAMPLES=${HOME}/projects/DeepLearningExamples
```

# Running docker containers:
```bash
nvidia-docker run --privileged --rm -it --ipc=host --net=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /dev/infiniband:/dev/infiniband -v ${DL_EXAMPLES}/PyTorch/Classification/RN50v1.5/:/workspace/resnet50 -v /storage/imagenet-recordio:/mnt/data nvcr.io/nvidia/mxnet:19.05-py3
cd /workspace/resnet50
```

# Single node benchmarks
Run tests with 1, 2, 4 and 8 GPUs using the following replica batch sizes - 64, 128, 256 and 512.
```bash
python benchmark.py -n 1,2,4,8 -b 64,128,256,512  --dtype float16 -o /root/resnet50_fp16.json  --data-root /mnt/data -i 100 -e 12 -w 4 --num-examples 25600
```

Where:
1. `-n` Number of GPUs to use
2. `-b` Replica (per-GPU) batch size
3. `--dtype` Precision
4. `-o` File name of a benchmark report
5. `--data-root` Path to training and validation data sets
6. `-e` Number of epochs to benchmark
7. `-w` Number of warm-up epochs
8. `--num-examples` Number examples in one epoch

# Multi-GPU benchmarks
Connect to nodes, run containers on each node as described above. Example provided here is for two nodes:
1. On each node run:
   ```bash
   # We'll use Etherner, change to `ib` for Infiniband
   export NCCL_SOCKET_IFNAME=eno1
   # PyTorch processes need to know each other. They use rendezvous address (host and port). It should be one of the hosts. 
   export RENDEZVOUS_HOST=HOST_ID_ADRESS
   export RENDEZVOUS_PORT=29500
   export RENDEZVOUS=${NCCL_SOCKET_IFNAME}:${RENDEZVOUS_HOST}:${RENDEZVOUS_PORT}

   ```

2. On the first (master) host run:
   ```bash
    batch=128 && gpus_per_node=8 && nnodes=2 && ./dist_runner --rendezvous ${RENDEZVOUS} --num_workers ${nnodes}  --scheduler -n ${gpus_per_node} -b ${batch} --benchmark-iters 100 -e 12 --dtype float16 --report ./test.json  --data-root /mnt/data  --num-examples 25600
   ``` 
   
3. On other nodes run:
   ```bash
    batch=128 && gpus_per_node=8 && nnodes=2 && ./dist_runner --rendezvous ${RENDEZVOUS} --num_workers ${nnodes}  -n ${gpus_per_node} -b ${batch} --benchmark-iters 100 -e 12 --dtype float16 --report ./test.json  --data-root /mnt/data  --num-examples 25600
   ```
   The only difference is the `--scheduler` switch.