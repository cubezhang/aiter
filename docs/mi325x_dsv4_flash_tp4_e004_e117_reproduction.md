# MI325X DeepSeek-V4-Flash-FP8 TP4 reproduction

本目录保存 MI325X 上 DeepSeek-V4-Flash-FP8 的 E004/E117 保留通用方案及其原始复现文件。

## Included files

- 原始启动脚本：`docs/mi325x_dsv4_flash_tp4/commands/e004_start_dual_tp4_131k_singlecontainer.sh`
- 原始停止脚本：`docs/mi325x_dsv4_flash_tp4/commands/e004_stop_dual_tp4_131k_singlecontainer.sh`
- ATOM benchmark 脚本：`docs/mi325x_dsv4_flash_tp4/commands/e004_run_atom_6100_1024_u70.sh`
- 原始 Locust 脚本：`docs/mi325x_dsv4_flash_tp4/commands/e003_run_original_locust.sh`
- 保留的 30 行 A8W8 调优表：`aiter/configs/model_configs/dsv4_flash_mi325x_tp4_general_a8w8.csv`
- C57 top-k 源码：`docs/mi325x_dsv4_flash_tp4/topk_per_row_kernels.cu`
- Router：`docs/mi325x_dsv4_flash_tp4/router.py`
- Locust 测试程序：`docs/mi325x_dsv4_flash_tp4/locustfile.py`
- ATOM 来源与 commit：`docs/mi325x_dsv4_flash_tp4/atom_origin.txt`、`atom_commit.txt`
- 容器镜像 digest：`docs/mi325x_dsv4_flash_tp4/container_image.txt`
- 文件校验值：`docs/mi325x_dsv4_flash_tp4/sha256sums.txt`

## Host requirements

- 8 张 MI325X
- Docker 可正常访问 `/dev/kfd`、`/dev/dri`
- 已配置 ROCm
- 宿主机已设置 `iommu=pt`
- `kernel.numa_balancing=0`
- 模型位于：

```bash
/data/DeepSeek-V4-Flash-FP8
```

> 不要设置 `AITER_REBUILD=1`。该变量会强制重新编译 AITER JIT 模块，可能因缺少预构建模块或编译环境差异导致启动失败。

## 1. Clone source

```bash
cd /data

git clone https://github.com/cubezhang/aiter.git
cd aiter

git switch mi325-dsv4-ops
git pull --ff-only origin mi325-dsv4-ops

export REPO_ROOT=$PWD
export RUN_ROOT=/data/models/mi325_dsv4_perf_tuning_20260803_065439
```

查看此次复现所依赖的版本：

```bash
cat docs/mi325x_dsv4_flash_tp4/container_image.txt
cat docs/mi325x_dsv4_flash_tp4/atom_origin.txt
cat docs/mi325x_dsv4_flash_tp4/atom_commit.txt
cat docs/mi325x_dsv4_flash_tp4/sha256sums.txt
```

## 2. Required host files

E004 原始脚本使用已验证的宿主机依赖路径：

```bash
# 模型
/data/DeepSeek-V4-Flash-FP8

# 官方 ATOM 源码
/data/models/mi325_dsv4_reuse_tuning_20260801_103501/02_source/ATOM_official

# C57 Top-k 内核
/data/models/mi325_dsv4_model_tuning_round2_20260802/57_topk_ob_hybrid_bpp_20260802_141900/aiter/csrc/kernels/topk_per_row_kernels.cu

# Router
/data/models/mi325_dsv4_reuse_tuning_20260801_103501/08_dual_tp4/router.py

# 保留的 A8W8 调优表
/data/models/mi325_dsv4_perf_tuning_20260803_065439/experiments/e077_bpreshuffle_18row_overlay.csv
```

当前上传的 GitHub 文件保存了这些依赖的副本和校验值；原始 E004 脚本仍保持已验证的绝对路径，以保证同一环境中可直接复现。

## 3. Pull container image

E004 脚本会自动创建 Docker 容器。先拉取已验证镜像：

```bash
docker pull rocm/atom-dev:nightly_202607271535
```

## 4. Start E004 dual TP4 service

停止同名旧服务：

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_stop_dual_tp4_131k_singlecontainer.sh"
```

使用原始 E004 脚本创建并启动容器：

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_start_dual_tp4_131k_singlecontainer.sh"
```

该脚本自动创建：

- `mi325_dsv4_e004_dual`
- GPU `0,1,2,3` 上的 TP4 实例，端口 `18000`
- GPU `4,5,6,7` 上的 TP4 实例，端口 `18001`
- `mi325_dsv4_e004_router`
- least-connections Router，端口 `18080`

查看容器：

```bash
docker ps --filter name=mi325_dsv4_e004
```

查看日志：

```bash
docker logs -f mi325_dsv4_e004_dual
docker logs -f mi325_dsv4_e004_router
```

## 5. Verify service

```bash
curl -sS http://127.0.0.1:18000/v1/models
curl -sS http://127.0.0.1:18001/v1/models
curl -sS http://127.0.0.1:18080/router/status
```

流式请求验证：

```bash
curl -N http://127.0.0.1:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2",
    "messages": [
      {
        "role": "user",
        "content": "你好，请介绍一下你自己。"
      }
    ],
    "max_tokens": 256,
    "stream": true
  }'
```

## 6. ATOM benchmark

测试场景：

- 输入：`6100`
- 输出：`1024`
- 并发：`70`

执行已验证脚本：

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_run_atom_6100_1024_u70.sh"
```

完整 700 请求测试如脚本存在，可执行：

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e152_run_atom_6100_1024_u70_700.sh"
```

结果保存在：

```bash
$RUN_ROOT/atom_benchmark/
```

## 7. Locust benchmark

原始 Locust 测试使用平均约 6000 token 输入、1024 token 输出的混合数据集。

执行：

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e003_run_original_locust.sh"
```

结果保存在：

```bash
$RUN_ROOT/locust/
```

## 8. Stop service

```bash
bash \
  "$REPO_ROOT/docs/mi325x_dsv4_flash_tp4/commands/e004_stop_dual_tp4_131k_singlecontainer.sh"
```

也可以直接停止：

```bash
docker rm -f \
  mi325_dsv4_e004_dual \
  mi325_dsv4_e004_router
```

## Configuration summary

- 双 TP4：GPU `0-3` 与 `4-7`
- Router：least-connections，端口 `18080`
- MTP：`2`
- `max-model-len=131072`
- `max-num-batched-tokens=131072`
- `max-num-seqs=128`
- `gpu-memory-utilization=0.83`
- CUDAGraph sizes：`[1,2,4,8,16,24,32,48,64,72,96,128]`
- AITER Quick Reduce：`INT4`
- Prefix cache：关闭
