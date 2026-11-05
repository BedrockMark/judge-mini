# 在 Vulcan 上测试 vLLM

`vllm_test/vllm_test.sh` 在登录节点运行（在任何目录执行都行；模型、日志、`.venv-vllm` 都放在项目根目录），它会：

1. 把模型下载到项目的 `./model/<org>__<name>/`（HF 缓存也放在 `./model/.hf_home`），已下载过就跳过；
2. `sbatch vllm_test/vllm_serve.sh`：申请 1 张 GPU，启动 vLLM 的 OpenAI 兼容服务（随机端口 + 随机 API key），并把地址写进 `job_log/vllm_<jobid>.endpoint`；
3. `sbatch --dependency=after:<serve_jobid> vllm_test/vllm_check.sh`：服务作业**开始运行后**才启动（CPU 作业），等待 `/health` 就绪，然后运行 `vllm_test/vllm_client_test.py`
   （模型列表、错误 key 返回 401、普通对话、流式输出、8 个并发请求），最后 `scancel` 掉服务作业。

## 一次性准备

```bash
# 在 GPU 交互作业里安装（helper.md 推荐，驱动更兼容）
salloc --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:59:00
cd <项目根目录>
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm vllm
exit
```

vLLM 装在单独的 `.venv-vllm` 里，不影响项目的 `.venv`。

## 运行

```bash
bash vllm_test/vllm_test.sh                               # 默认 Qwen/Qwen2.5-0.5B-Instruct
bash vllm_test/vllm_test.sh Qwen/Qwen2.5-7B-Instruct      # 任意 HF 模型 ID
KEEP_SERVER=1 bash vllm_test/vllm_test.sh                 # 测试后保留服务
MAX_MODEL_LEN=8192 VLLM_EXTRA_ARGS="--dtype half" bash vllm_test/vllm_test.sh
```

- 结果：`job_log/<check_jobid>.out`，最后一行是 `result=PASS` 或 `result=FAIL`。
- 服务日志：`job_log/<serve_jobid>.err`（vLLM 的日志主要输出到 stderr）。
- 需要登录的模型（如 Llama）先在登录节点 `HF_HOME=$PWD/model/.hf_home .venv-vllm/bin/hf auth login`。
- 服务端等待上限 `READY_TIMEOUT`（默认 600 秒）。大模型请改 `vllm_test/vllm_serve.sh` / `vllm_test/vllm_check.sh` 头部的 `#SBATCH --time`（以及 `--mem`、`--gres=gpu:N`，多卡会自动设置 tensor parallel）。

## 缩短排队

默认资源已压到测试够用的最小值：服务作业 1 GPU / 4 CPU / 16G / 15 分钟，检测作业 1 CPU / 2G / 20 分钟。
Alliance 集群按 `--time` 自动分配 `gpubase_bygpu_b*` 分区（不要手动指定 `--partition`），时限越短、资源越少，越容易被 backfill 插空提前启动。

```bash
sbatch --test-only vllm_test/vllm_serve.sh        # 估计开始时间（不会真正提交）
sinfo -o "%G" | tr ',' '\n' | sort -u             # 查看有哪些 GPU 类型，名字含 1g.10gb 之类的就是 MIG 分片
GPU_GRES=gpu:<分片类型>:1 bash vllm_test/vllm_test.sh   # 有分片时用分片（0.5B 模型一个最小分片就够）
VLLM_EXTRA_ARGS="--enforce-eager" bash vllm_test/vllm_test.sh   # 跳过 CUDA graph 捕获，启动更快
```
