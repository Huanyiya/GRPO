直接用环境内 Python 启动，能完全绕过 PATH/激活问题：
source /mnt/cpfs/users/wxh/miniconda3/etc/profile.d/conda.sh
conda activate sandboxfusion-server

cd /mnt/cpfs/users/wxh/SandboxFusion

/mnt/cpfs/users/wxh/miniconda3/envs/sandboxfusion-server/bin/python \
  -m uvicorn sandbox.server.server:app \
  --host 127.0.0.1 \
  --port 8080 \
  --log-level debug \
  2>&1 | tee sandboxfusion.log
看到下面这行就表示服务启动：
Uvicorn running on http://127.0.0.1:8080
然后另开一个终端测试：
curl -sS http://127.0.0.1:8080/run_code \
  -H 'Content-Type: application/json' \
  --data '{"code":"print(123)","language":"python"}'

  你只要在训练机启动前设置可访问的 SandboxFusion 地址即可：
export SANDBOX_FUSION_URL='http://127.0.0.1:8080/run_code'

bash /mnt/cpfs/users/wxh/GRPO/slime/scripts/run-qwen3.5-9B-grpo-nonthinking-code.sh