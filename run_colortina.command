#!/bin/zsh
set -u

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "首次启动：正在创建 venv 虚拟环境..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "找不到 python3。请先安装 Python 3.11 或更新版本。"
    echo
    echo "按任意键退出..."
    read -k 1
    exit 1
  fi
  python3 -m venv venv
fi

VENV_PY=""
if [ -x "venv/bin/python" ]; then
  VENV_PY="venv/bin/python"
elif [ -x "venv/bin/python3" ]; then
  VENV_PY="venv/bin/python3"
fi

if [ -z "$VENV_PY" ]; then
  echo "venv 虚拟环境不完整。请删除 venv 文件夹后重新双击本文件。"
  echo
  echo "按任意键退出..."
  read -k 1
  exit 1
fi

source venv/bin/activate

if [ ! -f "venv/.colortina-installed" ]; then
  echo "正在安装/检查依赖，这一步首次运行会比较久..."
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
  exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo
    echo "依赖安装失败，退出码：$exit_code"
    echo "请检查网络后重新双击本文件。"
    echo "按任意键关闭窗口..."
    read -k 1
    exit $exit_code
  fi
  echo "ok" > venv/.colortina-installed
fi

"$VENV_PY" main.py

exit_code=$?
if [ $exit_code -ne 0 ]; then
  echo
  echo "Colortina 启动失败，退出码：$exit_code"
  echo "按任意键关闭窗口..."
  read -k 1
fi
