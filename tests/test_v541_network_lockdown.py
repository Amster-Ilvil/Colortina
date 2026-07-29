"""V5.4.1 — 隐私网络硬锁：默认禁止一切对外连接，仅放行本机回环。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os
    env = dict(os.environ)
    env.pop("COLORTINA_ALLOW_NETWORK", None)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT),
        capture_output=True, text=True, timeout=120, env=env)


def test_lockdown_blocks_outbound_and_allows_loopback():
    code = (
        "from core.network_lockdown import install_lockdown, NetworkLockedError\n"
        "import socket\n"
        "assert install_lockdown() is True\n"
        "s = socket.socket()\n"
        "try:\n"
        "    s.connect(('93.184.216.34', 80))\n"
        "    raise SystemExit('outbound connection was NOT blocked')\n"
        "except NetworkLockedError:\n"
        "    pass\n"
        "finally:\n"
        "    s.close()\n"
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://example.com', timeout=5)\n"
        "    raise SystemExit('urllib was NOT blocked')\n"
        "except Exception as exc:\n"
        "    assert 'NetworkLocked' in type(exc).__name__ or '网络已锁定' in str(exc), exc\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "c = socket.socket(); c.connect(('127.0.0.1', port)); c.close(); srv.close()\n"
        "print('LOCKDOWN-OK')\n")
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LOCKDOWN-OK" in result.stdout


def test_lockdown_respects_explicit_allow():
    code = (
        "from core.network_lockdown import install_lockdown\n"
        "assert install_lockdown() is False\n"
        "import socket\n"
        "assert socket.socket.connect.__name__ == 'connect'\n"
        "print('ALLOW-OK')\n")
    result = _run(code, {"COLORTINA_ALLOW_NETWORK": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALLOW-OK" in result.stdout


def test_main_applies_policy_before_anything_else():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "apply_startup_policy" in src
    body = src.split("def main():", 1)[1]
    assert body.index("apply_startup_policy") < body.index("cv2")


def test_policy_locks_when_weights_present_and_opens_when_missing():
    code = (
        "import os, tempfile, types\n"
        "from core import network_lockdown as nl\n"
        "d = tempfile.mkdtemp()\n"
        "def cfg_for(present):\n"
        "    g = os.path.join(d, 'generator.zip')\n"
        "    dn = os.path.join(d, 'denoiser')\n"
        "    e = os.path.join(d, 'erika.pth')\n"
        "    os.makedirs(dn, exist_ok=True)\n"
        "    if present:\n"
        "        for path in (g, os.path.join(dn, 'net_rgb.pth'), e):\n"
        "            open(path, 'wb').write(b'0' * (65 * 1024))\n"
        "    return types.SimpleNamespace(GENERATOR_WEIGHTS_PATH=g,\n"
        "        DENOISER_WEIGHTS_DIR=dn, MANGA_LINE_WEIGHTS_PATH=e)\n"
        "missing = cfg_for(False)\n"
        "assert nl.required_weights_present(missing) is False\n"
        "state = nl.apply_startup_policy(missing)\n"
        "assert '临时放开' in state, state\n"
        "import socket\n"
        "assert socket.socket.connect.__name__ == 'connect'\n"
        "full = cfg_for(True)\n"
        "assert nl.required_weights_present(full) is True\n"
        "assert nl.lock_when_weights_ready(full) is True\n"
        "assert socket.socket.connect.__name__ == 'guarded_connect'\n"
        "print('POLICY-OK')\n")
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "POLICY-OK" in result.stdout


def test_lockdown_sets_offline_env_for_hf_stack():
    code = (
        "from core.network_lockdown import install_lockdown\n"
        "install_lockdown()\n"
        "import os\n"
        "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "assert os.environ['TRANSFORMERS_OFFLINE'] == '1'\n"
        "assert os.environ['HF_HUB_DISABLE_TELEMETRY'] == '1'\n"
        "print('ENV-OK')\n")
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ENV-OK" in result.stdout
