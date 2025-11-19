import os
import sys
import subprocess
import importlib.util
import yaml
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REQUIRED_ENV_VARS = [
    'API_FOOTBALL_KEY',
    'API_FOOTBALL_HOST',
    'FOOTBALL_DATA_TOKEN',
    'SPORTSDATAIO_KEY',
    # Add more as needed
]

REQUIRED_CONFIGS = [
    'config/api_endpoints.yaml',
    'config/paths.yaml',
    'config/model_params.yaml',
    'config/system.yaml',
    'config/config.yaml',
]

REQUIRED_FILES = [
    'data/reference/valid_matches.csv',
]

DASHBOARD_PORT = 8501
MCP_SERVER_URL = 'http://localhost:8080/status'


def check_python_version():
    print('Checking Python version...')
    if sys.version_info < (3, 10):
        print('❌ Python 3.10+ is required.')
        return False
    print('✅ Python version is OK.')
    return True

def check_dependencies():
    print('Checking Python dependencies...')
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'check'])
        print('✅ All dependencies are installed.')
        return True
    except subprocess.CalledProcessError:
        print('❌ Some dependencies are missing or incompatible.')
        return False

def check_env_file():
    print('Checking .env file...')
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if not os.path.exists(env_path):
        print('❌ .env file is missing.')
        return False
    with open(env_path) as f:
        env_lines = f.readlines()
    env_vars = {line.split('=')[0].strip() for line in env_lines if '=' in line and not line.strip().startswith('#')}
    missing = [var for var in REQUIRED_ENV_VARS if var not in env_vars]
    if missing:
        print(f'❌ Missing env vars: {missing}')
        return False
    print('✅ .env file and variables are OK.')
    return True

def check_config_files():
    print('Checking config files...')
    all_ok = True
    for rel_path in REQUIRED_CONFIGS:
        path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(path):
            print(f'❌ Missing config: {rel_path}')
            all_ok = False
        else:
            try:
                with open(path) as f:
                    yaml.safe_load(f)
            except Exception as e:
                print(f'❌ Invalid YAML in {rel_path}: {e}')
                all_ok = False
    if all_ok:
        print('✅ All config files are present and valid.')
    return all_ok

def check_required_files():
    print('Checking required data files...')
    all_ok = True
    for rel_path in REQUIRED_FILES:
        path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(path):
            print(f'❌ Missing file: {rel_path}')
            all_ok = False
    if all_ok:
        print('✅ All required files are present.')
    return all_ok

def check_mcp_server():
    print('Checking Fircrawl MCP server...')
    try:
        resp = requests.get(MCP_SERVER_URL, timeout=3)
        if resp.status_code == 200:
            print('✅ MCP server is running.')
            return True
        else:
            print(f'❌ MCP server returned status {resp.status_code}')
            return False
    except Exception as e:
        print(f'❌ MCP server not reachable: {e}')
        return False

def check_dashboard_port():
    print(f'Checking if dashboard port {DASHBOARD_PORT} is free...')
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', DASHBOARD_PORT))
    sock.close()
    if result == 0:
        print(f'❌ Port {DASHBOARD_PORT} is already in use.')
        return False
    print(f'✅ Port {DASHBOARD_PORT} is free.')
    return True

def main():
    checks = [
        check_python_version(),
        check_dependencies(),
        check_env_file(),
        check_config_files(),
        check_required_files(),
        check_mcp_server(),
        check_dashboard_port(),
    ]
    print('\n--- Health Check Summary ---')
    if all(checks):
        print('✅ All checks passed. System is ready!')
        sys.exit(0)
    else:
        print('❌ Some checks failed. Please fix the above issues.')
        sys.exit(1)

if __name__ == '__main__':
    main() 