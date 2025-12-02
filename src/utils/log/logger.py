import os
import yaml
import subprocess
import platform
import pkg_resources
from datetime import datetime

def get_git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()

        diff = subprocess.check_output(
            ["git", "diff"], stderr=subprocess.STDOUT
        ).decode().strip()

        return {"commit": commit, "dirty": bool(diff)}
    except:
        return {"commit": "unknown", "dirty": False}

def get_env_info():
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "packages": {p.project_name: p.version for p in pkg_resources.working_set},
    }

class ExperimentLogger:
    def __init__(self, config, base):
        self.base = base
        os.makedirs(self.base, exist_ok=True)
        os.makedirs(f"{self.base}/output", exist_ok=True)

        # Save config
        with open(f"{self.base}/config.yaml", "w") as f:
            yaml.safe_dump(config, f)

        # Save code state
        with open(f"{self.base}/code_state.yaml", "w") as f:
            yaml.safe_dump(get_git_info(), f)

        # Save env info
        with open(f"{self.base}/env.yaml", "w") as f:
            yaml.safe_dump(get_env_info(), f)

        # Create main log file
        self.logfile = open(f"{self.base}/log.txt", "w")

    def log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logfile.write(f"[{timestamp}] {msg}\n")
        self.logfile.flush()
        print(msg)

    def save_output(self, path_local, filename):
        target = f"{self.base}/output/{filename}"
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(path_local, target)
        return target

    def close(self):
        self.logfile.close()