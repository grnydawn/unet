#!/lustre/orion/cli115/proj-shared/grnydawn/repos/github/unet/venv/bin/python3

import sys
import runpy
import importlib
import importlib.util
from importlib import abc
import traceback
from pathlib import Path
import tempfile
import shutil

used_modules = {}
temp_dir = tempfile.mkdtemp(prefix="boda_lite_")

def create_boda_module():
    example_code = '''\
import time
import struct
import random
import threading
from typing import List, Tuple
import torch

with threading.Lock():
    _boda_env = Dict()
    # Define record format: double (timestamp), 10-char string, 2 ints, 3 floats
    RECORD_STRUCT = struct.Struct('d10sii3f')
    RECORD_FILE = "records.bin"

    #(max_records=10000, file_path=RECORD_FILE):
    buffer: List[bytes] = []
    max_records = max_records
    file_path = file_path

def add_record(tag: str, int1: int, int2: int, f1: float, f2: float, f3: float):
    ts = time.time()
    tag_bytes = tag.encode("utf-8")[:10].ljust(10, b' ')  # Ensure 10 bytes
    packed = RECORD_STRUCT.pack(ts, tag_bytes, int1, int2, f1, f2, f3)
    buffer.append(packed)

    if len(buffer) >= max_records:
        flush()

def flush():
    if not buffer:
        return
    with open(file_path, 'ab') as f:
        f.writelines(self.buffer)
    print(f"Flushed {len(buffer)} records to {file_path}")
    buffer.clear()

def close():
    flush()

'''

    path = Path(temp_dir) / "bodalitemodule.py"
    path.parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

    with open(path, 'w') as f:
        f.write(example_code)

def modify_script(original_path) -> str:
    # Check for #@boda lines
    try:
        with open(original_path, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None

    modified = False
    modified_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#@boda"):
            modified = True
            leading_spaces = line[:len(line) - len(stripped)]
            modified_lines.append(f"{leading_spaces}#XXX\n")
            modified_lines.append(f"{leading_spaces}#replaced\n")
            print(f"FOUND at {original_path}.")
        else:
            modified_lines.append(line)

    if modified:
        inserted = False
        for idx in range(len(modified_lines)):
            stripped = modified_lines[idx].lstrip()
            if stripped and not stripped.startswith("#"):
                inserted = True
                modified_lines.insert(idx, "import bodalitemodule\n")
                break
        
        if not inserted:
            modified_lines.insert(0, "import bodalitemodule")

        # Write to a temporary modified file
        temp_file_path = Path(temp_dir) / original_path.name
        with open(temp_file_path, 'w') as f:
            f.writelines(modified_lines)
        return temp_file_path

    return original_path


class TrackImportsFinder(abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        try:
            spec = importlib.util.find_spec(fullname)
            if spec and spec.origin and spec.origin.endswith(".py"):
                original_path = Path(spec.origin).resolve()

                if original_path in used_modules:
                    return None  # Already processed

                new_path = modify_script(original_path)
                if new_path == original_path:
                    used_modules[original_path] = original_path
                else:
                    used_modules[original_path] = new_path
                    spec = importlib.util.spec_from_file_location(fullname, new_path)

                return spec

            return None
        except Exception:
            return None

def main(script_path):

    sys.path.insert(0, temp_dir)
    create_boda_module()

    script_path = modify_script(Path(script_path).resolve())
    sys.path.insert(0, str(script_path.parent))
    sys.meta_path.insert(0, TrackImportsFinder())

    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except Exception:
        print("Error during execution of the script:")
        traceback.print_exc()
    finally:
        print("\nUsed Python files:")
        for orig, mod in used_modules.items():
            if orig != mod:
                print(f"{orig} -> MODIFIED")
            else:
                print(f"{orig}")

        # Optionally, remove temp_dir after use
        # shutil.rmtree(temp_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python boda_lite.py <script_to_run.py>")
        sys.exit(1)
    main(sys.argv[1])
