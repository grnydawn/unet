# boda_lite.py

import sys
import os
import re
import runpy
import argparse
import json
import tarfile
import importlib
import importlib.util
from importlib import abc
import traceback
from pathlib import Path
import tempfile
import shutil

boda_module_name = "bodalitemodule"
boda_record_struct = "fd"
boda_pat_v0 = re.compile(r"^(\s*)#@boda\s+(\w+)(.*)$", re.MULTILINE)

boda_module = '''\
import threading
import struct
import time
import json
import os
import socket
import tarfile
import torch

RECORD_STRUCT = struct.Struct('{recstruct}')

res_lock = threading.Lock()
res_map = dict()

rec_lock = threading.Lock()
rec_map = dict()

maxsize_inbytes = {maxsize_inbytes}

def flush_record(records, rec_path, res_path):
    if len(records) > 0:
        with open(rec_path, 'ab') as f:
            f.write(records)
        records.clear()

    with res_lock:
        with open(res_path, "w") as f:
            rpath = os.path.relpath(rec_path, os.path.dirname(res_path))
            res_map["file"][rpath] = -1 # value is reserved for futher use
            json.dump(res_map, f, indent=2)

def add_record(label: str):
    ts = time.time()
    if label not in res_map["labels"]:
        with res_lock:
            if label not in res_map["labels"]:
                res_map["labels"][label] = len(res_map["labels"])
    
    tid = threading.get_ident()
    records = rec_map[tid]
    records.extend(RECORD_STRUCT.pack(ts, res_map["labels"][label]))
    if len(records) > maxsize_inbytes:
        flush_record(records, rec_map["rec_path"][tid], rec_map["res_path"])

def _boda_profile_start():

    hostname = socket.gethostname()
    pid = os.getpid()

    with res_lock:
        tid = threading.get_ident()
        if "tid" not in res_map:
            res_map["tid"] = dict()

        res_map["tid"][tid] = len(res_map["tid"])

        if "file" not in res_map:
            res_map["file"] = dict()

        if "res_path" not in rec_map:
            rec_map["res_path"] = os.path.join("{outdir}", f"bodadata.{{hostname}}.{{pid}}.res")

        if "labels" not in res_map:
            res_map["labels"] = dict()

    with rec_lock:
        tid = threading.get_ident()
        rec_map[tid] = bytearray()
        if "rec_path" not in rec_map:
            rec_map["rec_path"] = dict()

        if tid not in rec_map["rec_path"]:
            _t = res_map['tid'][tid]
            rec_map["rec_path"][tid] = os.path.join("{outdir}", f"bodadata.{{hostname}}.{{pid}}.{{_t}}.rec")

def _boda_profile_stop():
    tid = threading.get_ident()
    flush_record(rec_map[tid], rec_map["rec_path"][tid], rec_map["res_path"])
    
    if threading.current_thread() == threading.main_thread():

        with tarfile.open("{bodafile}", 'w:gz') as tar:
            for file in rec_map["rec_path"].values():
                filename_only = os.path.basename(file)
                tar.add(file, arcname=filename_only)
            filename_only = os.path.basename(rec_map["res_path"])
            tar.add(rec_map["res_path"], filename_only)

        for file in rec_map["rec_path"].values():
            os.remove(file)
        os.remove(rec_map["res_path"])

def _boda_profile_event(label=""):
    add_record(label)
'''


def modify_script(original_path, tmp_dir) -> str:

    # Check for #@boda lines
    try:
        with open(original_path, 'r') as f:
            content = f.read()
    except Exception as e:
        import pdb; pdb.set_trace()
        return None

    pointer = 0
    new_content = ""
    for match in boda_pat_v0.finditer(content):
        start, stop = match.span()
        indent, command, args = match.groups()
        new_content += content[pointer:start]
        if command == "start":
            new_content += f"{indent}{boda_module_name}._boda_profile_start({args})\n"
        elif command == "stop":
            new_content += f"{indent}{boda_module_name}._boda_profile_stop({args})\n"
        elif command == "event":
            new_content += f"{indent}{boda_module_name}._boda_profile_event({args})\n"

        pointer = stop

    if pointer > 0:
        new_content += content[pointer:]
        new_content = f"import {boda_module_name}\n" + new_content

        # TODO: handle two common name files
        tmp_file_path = Path(tmp_dir) / ("boda_" + original_path.name)
        with open(tmp_file_path, 'w') as f:
            f.write(new_content)
        print(f"MODIFIED from {original_path} to {tmp_file_path}")
        return tmp_file_path

    return original_path

class TrackImportsFinder(abc.MetaPathFinder):

    def __init__(self, tmp_dir):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.used_modules = {}

    def find_spec(self, fullname, path, target=None):
        try:
            # Temporarily remove this finder to avoid recursion
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)  # Add it back
            if spec and spec.origin and spec.origin.endswith(".py"):
                original_path = Path(spec.origin).resolve()

                if original_path in self.used_modules:
                    return spec  # Already processed

                new_path = modify_script(original_path, self.tmp_dir)
                if new_path == original_path:
                    self.used_modules[original_path] = original_path
                else:
                    self.used_modules[original_path] = new_path
                    spec = importlib.util.spec_from_file_location(fullname, new_path)
                    print(f"NEW SPEC from {original_path} to {new_path}")
                return spec
            return None
        except Exception as e:
            return None


def _make_list(value):
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]

def parse_arguments():

    parser = argparse.ArgumentParser(description="boda_lite.py")
    parser.add_argument("--boda-tmpdir", type=str, help="Temporary working directory")
    parser.add_argument("--boda-outdir", type=str, default="boda_output", help="Output directory")
    parser.add_argument("--boda-files",  type=_make_list, default=[], help="Boda files")
    parser.add_argument("--boda-maxsize",type=int, default=int(1E6), help="Maximum record file size")
    parser.add_argument("target_script", type=str, nargs="?", help="Python script to run")
    parser.add_argument("target_args",   nargs=argparse.REMAINDER, help="Args for target script")

    args = parser.parse_args()

    if args.boda_tmpdir is None:
        args.boda_tmpdir = tempfile.mkdtemp(prefix="bodalite_")

    args.boda_outdir = os.path.abspath(args.boda_outdir)

    if not os.path.isdir(args.boda_outdir):
        os.makedirs(args.boda_outdir, exist_ok=True)

    print(f"TMPDIR: {args.boda_tmpdir}")
    print(f"OUTDIR: {args.boda_outdir}")

    return args

def create_boda_module(args):

    path = Path(args.boda_tmpdir) / f"{boda_module_name}.py"

    with open(path, 'w') as f:
        boda_file = os.path.join(args.boda_outdir,
                    os.path.splitext(os.path.basename(args.target_script))[0] +
                    ".boda")

        f.write(boda_module.format(
            outdir=args.boda_outdir,
            bodafile=boda_file,
            recstruct=boda_record_struct,
            maxsize_inbytes=args.boda_maxsize))

        args.boda_files.append(boda_file)

    return path


def instrument_code(args, modpath):

    script_path = modify_script(Path(args.target_script).resolve(), args.boda_tmpdir)
    sys.path.insert(0, str(script_path.parent))
    sys.meta_path.insert(0, TrackImportsFinder(args.boda_tmpdir))

    return script_path

def instrument(args):

    # create boda module
    boda_module_path = create_boda_module(args)

    # instrument
    script_path = instrument_code(args, boda_module_path)

    return script_path

def collect_events(args):

    boda_res_files = []

    for idx, boda_file in enumerate(args.boda_files):
        if tarfile.is_tarfile(boda_file):
            tmpdir = os.path.join(args.boda_tmpdir, f"tmpboda_{idx}") 
            with tarfile.open(boda_file, 'r:gz') as tar:
                shutil.rmtree(tmpdir, ignore_errors=True)
                os.makedirs(tmpdir)
                tar.extractall(path=tmpdir)
            for f in Path(tmpdir).iterdir():
                if f.is_file() and f.name.endswith(".res"):
                    boda_res_files.append(f)
        else:
            boda_res_files.append(boda_file)

    for boda_res_file in boda_res_files:
        try:
            with open(boda_res_file, "r") as f:
                resdir = os.path.dirname(boda_res_file)
                data = json.load(f)
                import pdb; pdb.set_trace()
        except (json.JSONDecodeError, OSError):
            pass

def generate_report():
    pass

def analyze(args):

    collect_events(args)

    generate_report()


def main():

    # parse command-line arguments
    args = parse_arguments()

    # run instrumented code
    try:

        if args.target_script:
            # instrument script
            script_path = instrument(args)

            # Save original argv
            original_argv = sys.argv.copy()

            # Set sys.argv to simulate running the target script
            sys.argv = [args.target_script] + args.target_args

            runpy.run_path(str(script_path), run_name="__main__")

    except Exception:
        print("Error during execution of the script:")
        traceback.print_exc()

    finally:

        # Restore original argv
        sys.argv = original_argv

        # generate analysis report
        analyze(args)

        #shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()

