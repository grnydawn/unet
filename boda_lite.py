# boda_lite.py

import sys
import os
import re
import runpy
import argparse
import importlib
import importlib.util
from importlib import abc
import traceback
from pathlib import Path
import tempfile
import shutil

boda_module_name = "bodalitemodule"

boda_pat_v0 = re.compile(r"^(\s*)#@boda\s+(\w+)(.*)$", re.MULTILINE)

boda_module = '''\
import torch

def _boda_profile_start():
    print("profile start")

def _boda_profile_stop():
    print("profile stop")

def _boda_profile_event():
    print("collect event")
'''


def modify_script(original_path, tmp_dir) -> str:

    # Check for #@boda lines
    try:
        with open(original_path, 'r') as f:
            content = f.read()
    except Exception:
        return None

    pointer = 0
    new_content = ""
    for match in boda_pat_v0.finditer(content):
        start, stop = match.span()
        indent, command, _ = match.groups()
        new_content += content[pointer:start]
        if command == "start":
            new_content += f"{indent}{boda_module_name}._boda_profile_start()\n"
        elif command == "stop":
            new_content += f"{indent}{boda_module_name}._boda_profile_stop()\n"
        elif command == "event":
            new_content += f"{indent}{boda_module_name}._boda_profile_event()\n"

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


def parse_arguments():

    parser = argparse.ArgumentParser(description="boda_lite.py")
    parser.add_argument("--boda-tmpdir", type=str, help="Temporary working directory")
    parser.add_argument("--boda-outdir", type=str, default=".", help="Output directory")
    parser.add_argument("target_script", type=str, help="Python script to run")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="Args for target script")

    args = parser.parse_args()

    if args.boda_tmpdir is None:
        args.boda_tmpdir = tempfile.mkdtemp(prefix="bodalite_")

    args.boda_outdir = os.path.abspath(args.boda_outdir)

    if not os.path.isdir(args.boda_outdir):
        os.mkdir(args.boda_outdir)

    print(f"TMPDIR: {args.boda_tmpdir}")
    print(f"OUTDIR: {args.boda_outdir}")

    return args

def create_boda_module(tmpdir):

    path = Path(tmpdir) / f"{boda_module_name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        f.write(boda_module)

    return path


def instrument_code(args, modpath):

    script_path = modify_script(Path(args.target_script).resolve(), args.boda_tmpdir)
    sys.path.insert(0, str(script_path.parent))
    sys.meta_path.insert(0, TrackImportsFinder(args.boda_tmpdir))

    return script_path

def instrument(args):

    # create boda module
    boda_module_path = create_boda_module(args.boda_tmpdir)

    # instrument
    script_path = instrument_code(args, boda_module_path)

    return script_path

def collect_events():
    pass

def generate_report():
    pass

def analyze(args):

    collect_events()

    generate_report()


def main():

    # parse command-line arguments
    args = parse_arguments()

    # instrument script
    script_path = instrument(args)

    # run instrumented code
    try:
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

        #print("\nUsed Python files:")
        #for orig, mod in used_modules.items():
        #    if orig != mod:
        #        print(f"{orig} -> MODIFIED")
        #    else:
        #        print(f"{orig}")
        ## Optionally, remove tmp_dir after use
        #shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()

