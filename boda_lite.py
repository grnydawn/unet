
import sys
import runpy
import argparse
import importlib
import importlib.util
from importlib import abc
import traceback
from pathlib import Path
import tempfile
import shutil

boda_module = '''\
import torch
'''
def modify_script(original_path, temp_dir) -> str:

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
        # TODO: handle two common name files
        temp_file_path = Path(temp_dir) / original_path.name
        with open(temp_file_path, 'w') as f:
            f.writelines(modified_lines)
        return temp_file_path

    return original_path

class TrackImportsFinder(abc.MetaPathFinder):

    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        self.used_modules = {}

    def find_spec(self, fullname, path, target=None):
        try:
            spec = importlib.util.find_spec(fullname)
            if spec and spec.origin and spec.origin.endswith(".py"):
                original_path = Path(spec.origin).resolve()

                if original_path in self.used_modules:
                    return None  # Already processed

                new_path = modify_script(original_path, self.temp_dir)
                if new_path == original_path:
                    self.used_modules[original_path] = original_path
                else:
                    self.used_modules[original_path] = new_path
                    spec = importlib.util.spec_from_file_location(fullname, new_path)
                    print(f"{original_path} -> MODIFIED({new_path})")

                return spec

            return None
        except Exception:
            return None


def parse_arguments():

    parser = argparse.ArgumentParser(description="boda_lite.py")
    parser.add_argument("--boda-tempdir", type=str, help="Temporary directory for boda")
    parser.add_argument("target_script", type=str, help="Python script to run")
    parser.add_argument("target_args", nargs=argparse.REMAINDER, help="Args for target script")

    args = parser.parse_args()

    if args.boda_tempdir is None:
        args.boda_tempdir = tempfile.mkdtemp(prefix="bodalite_")

    return args

def create_boda_module(tempdir):

    path = Path(tempdir) / "bodalitemodule.py"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        f.write(boda_module)

    return path


def instrument_code(args, modpath):

    script_path = modify_script(Path(args.target_script).resolve(), args.boda_tempdir)
    sys.path.insert(0, str(script_path.parent))
    sys.meta_path.insert(0, TrackImportsFinder(args.boda_tempdir))

    return script_path

def instrument(args):

    # create boda module
    boda_module_path = create_boda_module(args.boda_tempdir)

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
        ## Optionally, remove temp_dir after use
        #shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

