#!/usr/bin/env python3

import argparse
from contextlib import ExitStack
import fcntl
import itertools
import os
import pathlib
import subprocess
import sys
import time

from xml.etree import ElementTree as XMLTree

import psutil

RESERVATION_DIR = pathlib.Path("/tmp/cuda-reservations")
NUM_GPUS = 8


def get_gpu_memory_usage():
    def _unpack_memory_entry(xml_item):
        text = xml_item.text
        unit_str = " MiB"
        assert text.endswith(unit_str)
        text = text.strip(unit_str)
        memory_mib = int(text)
        memory_bytes = memory_mib * (1024**2)
        return memory_bytes

    xml_text = subprocess.check_output(
        ["nvidia-smi", "--xml-format", "--query"], encoding="utf-8"
    )
    xml_tree = XMLTree.fromstring(xml_text)

    output = {}
    for gpu in xml_tree.iterfind("gpu"):
        gpu_id = int(gpu.find("minor_number").text)
        framebuffer_memory = gpu.find("fb_memory_usage")
        total = _unpack_memory_entry(framebuffer_memory.find("total"))
        reserved = _unpack_memory_entry(framebuffer_memory.find("reserved"))
        used = _unpack_memory_entry(framebuffer_memory.find("used"))
        free = _unpack_memory_entry(framebuffer_memory.find("free"))

        memory_usage = {
            key: _unpack_memory_entry(framebuffer_memory.find(key))
            for key in ["total", "reserved", "used", "free"]
        }
        output[gpu_id] = memory_usage

    return output


class LockFile:
    _flags = fcntl.LOCK_EX

    def __init__(self, filepath):
        self.filepath = filepath

    def __enter__(self):
        self.filepath.parent.mkdir(exist_ok=True, parents=True)
        self._file = self.filepath.open("a")
        self._file.__enter__()
        fcntl.flock(self._file, self._flags)

    def __exit__(self, *args, **kwargs):
        if self._file is not None:
            self._file.__exit__(*args, **kwargs)
            self._file = None


class AttemptedLockFile(LockFile):
    _flags = fcntl.LOCK_EX | fcntl.LOCK_NB

    def __init__(self, filepath):
        self.filepath = filepath

    def __enter__(self):
        try:
            super().__enter__()
            self.acquired = True
        except BlockingIOError:
            self.acquired = False


class UMask:
    def __init__(self, umask):
        self.umask = umask
        self._saved_umask = None

    def __enter__(self):
        self._saved_umask = os.umask(self.umask)

    def __exit__(self, *args):
        os.umask(self._saved_umask)
        self._saved_umask = None


class Reservations:
    def __init__(self, reservation_dir):
        self.reservation_dir = pathlib.Path(reservation_dir)

    @property
    def lock_file(self):
        with UMask(0):
            return LockFile(self.reservation_dir / ".lock")

    def gpu_lock_file(self, gpu_num):
        assert gpu_num < NUM_GPUS
        with UMask(0):
            return AttemptedLockFile(self.reservation_dir / f"gpu-{gpu_num}")

    def run(self, cmd, num_gpus_required=None, specific_gpus_required=None):
        if num_gpus_required is None and specific_gpus_required is None:
            num_gpus_required = 1
            gpu_pool = list(range(NUM_GPUS))
        elif num_gpus_required is None:
            gpu_pool = specific_gpus_required
            num_gpus_required = len(specific_gpus_required)
        elif specific_gpus_required is None:
            gpu_pool = list(range(NUM_GPUS))
        elif num_gpus_required is not None and specific_gpus_required is not None:
            raise ValueError(
                "Must specify either --num-gpus or --specific-gpus, but not both."
            )

        gpus_acquired = set()

        with ExitStack() as stack:
            # May only claim GPUs while the lock file is exclusively
            # held.  After claiming GPUs, release the lock file so
            # other processes can claim any remaining GPUs.
            with UMask(0), self.lock_file:
                for i, gpu_num in enumerate(itertools.cycle(gpu_pool)):
                    if i == len(gpu_pool):
                        print(
                            f"Only able to acquire {len(gpus_acquired)} out of {num_gpus_required} required, waiting."
                        )

                    if gpu_num in gpus_acquired:
                        continue

                    if i >= len(gpu_pool):
                        time.sleep(1.0)

                    context = self.gpu_lock_file(gpu_num)
                    context.__enter__()

                    if context.acquired:
                        memory_usage = get_gpu_memory_usage()[gpu_num]
                        if memory_usage["used"] < (1024**3):
                            gpus_acquired.add(gpu_num)
                            stack.push(context)
                            context = None
                        else:
                            print(
                                f"Reserved GPU {gpu_num}, but something else is using it without a reservation.  "
                                f"Moving on from GPU {gpu_num}, to avoid conflicting usage, "
                                f"but the unreserved usage should be investigated.",
                                file=sys.stderr,
                            )

                    if context is not None:
                        context.__exit__()

                    if len(gpus_acquired) >= num_gpus_required:
                        break

            # With the GPU-specific lock held, execute the command
            environ = os.environ.copy()
            environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(gpu) for gpu in sorted(gpus_acquired)
            )
            subprocess.check_call(cmd, env=environ)


def main(args):
    reservations = Reservations(RESERVATION_DIR)
    reservations.run(args.cmd, args.num_gpus, args.specific_gpus)


def arg_main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="The number of GPUs to reserve for this command.  (default = 1)",
    )
    parser.add_argument(
        "--specific-gpus",
        type=str,
        default=None,
        help="A comma-separated list of GPUs to be reserved.",
    )
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Start a pdb post mortem on uncaught exception",
    )
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="The command to run with the reservation",
    )

    args = parser.parse_args()
    if args.specific_gpus:
        args.specific_gpus = [int(num) for num in args.specific_gpus.split(",")]

    try:
        main(args)
    except Exception:
        if args.pdb:
            import pdb, traceback

            traceback.print_exc()
            pdb.post_mortem()
        raise


if __name__ == "__main__":
    arg_main()
