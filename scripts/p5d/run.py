# Started working on this to drive the C++ benchmark and get results of the same format as run.py. While working on it, some tests came back showing multi-gpu was about the same (~1% diff)
#  between the two, so I'm just going to stick with run.py.
# The command here to run the c++ benchmark should be correct, just need to launch and parse output, then compute and print stats. Also need to modify the C++ source to print stats for each
#  run, not just the average at the end.

import os
import numpy as np
from subprocess import Popen, PIPE

TRTLLM_HOME = os.environ["TRTLLM_HOME"]
BENCHMARK_EXE = os.path.join(TRTLLM_HOME, "benchmarks", "gptSessionBenchmark")
MODEL_STR = "llama"

ENGINES = os.environ["ENGINES"]
MODELNAME = os.environ["MODELNAME"]
MAX_INPUT = os.environ["MAX_INPUT"]
MAX_OUTPUT = os.environ["MAX_OUTPUT"]
MAX_BATCH = os.environ["MAX_BATCH"]

def benchmark(
        dtype,
        sharding,
        num_warmups=5,
        num_measurements=20,
        batch_size=1,
        input_length=2000,
        output_length=128,
        num_warmup=5,
        num_measurements=20
):

    # /opt/bin/cuda-reserve.py --num-gpus 2 mpirun -n 2 --allow-run-as-root ./benchmarks/gptSessionBenchmark --model "llama" --engine_dir "../../engines/CodeLlama-34b-instruct-hf_fp16_context_2560_2560_batch_1_TP_2/" --batch_size "1" --input_output_len "60,20" --warm_up 5 --num_runs 20

    engine_dir = f"{ENGINES}/{MODELNAME}_{dtype}_context_{MAX_INPUT}_{MAX_OUTPU}_batch_{MAX_BATCH}_TP_{sharding}"
    if sharding == 1:
        proc = Popen(["/opt/bin/cuda-reserve.py", "--num-gpus", sharding, BENCHMARK_EXE, "--model", MODEL_STR, "--engine_dir", engine_dir, "--batch_size", batch_size, "--input_output_len", f"{input_length},{output_length}", "--warm_up", num_warmup, "--num_runs", num_measurements], stdout=PIPE, stderr=PIPE)
    else:
        proc = Popen(["/opt/bin/cuda-reserve.py", "--num-gpus", sharding, "mpirun", "-n", sharding, "--allow-run-as-root", BENCHMARK_EXE, "--model", MODEL_STR, "--engine_dir", engine_dir, "--batch_size", batch_size, "--input_output_len", f"{input_length},{output_length}", "--warm_up", num_warmup, "--num_runs", num_measurements], stdout=PIPE, stderr=PIPE)
    


percentiles = [50, 90, 99]
latencies = np.percentile(self.times, percentiles)
throughputs = [self.output_tokens / latency for latency in latencies]

header = "| {:7} | {:7} |".format("seqlen", "genlen")
for p in percentiles:
    header += " {:9} | {:9} |".format(f"p{p}: sec", f"p{p}: tok/s")
print(header)
    
results = "| {:7} | {:7} |".format(self.input_tokens, self.output_tokens)
for l, t in zip(latencies, throughputs):
    results += " {:9.3f} | {:9.3f} |".format(l, t)
    
print(results)
