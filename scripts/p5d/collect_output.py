import os, itertools, sys

RESULTS = "/path/to/results_dir"

HEADER = "| seqlen  | genlen  | p50: sec  | p50: tok/s | p90: sec  | p90: tok/s | p99: sec  | p99: tok/s |"

def readfile(basename, dtype, sharding, inlen=None, outlen=None):
    if inlen and outlen:
        fname = f"{basename}_{dtype}_TP_{sharding}_context_{inlen}_{outlen}"
    else:
        fname = f"{basename}_{dtype}_TP_{sharding}"
    statfile = os.path.join(RESULTS, fname)
    if not os.path.exists(statfile):
        print(f"File {statfile} doesn't exist; skipping.")
        return
        
    with open(statfile, 'r') as f:
        text = f.read()
    i = text.find(HEADER)
    if i < 0:
        print(f"Error reading data from file {statfile}")
        return
    text = text[i+len(HEADER)+1:]
    res = text.split('\n')[0]
    res = res.replace("|", ",")
    res = f"{dtype},{sharding}" + res
    print(res)

if __name__ == "__main__":
    
    dtypes = ("fp16","fp8","int8","int4")
    shards = (1,2,4,8)
    contexts = (128,2364)

    if len(sys.argv) < 2:
        raise IndexError("Please supply base file name; e.g., 'output' or 'benchmark_output'")

    for d,s,i,o in itertools.product(dtypes, shards, contexts, contexts):
        readfile(sys.argv[1], d, s, i, o)
    
