import os
import torch
from torch.multiprocessing import Process


def dist_init(rank, num_procs, run_func, *func_args, **func_kwargs):
    """Initialize torch.distributed and execute the user function."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8081"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_procs)
    os.environ.pop("NCCL_DEBUG", None)

    init_method = 'tcp://' 
    init_method +=  os.environ["MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]


    torch.distributed.init_process_group(
        backend="nccl",
        world_size=num_procs,
        rank=rank,
        init_method=init_method)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    run_func(*func_args, **func_kwargs)

    # make sure all ranks finish at the same time
    torch.distributed.barrier()
    # tear down after test completes
    torch.distributed.destroy_process_group()

def dist_launcher(num_procs, run_func, *func_args, **func_kwargs):
    """Launch processes and gracefully handle failures."""

    # Spawn all workers on subprocesses.
    processes = []
    for local_rank in range(num_procs):
        p = Process(target=dist_init,
                    args=(local_rank, num_procs, run_func, *func_args),
                    kwargs=func_kwargs)
        p.start()
        processes.append(p)

    # Now loop and wait for a test to complete. The spin-wait here isn't a big
    # deal because the number of processes will be O(#GPUs) << O(#CPUs).
    any_done = False
    while not any_done:
        for p in processes:
            if not p.is_alive():
                any_done = True
                break

    # Wait for all other processes to complete
    for p in processes:
        p.join(200)

    failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
    for rank, p in failed:
        # If it still hasn't terminated, kill it because it hung.
        if p.exitcode is None:
            p.terminate()
            print(f"Worker {rank} hung.")
        if p.exitcode < 0:
            print(f"Worker {rank} killed by signal {-p.exitcode}")
        if p.exitcode > 0:
            print(f"Worker {rank} exited with code {p.exitcode}")
