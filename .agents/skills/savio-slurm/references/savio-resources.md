# Savio resource reference

Check these official Berkeley Research IT pages before choosing hardware:

- [Submitting jobs](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/submitting-jobs/): required account, partition, time, and optional QoS.
- [Available account/partition/QoS combinations](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/what-resources/): use `sacctmgr` rather than guessing access.
- [Specifying resources](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/specific-resources/): per-core versus per-node scheduling, memory, GPU requests, and constraints.
- [Current hardware](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/hardware-config/): core counts, RAM, GPU models, VRAM, and CPU:GPU ratios.
- [Scheduler configuration](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-config/): current partition allocation policies and QoS limits.
- [Job-script examples](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/running-your-jobs/scheduler-examples/): threaded, MPI, GPU, low-priority, array, and long-job patterns.

## Current routing rules

These are a starting point, not a substitute for checking the live pages:

- `savio4_htc`: shared CPU nodes, 56 cores on the main node class, 4 GB per requested core by default.
- `savio4_htc` plus `--constraint=savio4_m512`: 512 GB nodes and 8 GB per requested core.
- `savio3` and older standard/big-memory partitions: generally allocated per node; use the full node when possible.
- `savio4_gpu`: A5000 (24 GB VRAM, 4 CPU cores per GPU) or L40 (46 GB, 8 cores per GPU). Current regular FCA availability differs by model.
- `savio3_gpu`: heterogeneous V100, 2080 Ti, TITAN RTX, and A40 nodes. Always specify the model and its applicable QoS.
- Avoid explicit memory flags in ordinary Savio jobs. Resource requests and account type determine when exceptions are appropriate.

For this repository, historical scripts demonstrate `fc_nitrates`, `savio4_htc`, `savio3_gpu`, `a40_gpu3_normal`, and an A40 request with eight CPU cores. Confirm access and current suitability before reuse.
