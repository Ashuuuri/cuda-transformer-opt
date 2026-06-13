# Optimization Log — cuda-transformer-opt (INT8 track)

Chronological iteration journey for the INT8 attention / MLP kernels on
the `opt-dev` branch. **This is the detailed history (before/after ncu
numbers, full validation traces, negative results).** The distilled,
still-actionable conclusions — what is exhausted and must not be
re-attempted — live in `CLAUDE.md` §3–§4; this file is the evidence
behind them. Append a new `### Iteration N` here after each optimization
(format in `CLAUDE.md` §7).

## Iteration Log

### Iteration 1 - 2026-06-12
- **Change**: `kernels/int8_attention.cu` — templated the WMMA kernel on
  `(TILE_KV, HEAD_DIM)`; reordered QK^T and P@V loops (k-outer, fragment
  hoisting). Two negative results along the way: cp.async double-buffering
  (0.58-0.82x) and V register prefetch (0.75-0.9x), both kept as ablation
  flags.
- **Target metric**: local-memory traffic (runtime head_dim spilled
  frag_out to local memory), smem fragment-load count
- **Profiling results**: 1.6-2.2x speedup across the sweep grid; vs FA2
  0.30x → 0.62x
- **Accuracy validation** (via test_int8.py + sweep error columns; the
  five-gate framework was built afterwards):
  - [x] Math metrics (max err ~1e-3, unchanged)
  - [x] Edge cases (6 configs PASS)
- **Conclusion**: pass. Commit `9473d42`. Next: MLP dynamic quantization.

### Iteration 2 - 2026-06-12
- **Change**: `kernels/int8_mlp.cu` — dynamic quantization: GEMM1 epilogue
  gathers per-token row absmax → per-token hidden requant → GEMM2 per-row
  scales → dynamic per-tensor output scale. Static path kept as
  `INT8_MLP_DYNAMIC=0`.
- **Target metric**: accuracy (max/mean err); `launch__registers_per_thread`
  (epilogue statistics pushed regs 128 → 166, halving occupancy; capped
  back with `__launch_bounds__(256,2)`)
- **Profiling results**: latency +1-6% (worst +14%); accuracy at
  d_model=2048: max err 0.49 → 0.107 (4.6x), mean err 9.2x better
- **Accuracy validation**:
  - [x] Math metrics
  - [x] Edge cases
- **Conclusion**: pass. Commit `0db4387`. Next: `mma.sync` PTX path
  (attention).

### Iteration 3 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` — fused the REGPV softmax-weight
  pack into the P@V MMA loop (group-outer/slice-inner), dropping the
  `p_frag[n_kv_groups][4]` array for a per-group `pf[4]`; the VPREFETCH
  barrier moved ahead of the fused loop.
- **Target metric**: `sm__pipe_tensor_cycles_active` (and MUFU stall
  hiding) — interleaving the exp/pack with ldmatrix+mma across KV groups
  overlaps transcendental latency with the tensor pipe instead of
  serializing the two phases.
- **Profiling results**: latency vs previous baseline: int8_attn +1.5%,
  int8_mlp -0.2%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.957 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass. Next: `mma.sync` m16n8k32 with P kept in registers
  (attention).

### Iteration 4 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` — replaced the QK^T INT8 WMMA
  m16n16k16 path (`load_matrix_sync` + `frag_qk` fragments) with native
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (new helper
  `attn_mma_m16n8k32_s8`); Q/K operands now loaded as plain
  head-dim-contiguous 4-byte smem reads into an `int32_t qk[][8]` array, two
  n8 calls per logical n16 group filling x[0..3]/x[4..7] so the
  cast/softmax/P@V code is unchanged.
- **Target metric**: `sm__pipe_tensor_cycles_active` up and
  `l1tex__data_pipe_lsu_wavefronts_mem_shared` down — k=32 native
  instructions are denser on the tensor pipe than WMMA m16n16k16.s8 (which
  decomposes into multiple ops), and dropping `load_matrix_sync` removes the
  QK^T smem fragment round-trips that dominate the shared-memory wavefront
  count (priority #2/#3); registers unchanged (no spills, 2 blocks/SM
  preserved).
- **Profiling results**: latency vs previous baseline: int8_attn -1.4%,
  int8_mlp -0.2%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.957 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass.


### Iteration 5 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` (REGPV default path) — folded V's
  per-token scale `sV` out of the pre-barrier V-dequant loop into the
  softmax-weight pack: V is now stored in smem as a raw INT8→FP16 cast, `sV`
  is staged in a new `s_scale_V` smem array and multiplied into the packed
  `mma.m16n8k16` A-weights (`pf`), while `lsum` (softmax denominator) stays
  on the unscaled weights; `calc_smem` and the `INT8_ATTN_VPREFETCH` cast
  branch updated to match. Math is reassociation-identical (V_i8 is exact in
  FP16).
- **Target metric**: `smsp__warp_issue_stalled_barrier_per_warp_active.pct`
  (3.76% at head_dim=256, the highest stall) — the per-element `×sV` is
  removed from the V-dequant work that gates the load `__syncthreads`, and
  since the weights are reused across all `n_slices` head-dim slices this is
  strictly fewer multiplies than scaling every V element for
  head_dim ≥ TILE_Q, shortening the pre-barrier critical path.
- **Profiling results**: latency vs previous baseline: int8_attn -0.9%,
  int8_mlp -0.2%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.956 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass.


### Iteration 6 - 2026-06-13
- **Change**: `kernels/int8_attention.cu` (REGPV P@V loop) — software-pipelined
  the attn×V ldmatrix: hoisted slice 0's V-fragment ldmatrix before the slice
  loop and prefetch slice s+1's fragments into a second register set while
  slice s's two `mma.m16n8k16` issue (shared A-weights `pf` reused), instead of
  each slice's MMA waiting on its own ldmatrix; registers/occupancy unchanged
  (164/216/236, no spills), output bit-identical.
- **Target metric**: `sm__pipe_tensor_cycles_active` (tensor-pipe utilization,
  ~13–17%, the cuBLAS gap) — overlapping the ldmatrix smem→reg latency with
  tensor-core MMA issue removes the per-slice load→MMA stall (confirmed by
  ptxas hoisting LDSM ahead of HMMA in the SASS), feeding the tensor pipe more
  densely without touching the smem-wavefront or register/occupancy
  bottlenecks.
- **Profiling results**: latency vs previous baseline: int8_attn +0.0%,
  int8_mlp +0.3%
- **Accuracy validation** (all five gates passed; gate-1 numbers below):
  - [x] Gate 1: math metrics
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level
  ```
  Gate1 math     : cos=0.99995 top10=0.990 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(1.0000), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99966 top10=0.964 outlier=0.0000 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=0.99  -> PASS
                   worst-3: out(0.9997), h_requant(0.9998), h_gelu(0.9999)  -> PASS
  Gate1 math     : cos=0.99684 top10=0.956 outlier=0.0522 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9968), softmax(0.9979), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.97143 top10=0.563 outlier=0.3394 nan_ok=True  -> FAIL
  Gate2 stability: x_dequant=1.00 h_pre_gelu=0.99 h_gelu=0.99 h_requant=0.99 out=1.00  -> PASS
                   worst-3: x_dequant(0.9683), h_pre_gelu(0.9686), out(0.9714)  -> PASS
  Gate1 math     : cos=0.99987 top10=0.803 outlier=0.0302 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), softmax(0.9999), scores(1.0000)  -> PASS
  Gate1 math     : cos=0.99989 top10=0.974 outlier=0.0001 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9999), h_requant(1.0000), h_gelu(1.0000)  -> PASS
  Gate1 math     : cos=0.99745 top10=0.968 outlier=0.0685 nan_ok=True  -> FAIL
  Gate2 stability: qkv_dequant=1.00 scores=1.00 softmax=1.00 out=1.00  -> PASS
                   worst-3: out(0.9975), softmax(0.9977), scores(0.9999)  -> PASS
  Gate1 math     : cos=0.99934 top10=0.952 outlier=0.0002 nan_ok=True  -> PASS
  Gate2 stability: x_dequant=1.00 h_pre_gelu=1.00 h_gelu=1.00 h_requant=1.00 out=1.00  -> PASS
                   worst-3: out(0.9993), h_requant(0.9995), h_gelu(0.9996)  -> PASS
  Gate1 math     : cos=0.99997 top10=0.991 outlier=0.0000 nan_ok=True  -> PASS
  ```
- **Conclusion**: pass.


### Iteration 7 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` — replaced the WMMA `load_matrix_sync` +
  `mma_sync(m16n16k16)` inner loop of BOTH GEMM kernels with hand-rolled native
  `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (new `mlp_mma_m16n8k32_s8`,
  shared `mlp_int8_mainloop`), operands loaded as plain 4-byte k-contiguous smem
  words like the attention QK^T path. The B operand (weights) is staged
  k-contiguous, which int8 `mma.sync` requires, by **pre-transposing W1/W2 to
  `[N][K]`** each forward (`transpose_int8_kernel`, ~1-2% bandwidth pass — no
  pointer-keyed cache, to stay correct across validation datasets that reuse
  freed weight addresses). s32 accumulators are scattered back to the row-major
  c_smem tile (`mlp_store_acc16`) so both epilogues are unchanged. B smem layout
  `[k][n] stride 144` -> `[n][k] stride 48`; smem total unchanged (epilogue
  dominated, 32 KB -> still 2 blocks/SM).
- **Target metric**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
  and `l1tex__throughput` down — the hand-rolled load's per-instruction bank map
  `(12*group + tid_grp) mod 32` is a bijection over the 32 lanes (conflict-free),
  unlike `load_matrix_sync`'s int8 pattern which collided period-8 (2-way) and
  made ~46% of smem-load wavefronts conflicts. (XOR swizzle proved unnecessary —
  the access-pattern change alone fixes it while keeping the 16-byte `cp.async`
  stores; the iter-6 README 4-byte-store probe was the negative result that ruled
  out the stride route.)
- **Profiling results** (graded shape b=8 s=512 d_model=1024 d_ff=4096):
  - bank conflicts (ld.sum): **16.78M -> ~3.0K / 1.2K (≈0, -99.98%)**
  - `l1tex__throughput`: **78%/67% -> 39%/29%** (L1/smem no longer the ceiling)
  - smem-load wavefronts: **36.2M -> 10.0M / 8.1M**
  - `tensor_op_imma` active: 26%/31% (≈unchanged)
  - registers 128, 0 spills, 2 blocks/SM (unchanged)
  - **latency: -26% to -40% across the full 12-point sweep**, reproducible across
    two runs (d512 s512 0.42->0.31; d1024 s512 0.89->0.64, s4096 5.55->3.50;
    d2048 s4096 20.18->12.09). Speedup vs naive 0.41-0.48x -> 0.55-0.79x.
- **Accuracy validation** (all five gates PASS on all 8 datasets + task-level):
  - [x] Gate 1: math metrics (normal cos=1.00000)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (ppl fp16=504.81 int8=504.87 +0.011%, top1 100%)
- **Conclusion**: pass — largest single-iteration MLP gain so far. Next: push
  `tensor_op_imma` (still ~28%) toward the cuBLAS 60%+ ceiling now that the
  smem-pipe bottleneck is gone — a deeper k-stage pipeline or denser-mma schedule.


### Iteration 8 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` — raised the cp.async K-stage depth
  `INT8_MLP_STAGE_K` 32 → 64 (one-line default; the `mlp_int8_mainloop`
  `for kk in 0..STAGE_K step 32` loop already handles it, now 2 `mma.sync
  m16n8k32` per acc per cp.async round). smem strides grow `A/B_SMEM_STRIDE`
  48 → 80; stride 80 stays conflict-free (`stride/4 = 20 = 4·5`, gcd(5,8)=1 →
  the 8 ×4-multiple banks are all distinct, bijection over 32 lanes) and 16B-
  aligned (80 = 5·16), so iter-7's conflict-free property is preserved.
- **Target metric**: `smsp__warp_issue_stalled_long_scoreboard_per_warp_active`
  down and `sm__pipe_tensor_op_imma_cycles_active` up — after iter 7 cleared the
  smem-pipe ceiling, `long_scoreboard` (global-load latency, DRAM only 5–6% → pure
  latency) became a top stall; a deeper k-stage carries 2× data per cp.async round
  → half as many global-load sync points → less exposed latency + longer back-to-
  back MMA runs.
- **Profiling results** (graded shape b=8 s=512 d_model=1024 d_ff=4096):
  - `long_scoreboard`: **18.7%/27.4% → 5.5%/7.7%** (GEMM1/GEMM2)
  - `tensor_op_imma` active: **26.1%/30.9% → 29.7%/38.2%**
  - `wait` (now the top stall): 19.5%/24.3% → 21.5%/28.2%
  - `l1tex__throughput`: 39%/29% → 42%/34%; bank conflicts still ≈0 (8.4K/1.6K)
  - registers 128 (launch_bounds-capped; small 48/52 B spill), 2 blocks/SM (held);
    smem 32 → 40 KB, no occupancy loss (SM smem budget ≥ 80 KB)
  - **latency: −6% to −17% across the full 12-point sweep**, reproducible across
    two runs (d1024 s1024 1.03→0.89; s4096 3.50→3.02; d2048 s2048 6.20→5.18,
    s4096 12.09→10.04). Speedup vs naive 0.55–0.79× → 0.59–0.97×.
- **Accuracy validation** (all five gates PASS on all 8 datasets + task-level;
  numerics bit-identical to iter 7 — STAGE_K is pure tiling, same int32
  accumulation order):
  - [x] Gate 1: math metrics (normal mlp cos=0.99966)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (ppl fp16=504.81 int8=504.87 +0.011%, top1 100%)
- **Conclusion**: pass — second MLP win in a row. The two prior MLP ceilings
  (smem bank conflicts, global-load latency) are now both exhausted; the new
  bottleneck is the `wait` MMA-dependency stall (`tensor_op_imma` 30%/38% vs
  cuBLAS 60%+) at structurally-fixed 2-block occupancy. Next: a denser-mma
  schedule that overlaps more independent IMMA chains.

### Iteration 9 - 2026-06-13  (ACCURACY track, not perf)
- **Change**: realistic INT8 MLP quantization + realistic validation, in two
  coordinated parts (owner relaxed the no-interface-change rule for this):
  - **Phase 2 — per-channel/per-token MLP quant.** `kernels/int8_mlp.cu`:
    templated `gemm_int8_wmma_f16_kernel<apply_gelu, a_scale_per_row,
    b_scale_per_col>` so the epilogue applies a per-ROW activation scale and a
    per-COLUMN weight scale (gathered into `s_arow`/`s_bcol` smem); new entry
    point `int8_mlp_forward_per_channel` does **per-token activation +
    per-channel weight + per-token OUTPUT** quant (output now uses
    `quantize_rows_kernel` + GEMM2 per-row absmax, not `quantize_tensor_kernel`),
    returning the [T] per-row output scales. `int8_ext.cu`: tensor-scale
    `int8_mlp_forward` overload (pybind dispatch by arg type) returning a
    per-token `out_scale` tensor. The per-tensor scalar `int8_mlp_forward`
    (device sig + `tests/cuda/` .bin flow + sweep/profile/test_int8) is
    UNCHANGED.
  - **Phase 1 — real Gate 5.** `validate_int8.py`: Gate 5 rewritten to real
    GPT-2 perplexity on real WikiText-2 (`testdata/real_corpus.txt`) via the
    kernel-faithful per-channel/per-token sim; `run_mlp_kernel`/`sim_mlp` updated
    for per-token output dequant. `generate_test_data.py`: removed the now-XPASS
    `outlier` MLP xfail.
- **Target metric**: real-model accuracy (Gate 5 perplexity), NOT a perf metric
  — this is the accuracy track. The graded per-tensor path (sweep.py) is
  structurally untouched; sweep latency unchanged vs iteration 8 (within noise).
- **Key finding**: per-tensor MLP *output* quant gives **+64.4%** GPT-2
  perplexity (crushes output channel outliers); per-channel weights alone with
  per-tensor output is still +64%. Per-token output quant is the fix → −0.07%.
  The old random-weight Gate 5 masked all of this (falsely +0.011%).
- **Profiling / accuracy results**:
  - Gate 5: fp16 ppl=31.9462 → int8 ppl=31.9246 (**−0.068%**, well under 2%).
  - `outlier`/`boundary`/`stress` MLP now PASS outright (outlier was the one
    substantive XFAIL; cos 0.971 → >0.999).
- **Accuracy validation** (all 5 gates PASS on all 8 datasets + real Gate 5):
  - [x] Gate 1: math metrics (normal cos=1.00000)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (real GPT-2, −0.068%)
- **Conclusion**: pass — the INT8 MLP is now accurate on a real model, validated
  by a realistic gate. Perf targets (tensor_op_imma toward cuBLAS) unchanged and
  still live.

### Iteration 10 - 2026-06-13  (NEGATIVE RESULT — reverted, do not re-attempt)
- **Change (reverted)**: `kernels/int8_mlp.cu` `mlp_int8_mainloop` — the
  documented "denser-mma schedule" lever. Restructured the inner mma issue:
  hoisted BOTH row tiles' A fragments per kk into `a[WARP_ROW_TILES][4]`, made
  the col-tile loop outer so each B (weight) fragment loads **once** and is
  reused across row tiles (previously loaded inside the rt loop → looked 2×
  redundant), with the 4 mma per col tile (2 rt × 2 n8-halves) issuing
  back-to-back into 4 distinct accumulators.
- **Target metric**: `l1tex__data_pipe_lsu_wavefronts_mem_shared` down (halve
  B-LDS) and `tensor_op_imma` up / `wait` down (denser back-to-back IMMA).
- **Profiling results** (graded shape b=8 s=512, median of 5; full 12-pt sweep):
  - **latency: +1.5% to +3.0% across ALL 12 sweep shapes** — a systematic
    REGRESSION, not noise.
  - **GEMM2 metrics byte-identical**: `tensor_op_imma` 38.25→38.24%, `wait`
    28.17→28.18%. → ptxas was **already CSE-ing the "redundant" B loads** across
    the unrolled rt loop; there were no shared-loads to remove. The reorder
    bought nothing.
  - **GEMM1 got worse**: `tensor_op_imma` 30.18→28.90%, `wait` 18.96→21.62%
    (the hoisted A fragments' larger live-register footprint traded
    `long_scoreboard` 11.90→5.41% for more `wait`). regs 128 (held), still
    2 blocks/SM.
- **Accuracy validation**: all 5 gates PASS bit-identically (pure reorder, same
  int32 accumulation order; Gate 5 real GPT-2 −0.068% unchanged) — but accuracy
  is irrelevant here since latency regressed.
- **Conclusion**: NEGATIVE — reverted (a regression with no metric improved is
  not kept even behind a flag). **Two lessons, do not re-attempt:** (1) the
  warp-level mma issue order is **already at ptxas's optimum** — GEMM2 is
  byte-identical under reorder, so intra-warp rescheduling has no headroom; the
  "redundant B load" was a phantom (compiler CSE'd it). (2) The `wait` stall
  (top stall, ~28% GEMM2 = mma-result-dependency latency) is **structural**: at
  the fixed 2-block/SM occupancy the only way to hide it is more independent
  accumulator chains, but the `acc[2][4][8]` tile is already 64 regs and any
  more spills. **This is the same wall the occupancy lever hit — the MLP perf
  well is now dry for both the occupancy AND the denser-mma levers.** The ONLY
  remaining perf lever in the whole project is the attention online-softmax
  dependency chain (§4 #2) — deep and high-risk.

### Iteration 11 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` + `int8_ext.cu` — added a **prepacked
  (transpose-once) MLP entry point** for static-weight inference, WITHOUT
  touching the (exhausted) GEMM internals. The all-in-one `int8_mlp_forward`
  re-transposes W1/W2 to `[N][K]` (k-contiguous, required by the m16n8k32 B
  fragment) on **every** forward; weights are constant across forwards, so this
  is pure per-call overhead. New: `transpose_int8_weights(W1,W2)->(W1T,W2T)`
  (the same one-shot pass, callable once at load) + `int8_mlp_forward_prepacked`
  (per-tensor scales, takes already-transposed weights, `weights_prepacked=true`
  skips the internal transpose). Both old device signatures (`int8_mlp_forward`,
  `int8_mlp_forward_per_channel`) and the `tests/cuda/` .bin flow are UNCHANGED;
  this is a NEW path per the relaxed-interface rule. `sweep.py` now transposes
  once outside the timed loop and grades the prepacked path (realistic static-
  weight inference); `collect_profile.py` profiles both so the dashboard shows
  the transpose removed.
- **Target metric**: per-forward transpose share of MLP forward time
  (torch.profiler kernel-time breakdown) → 0, and end-to-end MLP forward latency.
  This is a forward-orchestration win, NOT a GEMM-internal ncu-metric move (the
  GEMM kernels are byte-identical — output bit-identical, verified).
- **Profiling results** (graded grid, median of 5 runs; bench_prepacked.py):
  - torch.profiler MLP forward split (b=8 s=512 d_model=1024): all-in-one =
    79.0% GEMM + **10.4% transpose W** + 7.4% quantize_rows + 3.3% other;
    prepacked = **88.3% GEMM + 0% transpose** + 8.0% quantize_rows + 3.7% other.
  - end-to-end latency vs all-in-one, transpose-once: **−22.1% (d512 s512),
    −9.3% (d1024 s512), −11.9% (d2048 s512)** at the graded s=512; the win
    shrinks with seq_len as the GEMM dwarfs the fixed transpose (−1.5% to −3%
    at s=4096). All deltas reproducible (median of 5); s≤2048 all beyond the
    2% noise floor.
  - GEMM absolute time unchanged (24.69k→24.60k µs, within noise); output
    `torch.equal` bit-identical to `int8_mlp_forward` on all 12 sweep shapes.
- **Accuracy validation** (all 5 gates PASS on all 8 datasets + real Gate 5;
  the prepacked path is bit-identical to the validated per-tensor path, and the
  per-channel/per-tensor entry points it shares are untouched):
  - [x] Gate 1: math metrics (normal cos=1.00000)
  - [x] Gate 2: numeric stability
  - [x] Gate 3: stage error trace
  - [x] Gate 4: edge cases
  - [x] Gate 5: task-level (real GPT-2, −0.068%)
- **Conclusion**: pass — a real end-to-end MLP-forward win (largest at the
  graded s=512: ~9–22%) found OUTSIDE the perf-exhausted GEMM, by amortizing the
  static-weight transpose. The GEMM inner loop remains dry; the lesson is that
  the forward orchestration around it was not. Next: still the attention
  online-softmax dependency chain (§4 #2) for kernel-internal perf.

### Iteration 12 - 2026-06-13  (NEGATIVE RESULT — kept behind OFF flag `INT8_MLP_FINE_WARP`, do not re-attempt)
- **Change (flagged OFF)**: `kernels/int8_mlp.cu` tile defines — finer **4×4 warp
  tiling**. `WARP_COL_TILES` 4→2 so each warp owns a 2×2 sub-grid of 16×16 tiles
  (`acc[2][2][8]` = 32 s32 regs, was `acc[2][4][8]` = 64), making the 128×128
  block a 16-warp / 512-thread grid (was 8-warp / 256-thread). `launch_bounds`
  follows `THREADS_PER_BLOCK` → `(512,2)`. The smaller acc was the whole point:
  it frees the registers that iters 8/10 proved you cannot free at 256 threads,
  so occupancy can finally rise. Default path (flag off) is byte-identical to
  iter 11 (256 thr, 128 regs, `acc[2][4][8]`).
- **Target metric**: `sm__warps_active` up (break the 2-block/16-warp ceiling)
  and the IMMA-result `wait` stall down → `tensor_op_imma` toward cuBLAS 60%+.
- **Profiling results** (graded b=8 s=512 d_model=1024 d_ff=4096; ncu launch-2):
  - `warps_active`: **23.8/21.5% → 47.6/41.9%** (GEMM1/GEMM2) — occupancy DID
    double, exactly as designed (2 blocks × 512 thr = 32 warps/SM, regs=64).
  - `wait` (IMMA-result dep, the top GEMM2 stall): **19.0/28.2% → 13.7/14.5%** —
    **halved.** Confirms the hypothesis: more IMMA-issuing warps DO hide `wait`.
  - **But `tensor_op_imma` NET FELL: 30.2/38.3% → 24.6/29.8%**, and **latency
    regressed +15% to +28% across the full 12-pt sweep** (graded d1024/s512
    0.51→0.60ms; d2048/s4096 9.79→12.49ms). Why: halving the acc forced
    `WARP_COL_TILES` 4→2, which **halves A/B-fragment reuse** — `short_scoreboard`
    (smem LDS) rose (GEMM2 0.66%→4.36%, GEMM1 3.76%→6.93%), per-warp arithmetic
    intensity dropped, and a small 44–76 B spill appeared. The finer tile's lost
    operand reuse more than cancels the occupancy/`wait` win.
- **Accuracy validation** (default path, all 5 gates PASS, bit-identical to iter
  11; flag-on path numerically correct via sweep's inline check):
  - [x] Gate 1 (normal cos=1.00000)  [x] Gate 2  [x] Gate 3  [x] Gate 4
  - [x] Gate 5 (real GPT-2, −0.068%)
- **Conclusion**: NEGATIVE — reverted to OFF-by-default flag. **This closes the
  occupancy lever from the OPPOSITE direction of iters 8/10.** Iters 8/10 showed
  you *cannot* raise occupancy at 256 threads (the 64-reg acc blocks 3 blocks/SM).
  Iter 12 shows you *can* raise it — to 47% warps_active, genuinely halving the
  `wait` stall — by going to a finer tile, but the operand reuse you trade away to
  shrink the acc costs more `tensor_op_imma` than the extra warps buy back. So the
  `wait` stall is real and warp-count *does* relieve it, yet there is no
  register-budget path that raises occupancy *without* surrendering the reuse that
  feeds the IMMA pipe — the two are coupled through the acc-tile register cost.
  **The MLP GEMM occupancy lever is now exhausted from both sides; do not
  re-attempt finer warp tiling.** The only remaining kernel-internal perf lever in
  the project stays the attention online-softmax dependency chain (§4 #2).

### Iteration 13 - 2026-06-13
- **Change**: `kernels/int8_mlp.cu` — fuse the standalone `quantize_rows` pass
  (FP16 hidden → INT8) into GEMM2's staged smem load. New kernel
  `gemm_int8_wmma_f16_a16fused_kernel` reads the FP16 hidden directly as its A
  operand and quantizes each per-token row to INT8 *in shared memory* during the
  cp.async staging (using the per-row absmax GEMM1 already wrote to
  `s_row_absmax`), then runs the identical m16n8k32 main loop + FP16 epilogue.
  Eliminates one kernel launch and the int8-hidden HBM write+read round-trip.
  Uses dynamic smem (61440 B: FP16 staging double-buffer pushes past the 48 KB
  static limit; `cudaFuncSetAttribute` opts into the larger carveout, 2 blocks/SM
  preserved). The existing int8-input path (`mlp_int8_mainloop` +
  `gemm_int8_wmma_f16_kernel`) is left byte-identical; the fused path is gated
  behind `INT8_MLP_FUSE_QUANT`. Motivation: `collect_profile.py` showed
  `quantize_rows` is the largest remaining non-GEMM slice of the graded forward
  (8.0% of the prepacked path; the iter-11 transpose-once already removed the
  10.4% transpose), and the GEMMs themselves are CLOSED (§4 #1).
- **Target metric**: forward wall-clock (remove the 8% `quantize_rows` kernel +
  its ~48 MB HBM round-trip on the hidden); torch.profiler per-kernel split.
- **Profiling results**:
  - ptxas: fused kernel 128 regs (launch_bounds cap held → 2 blocks/SM), small
    48–60 B spill, 61440 B dynamic smem. No spill/occupancy red flags.
  - sweep (median of the 12-pt grid), flag ON vs the iter-11 baseline:
    **REGRESSED +45% to +63%** — d_model=1024/s=512 0.51→0.74 ms (+45%),
    d_model=1024/s=4096 2.96→4.33 ms (+46%), d_model=2048/s=4096
    9.79→15.94 ms (+63%). Regression scales with size (i.e. with GEMM2 K-stage
    count), consistent with per-stage overhead, not a fixed cost.
  - default path (flag OFF) re-swept: bit-restored to baseline (d1024/s512
    0.51 ms, d2048/s4096 9.78 ms) — confirms the `#else` path is byte-identical.
- **Accuracy validation** (flag ON, all 5 gates PASS — the math is correct, the
  fused output is numerically identical to the unfused path):
  - [x] Gate 1 (normal cos=1.00000)  [x] Gate 2  [x] Gate 3  [x] Gate 4
  - [x] Gate 5 (real GPT-2, −0.068%, unchanged)
- **Conclusion**: NEGATIVE — reverted to OFF-by-default flag. **Cause:** GEMM2 is
  wait-bound (IMMA-result dependency, §4 #1). The on-load conversion adds a serial
  `convert_fp16A_to_int8 + __syncthreads` per K-stage *between* the cp.async-wait
  and the mma — it is NOT overlapped with compute (the convert writes the same
  int8 buffer the mma then reads) — plus the FP16 staging doubles A's HBM load
  bytes. Over the 64–128 K-stages of GEMM2 that serial per-stage overhead more
  than doubles the kernel, swamping the 8% the removed `quantize_rows` saved.
  **Rules out** fusing the hidden requant into GEMM2's load path: the separate
  memory-bound `quantize_rows` (running at ~70% HBM bandwidth as its own kernel)
  is *cheaper* than paying conversion latency inside the latency-bound GEMM. The
  lever isn't dead in principle — decoupling the convert from the mma critical
  path (triple-buffer, or convert stage s+1 *during* the mma of stage s) could
  hide it — but that is a deep restructure for at most an 8% ceiling, lower
  priority than the attention softmax chain (§4 #2). Kept behind
  `INT8_MLP_FUSE_QUANT` (OFF) so the experiment + the "GEMM2 can't absorb on-load
  convert" finding are preserved. **The MLP forward orchestration is now also
  exhausted** (transpose amortized iter 11; quantize_rows fusion negative iter 13).

### Iteration 14 - 2026-06-13  (NEW REGIME — decode / INT8 KV-cache attention)
- **Change**: new kernel `kernels/int8_decode_attention.cu` (+ pybind entry
  `int8_decode_attention_forward` in `int8_ext.cu`) for the **decode** regime
  (seq_q == 1), which the square WMMA kernel cannot serve (it tiles Q into 16-row
  WMMA fragments — 15/16 padding when seq_q==1). Decode is a GEMV (Q·Kᵀ) +
  weighted sum (P·V), **bandwidth-bound on streaming the KV cache**, which is
  exactly where an INT8 KV cache pays off (K,V at 1 byte vs FP16's 2 → half the
  bytes streamed per token). Two structural sub-versions, both this iteration:
  - **v1 (one warp per (b,h))**: each of the 32 lanes owns DPL=HEAD_DIM/32 head
    dims, streams the whole KV cache, int8×int8 partial dot + warp-shuffle reduce
    + online softmax, barrier-free / no smem. Correct but **lost to FP16 SDPA
    (0.04–0.90×, ~34% HBM BW)**: decode's parallelism is only batch*heads, so few
    warps are resident and memory latency is not hidden (grid-starved at small
    b*h; serial single-warp KV scan at large S).
  - **v2 (split-KV / flash-decoding, the shipped path)**: also split the KV
    dimension — NSPLIT warps cooperate per (b,h), each scanning one KV chunk into
    a PARTIAL online-softmax state (m, l, unnormalized Σexp(score−m)·V); a second
    `int8_decode_combine_kernel` merges the NSPLIT partials per (b,h) with the
    standard log-sum-exp rescale. Multiplies resident warps by NSPLIT → fills the
    SMs → saturates BW and cures grid starvation. NSPLIT chosen by the host
    (MIN_CHUNK=128, TARGET_UNITS=4096, MAX_SPLIT=128). Partials live in a cached
    `cudaMalloc` scratch (grows; no per-step malloc). HEAD_DIM ∈ {64,128}.
- **Target metric**: decode-step latency vs FP16 SDPA, and HBM bandwidth
  (ours GB/s vs the A100-SXM4 ~1555 GB/s peak). Benchmarked by
  `bench_attn_decode.py` (CONFIGS grid × SEQS 1024/2048/4096).
- **Profiling results** (mean ms / decode step over 50 iters; `bench_attn_decode.py`):
  - **HEAD_DIM=128 (B=64 H=16): ours 1.23× FASTER than FP16 SDPA across all seq
    lens** (S=1024 0.354 vs 0.436 ms; S=2048 0.669 vs 0.824; S=4096 1.300 vs
    1.602), **757–826 GB/s ≈ 52% of HBM peak** — the INT8-KV-cache latency win,
    at half the KV bytes.
  - **HEAD_DIM=64: still loses (0.46–0.94×)**, plateaus at **~390 GB/s ≈ 25% of
    peak** — *instruction-bound, not bandwidth-bound*: DPL=2 (only 2 bytes/lane
    per KV step) so the fixed per-step overhead (6-step `__shfl` reduction +
    2× `__expf` + broadcast) dominates the tiny load. More splits cannot fix an
    instruction-bound loop; cuBLAS/cuDNN's tuned GEMV handles small-D better.
  - v2 vs v1: substantial improvement (v1 maxed 0.90× / 531 GB/s; v2 best
    1.23× / 826 GB/s; small-batch B=8 H=8 went 0.04×→0.57×).
  - KV-cache footprint (structural, all shapes): INT8 is **half** the FP16 bytes
    (e.g. B=128 H=16 S=4096 D=64: 1074 MB vs 2147 MB) — more context / bigger
    batch per card, half the bytes streamed per token.
- **Accuracy validation**: decode kernel cos vs fp32 SDPA ground truth
  **0.99994–0.99995** across every shape (incl. ragged S). The 5-gate
  `validate_int8.py` (prefill MLP + attention paths) still **ALL PASS** after the
  build change — Gate 5 real GPT-2 −0.068% unchanged — confirming the new source
  + the `int8_ext.cu` binding did not perturb the existing kernels. (The decode
  kernel is a separate entry point; the square `int8_attention_forward`,
  `int8_mlp_forward`, and the `tests/cuda/` .bin flow are untouched.)
- **Conclusion**: PASS / SHIPPED for HEAD_DIM=128 (a real "beats FP16 SDPA"
  decode result, 1.23×, the common real-LLM head size). HEAD_DIM=64 is left as
  documented instruction-bound future work — the fix is a different lane layout
  (**one lane per KV position**: each lane does a full 64-dim dot via `dp4a`, no
  per-j shuffle reduction → one reduction per 32 positions instead of per
  position), a separate coherent iteration. v1 (one-warp-per-(b,h)) is **not**
  kept as a flag — it was strictly superseded by v2 within the same iteration, so
  only the split-KV path ships. This iteration opens the **decode regime** as the
  place the INT8 KV-cache byte advantage is realized (the prefill/square kernel's
  value is compute, not the cache); see `CLAUDE.md` §4 "What winning means".

### Iteration 15 - 2026-06-13  (decode dp4a lane-per-position — WIN @ D=64)
- **Change**: `kernels/int8_decode_attention.cu` — new partial kernel
  `int8_decode_partial_dp4a_kernel<HEAD_DIM>` that flips the warp's intra-lane
  division. The original `int8_decode_partial_kernel` gives each lane DPL=HEAD_DIM/32
  *dims* → it needs a 6-step `__shfl` score reduction **per KV position** (the
  fixed per-step overhead that left D=64 instruction-bound at ~25% HBM BW, iter
  14). The dp4a variant gives each lane a whole KV **position** per 32-position
  tile: lane L scores position (t+L) with a FULL HEAD_DIM dot via `__dp4a`
  (HEAD_DIM/4 ops, NO per-position reduction), the warp reduces only **twice per
  tile** (softmax max + sum via `__shfl_xor`). P·V flips back to dims-layout (lane
  owns DPL acc dims, V read coalesced) and broadcasts each position's `p·sV` with a
  single `__shfl`. Q stays resident as HEAD_DIM/4 int32 words. Barrier-free, no
  smem, unchanged combine kernel. Selected per head_dim by the host (see below);
  `INT8_DECODE_DP4A` (default 1) forces the original D=64 path off for A/B.
- **Target metric**: decode-step latency / HBM BW at HEAD_DIM=64 (the iter-14
  instruction-bound case); `bench_attn_decode.py`.
- **Profiling results** (mean ms / decode step, vs FP16 SDPA; dp4a vs iter-14):
  - **HEAD_DIM=64 — WIN.** B≥32 (serving scale) went **0.61–0.94× → 1.24–1.56×**
    (e.g. B=32 S=4096 0.74×→1.56×; B=64 S=2048 0.67×→1.35×; B=128 S=4096
    0.64×→1.26×). **HBM BW ~390 → up to 786 GB/s (≈2×, ~50% of peak)** — the
    hypothesis held exactly: removing the per-position reduction moved D=64 from
    instruction-bound back to bandwidth-bound. Tiny B=8 improved 0.46–0.66× →
    0.80–1.04× (still grid-starved at batch×heads=64 on 108 SMs — a batch-size
    limit, not a kernel one).
  - **HEAD_DIM=128 — dp4a REGRESSES (0.86–0.93×).** NW=32 int32 Q words held in
    registers cut occupancy; the original lane-per-dim split-KV (DPL=4 already
    amortizes the reduction well) keeps its 1.23× win. So the host **dispatches the
    empirically-best partial per head_dim**: D=64 → dp4a, D=128 → original. The
    dp4a-128 path is left compiled-out of the default dispatch (a documented
    negative; reachable for ablation but not shipped).
- **Accuracy validation**: decode cos vs fp32 SDPA **0.99994–0.99995** across all
  shapes (unchanged). 5-gate `validate_int8.py` **ALL PASS** (Gate 5 real GPT-2
  −0.068%, unchanged) — dp4a is an isolated new decode kernel; the prefill MLP /
  attention paths and the `tests/cuda/` .bin flow are untouched.
- **Conclusion**: PASS / SHIPPED for HEAD_DIM=64. Decode attention now **beats
  FP16 SDPA at every serving-scale shape (B≥32) for both head dims** (D=64
  1.24–1.56× via dp4a, D=128 1.23–1.24× via split-KV), at half the KV-cache bytes.
  This closes the iter-14 "D=64 instruction-bound" future-work item. **dp4a@D=128
  is a do-not-retry** (reg-pressure regression). Remaining decode future work is
  only the small-batch (B=8) grid-starvation, which is a launch-shape limit (more
  NSPLIT helps marginally; fundamentally batch×heads is just small). The
  prefill/square attention kernel remains perf-exhausted (only the online-softmax
  dependency chain, §4 #2, is left there).
