# S100P BPU: unpooled vs. ApertureVLA, same toolchain, same board

- Toolchain: `registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0`
  (network-isolated Docker), HBDK 4.7.5, `march nash-m`, single BPU core.
- Graphs: `scripts/export_instruction_onnx.py --sample-index 0` (LIBERO-10 task 7 instruction baked in;
  inputs `camera_views [2,3,256,256]`, `normalized_state [1,8]`).
  - Dense: `outputs/checkpoints/dense_mixed40k_v2/model_40000.pth`, ONNX vs PyTorch max abs 7.5e-7.
  - ApertureVLA: `outputs/checkpoints/early_compression_split4_pool2_detail_10k_v1/evaluation_ema.pth`,
    ONNX vs PyTorch max abs 4.5e-6.
- Compile: `hb_compile --fast-perf --march nash-m -m policy.onnx --input-shape camera_views 2x3x256x256
  --input-shape normalized_state 1x8` (fast-perf uses random calibration data: latency only, accuracy not
  validated by this build).
- Board: RDK S100P, system 4.0.3-Beta, UCP 3.7.3. The 3.7.0-compiled HBMs load and run on this board.
  `hrt_model_exec perf --model_file <m> --thread_num 1 --frame_count 500` (logs `perf_*.log`).

| Model | HBM SHA-256 (prefix) | Static estimate | Board average | Board max | FPS |
|---|---|---:|---:|---:|---:|
| Dense 216M | 1ce4b140 | 8.04 ms | 8.715 ms | 9.047 ms | 114.5 |
| ApertureVLA | cd95f16a | 5.33 ms | **5.638 ms** | 5.971 ms | **176.6** |

Pooling reduces board latency by 35% (1.55x); on the RTX 4090 with CUDA graphs it is 14% (2.63 to 2.25 ms).
Static DDR traffic per run: dense 222 MB, ApertureVLA 155 MB.
