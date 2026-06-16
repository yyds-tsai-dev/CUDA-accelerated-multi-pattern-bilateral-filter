# 架構導向雙邊濾波加速實驗整理

本報告整理 `CA_Final_Project.pdf` 中需要回覆的實驗分析題，並對照目前
`P1` 到 `P5` 的實作與量測結果。整體結論是：正確性結果合理，CUDA
SIMT 與 multi-pattern GPU 的趨勢符合預期；RVV 的 P2 reduction 版本在本
實作中沒有加速，原因是 reduction 本身的額外資料搬移與暫存成本超過收益，
報告中應明確說明。

## PDF 題目索引

完整掃描 `docs/CA_Final_Project.pdf` 後，真正需要回答的分析題集中在下列頁面：

| PDF page | Part | 需要回答的重點 |
| --- | --- | --- |
| p.17 | Part 2 - RVV Vector Reduction | cycle / instruction 是否如預期下降，並與 P1 baseline 比較。 |
| p.20 | Part 3 - SIMD-like RVV Parallelization | SIMD-like RVV 的 cycle / instruction 是否如預期下降。 |
| p.21 | Part 3 - SIMD-like RVV Parallelization | cycle reduction 與 instruction reduction 為何不同、`k` 是否越大越好、P3 為何不需要 reduction operations。 |
| p.31 | Part 4 - CUDA SIMT Implementation | 從 PTX / ncu 判斷程式偏 memory-intensive 或 compute-intensive。 |
| p.33 | Part 4 - CUDA SIMT Implementation | shared memory、block size、occupancy、latency hiding 的影響。 |
| p.35 | Part 5 - Multi-pattern GPU Parallelism | 從 PTX / ncu 判斷 multi-pattern 程式偏 memory-intensive 或 compute-intensive。 |
| p.36 | Part 5 - Multi-pattern GPU Parallelism | 為何 P5 更適合 GPU、為何需要大量 threads/warps、pattern 數量增加後效能與 occupancy 如何變化。 |

p.5、p.6、p.22、p.23、p.24 也有問號標題，例如 gem5、CUDA、PTXAS、ncu
能提供什麼資訊，但內容屬於課程背景說明，不是本次實驗結果分析題。p.34
是 Part 5 的 2D grid mapping 說明，不是 CUDA SIMT 的 Think 頁面。

## 實驗環境

- 課程 gem5/RVV Docker image: `weisheng505/gem5-rvv-image:v1`
- 實際 gem5 執行方式: Docker 因 `unshare: operation not permitted` 無法跑 container，所以使用本機 `/home/u5977862/gem5/build/RISCV/gem5.opt`
- gem5 模式: `TimingSimpleCPU`，classic caches，`l1d_size=32kB`，`l1i_size=32kB`，`cacheline_size=64`
- gem5 RVV: `VLEN = 256 bits`，`ELEN = 64 bits`
- CUDA 環境: host CUDA installation
- CUDA: `nvcc` build `cuda_12.9.r12.9/compiler.35813241_0`
- GPU: 2 x Tesla V100-SXM2-32GB
- Nsight Compute: 2025.2.0.0
- V100 architecture flag: `sm_70`

使用的主要命令：

```sh
./scripts/check_environment.sh
./scripts/collect_gem5_results.sh
./scripts/collect_cuda_results.sh
make -C P4 ptx
make -C P5 ptx
ncu --set basic ./P4/main 1024 1024 3 16 16 1 5
ncu --set basic ./P5/main 1024 1024 16 256 5
```

## 程式結構

| Path | 說明 |
| --- | --- |
| `common/bilateral_common.h` | 共同的 deterministic input 產生、scalar reference、checksum、`max_abs_diff`、PGM 輸出。 |
| `common/result_io.h` | 將每次實驗結果 append 到 `results/*.csv` 的工具。 |
| `P1/main.cpp` | scalar CPU baseline。 |
| `P2/main.cpp` | RVV reduction，將單一 output pixel 的 filter window terms 放進 vector 做 reduction。 |
| `P3/main.cpp` | SIMD-like RVV，將多個 output pixels 分配到 vector lanes。 |
| `P4/main.cu` | CUDA SIMT，包含 global-memory naive kernel 與 shared-memory kernel。 |
| `P5/main.cu` | Multi-pattern CUDA，用 `grid.y` 對應 pattern index，同時處理多個獨立 input patterns。 |
| `scripts/collect_gem5_results.sh` | 收集 P1-P3 host smoke、RISC-V build、gem5 stats。 |
| `scripts/collect_cuda_results.sh` | 收集 P4/P5 CUDA event timing 與 CSV。 |
| `results/` | CSV、gem5 stats、ncu、PTXAS log 等實驗證據。 |

## 演算法

本專題使用 arithmetic bilateral-style filter。每個 output pixel 由鄰近 window
的 weighted sum 產生：

```text
spatial = 1 / (1 + (dx*dx + dy*dy) / sigma_s2)
range   = 1 / (1 + (center - neighbor)^2 / sigma_r2)
weight  = spatial * range
num    += neighbor * weight
den    += weight
out     = num / den
```

這個形式保留 bilateral filter 的核心資料流：spatial weight、range weight、
per-pixel weighted summation，同時避免在 gem5/RVV 上使用成本較高且不一定
穩定的 `exp()`。

## 實作方法

### P1 Scalar Baseline

P1 是純 scalar C++ baseline。它對每個 pixel 依序掃過 `(2r+1)^2` 的鄰域，
計算 numerator 與 denominator，最後輸出 `num / den`。後續 P2 到 P5 都用
P1 的輸出作為 correctness reference。

### P2 RVV Vector Reduction

P2 的概念是讓 vector lanes 代表單一 output pixel 的 filter window terms。
程式先把每個 neighbor 的 `neighbor * weight` 與 `weight` 分別存進暫存
array，再用 RVV reduction 指令加總。這符合「vector reduction」的題目要求，
但本實作會增加額外的 array 寫入、讀取與 reduction overhead。

### P3 SIMD-like RVV Parallelization

P3 不把 lanes 用在同一個 pixel 的 reduction，而是讓 lanes 對應多個 output
pixels。對同一個 `(dx, dy)`，多個 lanes 同時計算不同 pixel 的 neighbor、
range weight、numerator、denominator，最後直接做 vector divide 與 strided
store。這就是 SIMD-like parallelization。

### P4 CUDA SIMT Implementation

P4 將 output pixels 映射到 CUDA threads。`naive/global` kernel 直接從 global
memory 讀取每個 pixel 的 window；`shared` kernel 先把 block tile 加上 halo
載入 shared memory，再由 block 內 threads 重複使用 tile 內資料。kernel
runtime 用 CUDA events 量測，H2D / D2H 另行列印，不混進 kernel timing。

### P5 Multi-pattern GPU Parallelism

P5 將多個 input patterns 合併為一個 batch。CUDA grid 的 `x` dimension 對應
單張影像中的 output index，`y` dimension 對應 pattern index，也就是 PDF p.34
要求的 2D grid mapping。這讓 GPU 一次看到更多 independent work，能更有效
填滿 SM、增加 active warps，改善 latency hiding。

## 實驗結果

### P1-P3 Host Smoke

這些 host timing 只用來確認程式可以在 native host 上跑通，不應與 gem5
simSeconds 或 CUDA event timing 混合比較。

| Part | Size | k | Checksum | max_abs_diff | chrono ms |
| --- | --- | --- | --- | --- | --- |
| P1 scalar | 32x32 | N/A | 6238116.209287 | N/A | 0.176672 |
| P1 scalar | 64x64 | N/A | 25461956.465892 | N/A | 2.181829 |
| P2 RVV reduction host fallback | 32x32 | N/A | 6238116.209287 | 0.000000 | 0.274514 |
| P2 RVV reduction host fallback | 64x64 | N/A | 25461956.465892 | 0.000000 | 1.121965 |
| P3 SIMD-like host fallback | 32x32 | 2 | 6238116.209287 | 0.000000 | 0.546490 |
| P3 SIMD-like host fallback | 32x32 | 4 | 6238116.209287 | 0.000000 | 0.178255 |
| P3 SIMD-like host fallback | 32x32 | 8 | 6238116.209287 | 0.000000 | 0.246257 |
| P3 SIMD-like host fallback | 64x64 | 4 | 25461956.465892 | 0.000000 | 2.202352 |

### P1-P3 gem5

| Part | gem5 options | Checksum | max_abs_diff | simSeconds | simInsts | cycles | CPI | D-cache miss rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 scalar | `32 32 3` | 6238116.195794 | N/A | 0.005621 | 3231709 | 11241402 | 3.478470 | 0.009360 |
| P1 scalar | `64 64 3` | 25461956.426892 | N/A | 0.016185 | 9648134 | 32369022 | 3.354952 | 0.004001 |
| P2 RVV reduction | `32 32` | 6238115.943961 | 0.000122 | 0.007834 | 4972814 | 15668938 | 3.150920 | 0.007982 |
| P2 RVV reduction | `64 64` | 25461955.495364 | 0.000122 | 0.025063 | 16594926 | 50125770 | 3.020548 | 0.003477 |
| P3 SIMD-like RVV | `32 32 2` | 6238116.195794 | 0.000000 | 0.005850 | 3440167 | 11699858 | 3.400956 | 0.014585 |
| P3 SIMD-like RVV | `32 32 4` | 6238116.195794 | 0.000000 | 0.005770 | 3338247 | 11540228 | 3.456972 | 0.014675 |
| P3 SIMD-like RVV | `32 32 8` | 6238116.195794 | 0.000000 | 0.005682 | 3262607 | 11364402 | 3.483227 | 0.014732 |
| P3 SIMD-like RVV | `64 64 4` | 25461956.426892 | 0.000000 | 0.015334 | 9160653 | 30668808 | 3.347884 | 0.010826 |

### P4 CUDA

| Width | Height | Radius | Block | Mode | Repeats | Checksum | max_abs_diff | avg_kernel_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 512 | 3 | 8x8 | naive/global | 5 | 1637512415.620656 | 0.000061 | 0.107213 |
| 512 | 512 | 3 | 8x8 | shared | 5 | 1637512415.620656 | 0.000061 | 0.103456 |
| 512 | 512 | 3 | 16x16 | naive/global | 5 | 1637512415.620656 | 0.000061 | 0.107546 |
| 512 | 512 | 3 | 16x16 | shared | 5 | 1637512415.620656 | 0.000061 | 0.103776 |
| 1024 | 1024 | 5 | 16x16 | naive/global | 5 | 6550815804.927626 | 0.000076 | 0.706176 |
| 1024 | 1024 | 5 | 16x16 | shared | 5 | 6550815804.927626 | 0.000076 | 0.654355 |
| 1024 | 1024 | 5 | 32x8 | naive/global | 5 | 6550815804.927626 | 0.000076 | 0.704563 |
| 1024 | 1024 | 5 | 32x8 | shared | 5 | 6550815804.927626 | 0.000076 | 0.656627 |

### P5 CUDA

| Width | Height | Radius | Patterns | Threads/block | Repeats | Checksum | max_abs_diff | avg_kernel_ms | avg_ms_per_pattern | total_kernel_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1024 | 3 | 1 | 256 | 5 | 6550831709.315681 | 0.000061 | 0.315187 | 0.315187 | 1.575936 |
| 1024 | 1024 | 3 | 2 | 256 | 5 | 13101845648.158327 | 0.000092 | 0.606144 | 0.303072 | 3.030720 |
| 1024 | 1024 | 3 | 4 | 256 | 5 | 26203780400.305031 | 0.000092 | 1.189171 | 0.297293 | 5.945856 |
| 1024 | 1024 | 3 | 8 | 256 | 5 | 52407367705.564903 | 0.000092 | 2.357427 | 0.294678 | 11.787136 |
| 1024 | 1024 | 3 | 16 | 256 | 5 | 104815167996.359741 | 0.000092 | 4.695072 | 0.293442 | 23.475361 |

### Nsight Compute 摘要

| Target | Theoretical occupancy | Achieved occupancy | Achieved active warps/SM | 主要觀察 |
| --- | --- | --- | --- | --- |
| P4 shared, 1024x1024, radius 3, 16x16 | 75% | about 70.4% | about 45.1 | 37 registers/thread，occupancy 主要受 register count 限制。 |
| P5 patterns=16, 1024x1024, radius 3 | 50% | about 48.5% | about 31.0 | 50 registers/thread，occupancy 同樣受 register count 限制。 |

P4 與 P5 的 ncu 結果都顯示 Compute (SM) Throughput 約 83-85%，Memory
Throughput 約 7-15%，DRAM Throughput 約 3%。因此目前 kernel 的瓶頸比較偏
compute-heavy，而不是 DRAM bandwidth-heavy。

## PDF 題目回答

### p.17 Part 2: RVV reduction 與 P1 baseline 比較

P2 的結果不符合「一定會下降」的直覺預期。以 32x32 來看，P1 的
simSeconds 是 `0.005621`，P2 是 `0.007834`，約為 P1 的 `1.39x`；simInsts
從 `3,231,709` 增加到 `4,972,814`，約 `1.54x`；cycles 從 `11,241,402`
增加到 `15,668,938`，約 `1.39x`。64x64 也類似，P2 simSeconds 約為 P1 的
`1.55x`，simInsts 約 `1.72x`。

原因是 P2 每個 output pixel 需要先建立 numerator/denominator term arrays，
再對兩個 arrays 做 vector reduction。雖然 reduction 指令本身可以平行加總，
但額外的暫存 array 寫入、讀取、vector setup、loop control 會增加指令數與
memory traffic。對目前 radius 3 的小 window 與 TimingSimpleCPU/gem5 設定來說，
這些 overhead 超過 reduction 的收益。因此 P2 的 correctness 合理，
但 performance 不應描述成加速。

### p.20-p.21 Part 3: SIMD-like RVV 結果與 Think 回答

P3 的趨勢比 P2 更接近預期。32x32 下，`k=2/4/8` 的 simSeconds 分別是
`0.005850`、`0.005770`、`0.005682`，隨著 `k` 增加有小幅改善。64x64 `k=4`
時，P3 simSeconds 是 `0.015334`，比 P1 的 `0.016185` 快約 `5.3%`，
simInsts 與 cycles 也約下降 `5%`。

cycle reduction 與 instruction reduction 不完全相同，因為 cycles 受到 CPI、
cache miss、memory latency、branch behavior、pipeline stalls 影響。即使指令數下降，
如果有較多 strided load/store 或 cache miss，cycles 不一定同比例下降；反過來，
如果 CPI 改善，cycles 也可能比 instruction reduction 更明顯。

`k` 不是越大一定越好。較大的 `k` 可以一次處理更多 output pixels，降低 loop overhead，
但也會增加 vector register pressure、strided memory access 成本與 tail/boundary
handling。當 `k` 超過硬體有效 vector length 或造成 cache/register 壓力時，
效能可能停滯甚至變差。本次 32x32 的 `k=8` 最快，但改善幅度很小，不能推論無限制增加
`k` 會持續加速。

P3 不需要 reduction operations，因為每個 lane 對應的是不同 output pixel。
每個 lane 自己累加該 pixel 的 numerator/denominator，最後做 vector divide。
P2 則是讓 lanes 對應同一個 pixel 的不同 window terms，所以必須把 lanes
reduce 成一個 scalar。P3 的好處是避免 reduction overhead，且平行處理多個 pixels；
代價是需要 strided load/store，邊界區域仍可能 fallback 到 scalar，資料連續性也不如
單純 unit-stride vector load。

### p.31 Part 4: CUDA SIMT 是 memory-intensive 還是 compute-intensive

P4 從 ncu 來看比較偏 compute-intensive。P4 shared kernel 的 Compute (SM)
Throughput 約 `83-84%`，Memory Throughput 約 `14-15%`，DRAM Throughput 約
`2.7%`。PTXAS 顯示 shared kernel 使用 `37` registers/thread，naive kernel 使用
`47` registers/thread，沒有 spill stores 或 spill loads。這表示目前主要壓力不是
global memory bandwidth，而是每個 output pixel 內大量 weight arithmetic、除法與累加。

Shared memory 仍然有幫助，因為它降低相鄰 threads 對同一個 halo/window 資料的重複
global loads；但因為 kernel 已偏 compute-heavy，所以改善幅度是小到中等，而不是數倍。

### p.33 Part 4: shared memory、block size、occupancy、latency hiding

Shared memory 在 block 內 threads 會重複讀取鄰近資料時比較有效。Bilateral filter
每個 output pixel 都讀取周圍 window，相鄰 pixels 的 window 高度重疊，所以把 tile
與 halo 放入 shared memory 可以減少 global memory 重複讀取。本次 P4 shared mode
在 512x512 radius 3 約快 `3.5%`，在 1024x1024 radius 5 約快 `6.8-7.3%`。
radius 越大、重用越多，shared memory 的收益越明顯。不過 shared memory 也會增加
tile loading、`__syncthreads()` 與 shared-memory capacity 成本，所以不是所有情況都會大幅加速。

Block size 不是越大越好。較大的 block 可以增加每個 block 的 threads，也可能改善
memory coalescing 或減少 block 管理成本；但同時會受 registers/thread、shared memory
per block、maximum threads/block、warp scheduling 限制。當 block 太大，SM 上能同時
resident 的 blocks 可能變少，occupancy 反而下降。

Block size 會影響同時執行的 threads/warps 數量與 occupancy。Occupancy 是 active
warps 相對於硬體上限的比例；它受 block size、register count、shared memory usage
共同限制。本次 P4 shared kernel theoretical occupancy 是 `75%`，achieved occupancy
約 `70.4%`，active warps/SM 約 `45.1`，主要受 register count 限制。較高 occupancy
通常能讓 GPU 在某些 warps 等待 memory 或 long-latency instruction 時切換到其他
warps 執行，提升 latency hiding。不過 occupancy 不是越高一定越快；若 kernel 已經偏
compute-heavy，瓶頸可能在 arithmetic pipeline，而非單純缺 warps。

### p.35 Part 5: Multi-pattern CUDA 是 memory-intensive 還是 compute-intensive

P5 也偏 compute-intensive。ncu 顯示 P5 patterns=16 的 Compute (SM) Throughput
約 `85%`，Memory Throughput 約 `7.6%`，DRAM Throughput 約 `2.9%`。PTXAS 顯示
P5 kernel 使用 `50` registers/thread，沒有 spill stores 或 spill loads。雖然 P5
會讀寫更多 input/output images，但每個 pixel 仍要做完整 window arithmetic，因此主要
限制仍是 SM compute throughput 與 register pressure，而不是 DRAM bandwidth。

### p.36 Part 5: 為何 P5 更適合 GPU，以及 pattern 數量的影響

P5 比 P4 更適合 GPU，是因為它把多個 independent input patterns 一次交給 GPU。
P4 只處理單一 pattern，可用 parallelism 主要來自一張圖的 pixels；P5 額外增加
pattern dimension，讓 grid 有更多 blocks 和 warps。這種工作彼此獨立，幾乎不需要
跨 pattern synchronization，很適合 GPU 的 SIMT 執行模型。

GPU 需要大量 threads/warps 才能發揮 scalability，因為單一 warp 遇到 memory latency、
instruction latency 或 pipeline dependency 時，scheduler 可以切換到其他 ready warps。
如果工作量太少，SM 無法維持足夠 active warps，latency hiding 不足，硬體利用率會下降。

增加 pattern 數量後，總 kernel time 會隨總工作量增加，但 per-pattern time 有小幅改善。
本次 P5 從 1 pattern 到 16 patterns，`avg_kernel_ms` 從 `0.315187 ms` 增加到
`4.695072 ms`，接近隨工作量增加；但 `avg_ms_per_pattern` 從 `0.315187 ms` 降到
`0.293442 ms`，約改善 `6.9%`。這表示更多 patterns 提供更多 parallel work，
讓 GPU utilization 和 latency hiding 更好，但不會線性免費加速，因為每個 pattern
仍有固定的 per-pixel arithmetic work，且 register pressure 將 occupancy 限制在約 `50%`
theoretical、`48.5%` achieved。

## 結果合理性判斷

正確性方面，P2/P3 對 scalar reference 的 `max_abs_diff` 為 `0.000000` 或
`0.000122`，P4/P5 對 reference 的 `max_abs_diff` 約 `0.000061-0.000092`。
這些差異屬於 floating-point 運算順序不同造成的小誤差，合理且可接受。

效能方面，P2 沒有加速是合理的負面結果，因為 reduction implementation 的 overhead
很高。P3 只有小幅加速也合理，因為小尺寸測試、strided memory access、邊界 fallback
都會壓低 SIMD-like 的收益。P4 shared memory 小幅加速符合 data reuse 預期，且 radius
較大時改善較明顯。P5 的 per-pattern time 隨 patterns 增加略降，符合 GPU 需要大量
independent work 來提高 latency hiding 的架構特性。

因此，本專題最好的敘述不是「每一階段都加速」，而是「不同平行化方式揭露不同 overhead
與硬體限制」：P2 展示 reduction overhead，P3 展示 lanes 對 output pixels 的 SIMD-like
優勢，P4 展示 shared memory 對 stencil/window reuse 的幫助，P5 展示 GPU 對大量獨立
patterns 的 scalability。

## 問題與限制

- Docker 已安裝但 container execution 被 kernel namespace 限制阻擋，因此 gem5 使用本機 build。
- gem5 `TimingSimpleCPU` 的絕對時間不可與 host `chrono` 或 CUDA event timing 直接比較。
- P2 的 RVV reduction 實作為了清楚呈現 reduction path，使用暫存 arrays；這會讓 performance
  被 memory traffic 與 setup overhead 主導。
- ncu profiling 會讓程式輸出的 `avg_kernel_ms` 明顯變大，因此正式 timing 使用
  `results/p4_cuda.csv` 與 `results/p5_multi_pattern.csv`，ncu 主要用於 occupancy、
  throughput、register count 分析。

## 結論

本實驗證明 correctness 已通過 reference comparison，且主要效能趨勢合理。RVV reduction
不一定帶來加速，尤其當 reduction 前後需要額外暫存與資料搬移時；SIMD-like RVV 將 lanes
分配給不同 pixels，能避免 reduction overhead 並取得小幅改善；CUDA shared memory 能利用
window overlap 取得穩定但有限的加速；multi-pattern GPU parallelism 則最能展示 GPU 對大量
independent work 的需求與優勢。整體來看，最有效的方向是提高可同時執行的 independent work，
同時控制 register pressure、shared memory usage 與 memory access pattern。
