# MedSAM→Swin-UMamba 正式实验流程（当前收缩版）

> 适用范围：当前正式 baseline 主线，仅保留  
> 3D：`btcv, synapse, acdc, prostate158`  
> 2D：`kvasirseg, cvc_clinicdb, tn3k, tg3k, ddti, otu_2d, monuseg, ph2`  
> 当前正式学生端只保留两组：`baseline` 与 `upper`。  
> 当前 baseline 目录结构已经固定：**只允许覆盖重写，不允许自动新建新分支目录**。  
> 若为 debug，允许手动新建目录，但必须**手动指定**，不能让批处理脚本自动生成。

---

## 1. 本版总定义

### 1.1 当前正式主线只保留两种模式
- `baseline`：弱监督主线，训练标签为 `pseudo_student/tri_train`
- `upper`：全监督上界，训练标签为 `student_gt`

### 1.3 当前这版真正回答的问题
1. 在严格无泄漏、三空间显式建模、native-space tight box、train-only tri pseudo 的设定下，`tri pseudo` 能否训练出可用 student。
2. 在同一 student、同一 crop、同一评估口径下，`upper` 相对 `baseline` 的差距有多大。

---

## 2. 最重要的硬规则

### 2.1 目录规则
当前 baseline 目录结构已经固定，**所有正式运行只能覆盖原目录**：

- `processed/<dataset>/fold_0/...`
- `work_dir/baseline/<dataset>/fold_0/...`
- `work_dir/upper/<dataset>/fold_0/...`
- `data/vis/teacher/<dataset>/...`
- `data/vis/student/<dataset>/...`

允许执行：
- 删除旧结果
- 在**原固定路径**下重新生成同名文件和同名目录

禁止执行：
- 自动新建 `fold_0_gpu2`
- 自动新建 `fold_0_debug`
- 自动新建 `baseline_v2`
- 自动新建时间戳目录
- 自动新建任何“为了区分实验”的额外层级

### 2.2 debug 例外
如果只是 debug：
- 可以新建目录
- 但必须由你**手动输入目录名**
- 不能写进正式批处理脚本里自动生成
- debug 结果不得混入正式 baseline 统计

### 2.3 正式批处理的工作方式
正式训练、推理、评估脚本都遵循：
- 先 `rm -rf` 原目标目录
- 再 `mkdir -p` 同一个固定目录
- 最后把新结果写回原位置

也就是说：**是覆盖重写，不是并行保留多个版本**。

---

## 3. 三套空间必须严格区分

### 3.1 Native Space
原始图像 / 原始 GT 空间。  
这是最终正式评估的唯一真空间。

### 3.2 Teacher Space
MedSAM 输入空间，固定为：

- `1024 × 1024 × 3`

### 3.3 Student Space
Swin-UMamba 输入空间，按数据集固定尺寸。

### 3.4 三空间规则
任何 bbox、伪标签、预测结果的跨空间变换都必须依赖：

- `geometry_meta.json`

禁止：
- 手写 resize 推导
- 在下游脚本里凭经验反推坐标
- 直接把 student-space 预测拿去做 3D 正式指标

---

## 4. 当前正式数据集范围

### 4.1 3D 数据集
- `btcv`
- `synapse`
- `acdc`
- `prostate158`

### 4.2 2D 数据集
- `kvasirseg`
- `cvc_clinicdb`
- `tn3k`
- `tg3k`
- `ddti`
- `otu_2d`
- `monuseg`
- `ph2`

> 注意：之前一些文档里出现过 `drive / chasedb1 / hrf`，那是更早的大列表。  
> 你这次已经明确收缩，当前正式批处理就只保留上面这 12 个数据集。

---

## 5. 当前固定的 student crop

### 5.1 3D
- `btcv`: `512 × 512`
- `synapse`: `512 × 512`
- `acdc`: `320 × 320`
- `prostate158`: `320 × 320`

### 5.2 2D
- `kvasirseg`: `352 × 352`
- `cvc_clinicdb`: `352 × 352`
- `tn3k`: `256 × 256`
- `tg3k`: `256 × 256`
- `ddti`: `256 × 256`
- `otu_2d`: `256 × 256`
- `monuseg`: `512 × 512`
- `ph2`: `256 × 256`

### 5.3 crop 锁定规则
一旦某个数据集 crop 已用于当前 baseline 周期：
- 不允许中途修改
- 如果以后要改，必须作为**新 crop 分支实验**
- 并重跑所有 student-side 相关链路

但当前你要求的是：
- baseline 文件结构固定
- 正式脚本只覆盖重写

所以这版默认：**不改 crop，不开新 crop 分支**。

---

## 6. 数据划分协议

### 6.1 3D 数据
#### `btcv`
- patient / case / volume-level split
- 禁止 slice-level random split

#### `synapse`
- benchmark split
- 同一 case 的所有 slice 必须在同一 split

#### `acdc`
- official challenge split

#### `prostate158`
- official fixed patient list

### 6.2 2D 数据
#### `kvasirseg`
- fixed widely-used split

#### `cvc_clinicdb`
- fixed widely-used split

#### `tn3k`
- official split

#### `tg3k`
- official `train/val` 映射为当前 `train/test`

#### `ddti`
- stratified split，按良恶性分层

#### `otu_2d`
- internal pre-defined split

#### `monuseg`
- 以当前 processed 中已经固定下来的 split 为准

#### `ph2`
- stratified split，按病理类别分层

### 6.3 无泄漏原则
全流程严格禁止：
- 生成并使用 `test pseudo`
- 用 test split 调阈值
- 用 test split 选 checkpoint
- 用 test prompt 反哺 student 训练
- 3D 数据做 slice-level 洗牌

---

## 7. 固定目录合同

以 `processed/<dataset>/fold_0/` 为例：

```text
processed/<dataset>/fold_0/
├── teacher_npy/
│   ├── imgs/
│   └── gts/
├── student_npy/
│   ├── imgs/
│   └── gts/
├── pseudo_teacher/
│   ├── tri_train/
│   └── vis_train/
├── pseudo_student/
│   ├── tri_train/
│   └── vis_train/
├── prompts/
│   ├── prompts_train.json
│   └── prompts_test.json
└── meta/
    ├── manifest.json
    ├── split_meta.json
    ├── geometry_meta.json
    ├── label_meta.json
    └── leakage_audit.json
```

以 `work_dir` 为例：

```text
work_dir/
├── baseline/<dataset>/fold_0/
│   ├── last.pth
│   ├── train_log.csv
│   ├── pred_test/
│   ├── eval_2d/ 或 eval_3d/
│   ├── run_train_baseline.log
│   ├── run_infer_baseline.log
│   └── run_eval_baseline.log
└── upper/<dataset>/fold_0/
    ├── last.pth
    ├── train_log.csv
    ├── pred_test/
    ├── eval_2d/ 或 eval_3d/
    ├── run_train_upper.log
    ├── run_infer_upper.log
    └── run_eval_upper.log
```

### 7.1 三个最关键元信息文件
必须齐全：
- `manifest.json`
- `split_meta.json`
- `geometry_meta.json`

任何路径查找、空间恢复、3D 重建都必须依赖它们，禁止靠文件名猜。

### 7.2 当前正式可视化合同
当前正式 teacher / student 审阅结果统一收口到：

- `MedSAM-main/data/vis/teacher/<dataset>/`
- `MedSAM-main/data/vis/student/<dataset>/`

当前唯一正式可视化入口为：

- `/storage/baiyuting/data/Swin-UMamba-main/pipeline/screen_cases.py`

说明：
- `processed/.../pseudo_*/vis_train/` 可以作为历史或中间目录存在，但不再作为 canonical 审阅输出根目录。
- `work_dir/.../eval_*/vis/` 不再作为正式审阅目录合同。

---

## 7.5 Stage 0：2D raw mask 审计与 organized label 修复

### 7.5.1 为什么要先做这一层
当前正式 2D 数据集中，部分 raw mask 不是纯二值源，而是 JPEG / 灰度 mask。

已经确认：
- `tn3k`
- `tg3k`
- `kvasirseg`

如果继续用统一 `arr > 0` 做二值化，会把 JPEG 压缩边缘整体并进前景，导致：
- organized `label.png` 外沿变粗
- tight box 偏大
- teacher pseudo 与 student 学习输入继承同样偏差

因此当前正式版要求：
- 2D 数据集必须先做 raw mask 审计
- 先修 organized label
- 再继续 Stage 1 预处理、Stage 1.5 prompt、Stage 2 pseudo 和 student 学习链

### 7.5.2 当前正式 2D 源链修复规则
#### JPEG / 灰度 mask 数据集
- `tn3k`
- `tg3k`
- `kvasirseg`

正式规则：
- 不再使用统一 `>0`
- 默认阈值二值化写死为 `>= 128`
- 若某数据集后续需要更严格阈值，必须先写入代码规则表与文档，不能在下游脚本里临时改

#### 纯二值源数据集
- `cvc_clinicdb`
- `ddti`
- `otu_2d`
- `ph2`

正式规则：
- 保持纯二值原样
- 不做前景扩张

### 7.5.3 当前已知的系统性非病灶组件
#### `tg3k`
- 左上固定小白点
- 顶边孤立小点

当前正式版把它们视为系统性非病灶组件：
- 不进入 organized label 正式前景
- 不参与 prompt bbox
- 不参与 teacher / student 正式审阅框显示

#### `kvasirseg`
- 固定角点 / 极小 JPEG 残留

当前正式版同样视为系统性非病灶组件处理。

### 7.5.4 审计产物
organized 层每个 2D 数据集都会额外写出：
- `meta/label_source_audit.json`
- `meta/label_source_audit.csv`
- `meta/label_source_audit_summary.json`

用途：
- 记录 raw mask 来源与唯一值分布
- 记录当前数据集使用的二值化规则
- 记录是否移除了系统性非病灶组件

---

## 8. Stage 1：预处理

### 8.1 目的
建立三空间合同：
- `native -> teacher`
- `teacher -> native`
- `native -> student`
- `student -> native`

并导出：
- `teacher_npy/imgs`
- `teacher_npy/gts`
- `student_npy/imgs`
- `student_npy/gts`
- `manifest.json`
- `split_meta.json`
- `geometry_meta.json`

### 8.2 正式命令
```bash
cd /storage/baiyuting/data/MedSAM-main

python utils/processed.py   --organized_root /storage/baiyuting/data/MedSAM-main/data/organized   --processed_root /storage/baiyuting/data/MedSAM-main/data/processed   --datasets btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,monuseg,ph2   --student_size_json /storage/baiyuting/data/MedSAM-main/utils/data/student_size_policy_literature.json
```

### 8.3 运行后必须检查
- `meta/manifest.json`
- `meta/split_meta.json`
- `meta/geometry_meta.json`

尤其确认：
- 3D 数据有 `slice_idx`
- 3D 数据有 `spacing`
- `teacher_target_size` 正确
- `student_target_h / student_target_w` 正确

---

## 9. Stage 1.5：strict prompt 生成

### 9.1 固定规则
bbox 必须满足：
- 只由 GT 自动生成
- 只在 native space 定义
- 只允许 tight bbox
- 目前不允许 expansion / margin / jitter / random perturbation

### 9.2 三种规则
- `instance`：单类多目标
- `union`：弥散结构整体取一个框
- `per_class_component`：多类任务每类按离散连通域分别出框

### 9.3 当前 12 个数据集的建议口径
- `btcv / synapse / acdc / prostate158`：`per_class_component`
- `kvasirseg / cvc_clinicdb / tn3k / tg3k / ddti / otu_2d / monuseg / ph2`：默认单前景任务，按 `instance` 或单目标 tight box 主线处理

### 9.3.1 当前正式新增规则：连通域独立 bbox
当前正式版补充写死：

- 2D：一真实病灶一框
- 3D 多类别：同类多个离散区域也必须分别出框
- 严禁再用一个 union box 套住同类多个离散连通域

也就是说：
- 2D 多病灶样本不能把两个病灶合成一个大框
- `btcv / synapse / acdc / prostate158` 中，同一器官若在某个 slice 出现多个离散连通域，也必须分别给 box

### 9.4 正式命令
```bash
cd /storage/baiyuting/data/MedSAM-main

python generate_prompts.py   --processed_root /storage/baiyuting/data/MedSAM-main/data/processed   --datasets btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,monuseg,ph2
```

### 9.5 prompt 对齐检查
正式要求：
- `missing_in_manifest = 0`

建议检查：
```bash
python - <<'PY'
import os, json

fold_root = "/storage/baiyuting/data/MedSAM-main/data/processed/kvasirseg/fold_0"

with open(os.path.join(fold_root, "meta", "manifest.json"), "r", encoding="utf-8") as f:
    manifest = json.load(f)

with open(os.path.join(fold_root, "prompts_train.json"), "r", encoding="utf-8") as f:
    prompts = json.load(f)

manifest_train = {x["slice_name"] for x in manifest if x.get("split") == "train"}
prompt_keys = set(prompts.keys())

print("manifest_train =", len(manifest_train))
print("prompt_keys =", len(prompt_keys))
print("missing_in_manifest =", len(prompt_keys - manifest_train))
print("first_20_missing =", list(sorted(prompt_keys - manifest_train))[:20])
PY
```

### 9.6 当前最容易出错的点
最常见问题：
- 旧 `prompts_train.json` 没删
- 新旧 prompt 混读
- `find_prompt_json(...)` 优先读到了旧 prompt

所以正式重跑前建议先删：
- 根目录旧 prompt
- `prompts/` 子目录旧 prompt

---

## 10. Stage 2：train tri pseudo 生成

### 10.1 当前主线唯一合法标签定义
对每个像素：

- 框外：`0`
- 框内且被 MedSAM 判定为前景：`1..K`
- 框内但未被确认前景：`255`

### 10.2 补充规则
- 二分类任务前景 = `1`
- 多分类任务保留原始类别 `1..K`
- 多类冲突区统一写 `255`
- 3D 空切片直接全 `0`，不送入 MedSAM

### 10.3 当前正式主线只保留
- `pseudo_teacher/tri_train`
- `pseudo_student/tri_train`
- 可选 `vis_train`

### 10.4 当前明确不保留
- `hard pseudo`
- `weight_map`
- `test pseudo`

### 10.5 正式命令
```bash
cd /storage/baiyuting/data/MedSAM-main

python generate_pseudo_labels.py   --base_dir /storage/baiyuting/data/MedSAM-main/data   --checkpoint /storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth   --datasets btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,monuseg,ph2   --fold all   --split train   --overwrite   --vis_limit 0
```

### 10.6 这里必须写死的规则
- 正式批量只跑 `--split train`
- 正式批量统一 `--vis_limit 0`
- 正式批量允许 `--overwrite`
- 不允许 `--split all`
- 不允许生成 `test pseudo`

### 10.7 小样本 sanity check
第一次建议先跑一个代表性数据集：
- `btcv`

比如：
```bash
python generate_pseudo_labels.py   --base_dir /storage/baiyuting/data/MedSAM-main/data   --checkpoint /storage/baiyuting/data/MedSAM-main/work_dir/MedSAM/medsam_vit_b.pth   --datasets btcv   --fold fold_0   --split train   --max_samples 120   --overwrite   --vis_limit 0
```

### 10.8 伪标签阶段必须检查
- `pseudo_student/tri_train` 是否存在
- `pseudo_box255_stats_train.json` 是否正常
- `stage_time_pseudo_train.json` 是否生成
- `255` 是否只在 bbox 内
- 空切片是否全 0

---

## 11. Stage 2.5：学生端数据检查

正式训练前必须做一次数据集读取检查：

```bash
python test_student_patch_dataset.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --split train
```

必须确认：
- baseline 能正确读 `pseudo_student/tri_train`
- upper 能正确读 `student_gt`
- image / mask shape 对齐
- `255` 只在 baseline 中出现

---

## 12. Stage 3：学生端训练

### 12.1 当前正式两组
#### baseline
- 训练标签：`pseudo_student/tri_train`
- `255` 作为 ignore

#### upper
- 训练标签：`student_gt`

### 12.2 当前固定超参数主线
- `epochs = 50`
- `lr = 1e-4`
- `weight_decay = 0.05`
- `freeze_encoder_epochs = 10`
- `amp = on`
- `deep_supervision = on`

### 12.3 当前 batch size / num_workers
- `btcv`: `batch_size=1`, `num_workers=0`
- `synapse`: `batch_size=1`, `num_workers=0`
- `acdc`: `batch_size=2`, `num_workers=2`
- `prostate158`: `batch_size=2`, `num_workers=2`
- `kvasirseg`: `batch_size=8`, `num_workers=4`
- `cvc_clinicdb`: `batch_size=8`, `num_workers=4`
- `tn3k`: `batch_size=8`, `num_workers=4`
- `tg3k`: `batch_size=8`, `num_workers=4`
- `ddti`: `batch_size=8`, `num_workers=4`
- `otu_2d`: `batch_size=8`, `num_workers=4`
- `monuseg`: `batch_size=8`, `num_workers=4`
- `ph2`: `batch_size=8`, `num_workers=4`

### 12.4 正式训练命令模板
#### baseline
```bash
CUDA_VISIBLE_DEVICES=2 python -u /storage/baiyuting/data/Swin-UMamba-main/pipeline/train_student.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --mode baseline   --epochs 50   --batch_size <bs>   --num_workers <nw>   --lr 1e-4   --weight_decay 0.05   --freeze_encoder_epochs 10   --amp   --deep_supervision   --pretrained_ckpt /storage/baiyuting/data/Swin-UMamba-main/data/pretrained/vmamba/vmamba_tiny_e292.pth   --out_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0
```

#### upper
```bash
CUDA_VISIBLE_DEVICES=2 python -u /storage/baiyuting/data/Swin-UMamba-main/pipeline/train_student.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --mode upper   --epochs 50   --batch_size <bs>   --num_workers <nw>   --lr 1e-4   --weight_decay 0.05   --freeze_encoder_epochs 10   --amp   --deep_supervision   --pretrained_ckpt /storage/baiyuting/data/Swin-UMamba-main/data/pretrained/vmamba/vmamba_tiny_e292.pth   --out_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/upper/<dataset>/fold_0
```

### 12.5 当前正式批处理脚本的行为
你的 `run_train_all.sh` 已经写死：
- 只处理这 12 个数据集
- `btcv / synapse` 固定 `bs=1, nw=0`
- 其余 3D 用 `bs=2, nw=2`
- 其余 2D 用 `bs=8, nw=4`
- 输出目录固定为 `work_dir/<mode>/<dataset>/fold_0`
- 运行前先删原目录再写回同一目录

---

## 13. Stage 4：推理

### 13.1 输出目录
固定写入：
- `work_dir/baseline/<dataset>/fold_0/pred_test`
- `work_dir/upper/<dataset>/fold_0/pred_test`

### 13.2 正式命令模板
```bash
python /storage/baiyuting/data/Swin-UMamba-main/pipeline/infer_student.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --dataset <dataset>   --mode baseline   --deep_supervision   --ckpt /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/last.pth   --out_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/baseline/<dataset>/fold_0/pred_test   --amp
```

upper 只要把 `mode` 和 `ckpt/out_dir` 换成 upper。

### 13.3 当前批处理脚本行为
`run_infer_eval_all.sh` 的规则也是：
- 先删 `pred_test`
- 再在原固定路径重建
- 不新建其他版本目录

---

## 14. Stage 5：正式评估

### 14.1 2D 数据
当前正式指标：
- `Dice`
- `IoU`
- `MAE`（只对 `kvasirseg`、`cvc_clinicdb` 加 `--report_mae`）

正式命令模板：
```bash
python /storage/baiyuting/data/Swin-UMamba-main/pipeline/eval_2d.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --pred_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/pred_test   --save_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/eval_2d   --split test   --require_native_gt
```

### 14.2 3D 数据
当前正式指标：
- `DSC`
- `HD95(mm)`
- `ASSD(mm)`

正式原则：
- 必须先恢复到 native slice 尺寸
- 必须按 `slice_idx` 排序重建 volume
- 必须使用真实 `spacing`
- 多类任务必须逐器官统计

正式命令模板：
```bash
python /storage/baiyuting/data/Swin-UMamba-main/pipeline/eval_3d.py   --fold_root /storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0   --pred_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/pred_test   --save_dir /storage/baiyuting/data/Swin-UMamba-main/work_dir/<mode>/<dataset>/fold_0/eval_3d   --split test   --require_native_gt
```

### 14.3 当前 3D class name map 必须补齐
正式映射至少要保证：
- `btcv`
- `synapse`
- `acdc`
- `prostate158`

都能输出器官名，而不只是 `class_1/class_2/...`。

---

## 15. Stage 6：计时与汇总

### 15.1 当前正式优先级
时间读取优先级：

```text
stage_time.json > train_log.csv > file mtime estimate
```

### 15.2 每阶段应有的 stage_time
#### MedSAM 侧
- `stage_time_preprocess.json`
- `stage_time_prompts.json`
- `stage_time_pseudo_train.json`

#### Swin-UMamba 侧
- `stage_time_train_baseline.json`
- `stage_time_infer_baseline.json`
- `stage_time_eval_baseline.json`
- `stage_time_train_upper.json`
- `stage_time_infer_upper.json`
- `stage_time_eval_upper.json`

### 15.3 汇总前必须确认
如果 timer 还显示很长的旧时间：
- 先确认对应阶段是否真的生成了新的 `stage_time_*.json`
- 如果没有，就必须先重跑该阶段
- 再重跑 timer

---

## 16. 当前最稳执行顺序

### 16.1 单数据集先跑通
建议先拿 `btcv` 跑通：

```text
processed
→ generate_prompts
→ 对齐检查（missing_in_manifest = 0）
→ generate_pseudo_labels (train, vis_limit=0)
→ 检查 pseudo stats + stage_time
→ test_student_patch_dataset
→ train_student baseline
→ train_student upper
→ infer baseline
→ eval baseline
→ infer upper
→ eval upper
→ timer 汇总
```

### 16.2 全量批跑顺序
确认 `btcv` 跑通后，再批量扩到 12 个数据集：

1. 全部数据集 `processed`
2. 全部数据集 `generate_prompts`
3. 全部数据集 `generate_pseudo_labels --split train --vis_limit 0`
4. 逐个 spot check `pseudo_student/tri_train`
5. 跑 `run_train_all.sh`
6. 跑 `run_infer_eval_all.sh`
7. 跑总表汇总脚本
8. 跑 timer 汇总

---

## 2026-04 GT 修复与统一可视化变更摘要

### 一、这次为什么要改
本轮变更不是单纯“换一套可视化样式”，而是修正式 baseline 中已经进入源链的 2D GT 偏差，并把 teacher / student 的审阅入口统一。

核心原因有两类：

#### 1. 2D GT 源链问题
已经确认部分 2D 数据集的 raw mask 不是纯二值源，而是 JPEG / 灰度 mask：
- `tn3k`
- `tg3k`
- `kvasirseg`

旧 organize 逻辑使用统一：

```python
arr = (arr > 0).astype(np.uint8) * 255
```

这会把 JPEG 压缩边缘整体并进前景，导致：
- organized `label.png` 外沿变粗
- native-space tight box 偏大
- teacher pseudo 继承偏差
- student baseline 训练标签也继承偏差

#### 2. 审阅入口分散
旧版 teacher / student 审阅图分散在不同目录与脚本里，不利于统一排查：
- teacher 历史图可能来自旧 `vis_train`
- student 图可能来自 `work_dir/.../eval_*/vis`
- teacher / student 还有不同入口脚本

当前正式版统一为：
- 唯一正式入口：`/storage/baiyuting/data/Swin-UMamba-main/pipeline/screen_cases.py`
- 唯一 canonical 审阅目录：`MedSAM-main/data/vis/teacher|student/<dataset>/`

---

### 二、这次适用范围
当前正式版仍只覆盖 12 个正式 baseline 数据集：

#### 3D
- `btcv`
- `synapse`
- `acdc`
- `prostate158`

#### 2D
- `kvasirseg`
- `cvc_clinicdb`
- `tn3k`
- `tg3k`
- `ddti`
- `otu_2d`
- `monuseg`
- `ph2`

其中：

#### 需要修 organize GT 源链的 2D 数据集
- `tn3k`
- `tg3k`
- `kvasirseg`

#### 需要统一审计但通常保持纯二值原样的 2D 数据集
- `cvc_clinicdb`
- `ddti`
- `otu_2d`
- `monuseg`
- `ph2`

#### 需要修 prompt / 多类着色 / 可视化规则的 3D 数据集
- `btcv`
- `synapse`
- `acdc`
- `prostate158`

---

### 三、数据源从哪里来

#### 1. raw
原始数据统一来自：

```text
/storage/baiyuting/data/MedSAM-main/data/raw/<dataset>/
```

这里保存官方或当前正式使用的原始图像与原始标签来源。

#### 2. organized
organized 层来自 raw，经 `organize_datasets.py` 统一生成：

```text
/storage/baiyuting/data/MedSAM-main/data/organized/<dataset>/
```

这里负责：
- 固定 split 落地
- 图像 / 标签统一命名
- 2D label 规范化与源链审计

#### 3. processed
processed 层来自 organized，经 `utils/processed.py` 生成：

```text
/storage/baiyuting/data/MedSAM-main/data/processed/<dataset>/fold_0/
```

这里负责：
- `teacher_npy`
- `student_npy`
- `manifest.json`
- `split_meta.json`
- `geometry_meta.json`

#### 4. prompts
prompt 由 native GT 自动生成，写到：

```text
processed/<dataset>/fold_0/prompts/
```

正式入口：
- `/storage/baiyuting/data/MedSAM-main/generate_prompts.py`

#### 5. pseudo
teacher tri pseudo 与 student tri pseudo 来自：
- processed fold
- prompts
- MedSAM checkpoint

正式入口：
- `/storage/baiyuting/data/MedSAM-main/generate_pseudo_labels.py`

#### 6. student 学习链
student 训练、推理、评估依赖：
- `processed/<dataset>/fold_0`
- `pseudo_student/tri_train`
- `student_gt`

正式入口位于：
- `/storage/baiyuting/data/Swin-UMamba-main/pipeline/train_student.py`
- `/storage/baiyuting/data/Swin-UMamba-main/pipeline/infer_student.py`
- `/storage/baiyuting/data/Swin-UMamba-main/pipeline/eval_2d.py`
- `/storage/baiyuting/data/Swin-UMamba-main/pipeline/eval_3d.py`

---

### 四、当前正式目录结构

#### 1. organized
```text
data/organized/<dataset>/
├── meta/
│   ├── raw_manifest.json
│   ├── split_meta.json
│   ├── sanity_check.csv
│   ├── label_source_audit.json
│   ├── label_source_audit.csv
│   └── label_source_audit_summary.json
├── train/cases/<case_id>/
└── test/cases/<case_id>/
```

#### 2. processed
```text
data/processed/<dataset>/fold_0/
├── teacher_npy/
│   ├── imgs/
│   └── gts/
├── student_npy/
│   ├── imgs/
│   └── gts/
├── pseudo_teacher/
│   └── tri_train/
├── pseudo_student/
│   └── tri_train/
├── prompts/
│   ├── prompts_train.json
│   └── prompts_test.json
└── meta/
    ├── manifest.json
    ├── split_meta.json
    ├── geometry_meta.json
    ├── label_meta.json
    └── leakage_audit.json
```

#### 3. 可视化
```text
data/vis/
├── teacher/<dataset>/
│   ├── activation/
│   ├── preview/
│   ├── index.json
│   ├── preview_index.json
│   ├── palette.json
│   ├── activation_summary.json
│   ├── activation_under_ranking.json / .csv
│   ├── activation_over_ranking.json / .csv
│   └── activation_both_ranking.json / .csv
└── student/<dataset>/
    ├── pseudo_suspect/
    ├── model_suspect/
    ├── both_bad/
    ├── boundary_suspect/
    ├── palette.json
    └── screen_summary.json
```

#### 4. student 正式工作目录
```text
work_dir/
├── baseline/<dataset>/fold_0/
│   ├── pred_test/
│   ├── eval_2d/ 或 eval_3d/
│   ├── stage_time_train_baseline.json
│   ├── stage_time_infer_baseline.json
│   └── stage_time_eval_baseline.json
└── upper/<dataset>/fold_0/
    ├── pred_test/
    ├── eval_2d/ 或 eval_3d/
    ├── stage_time_train_upper.json
    ├── stage_time_infer_upper.json
    └── stage_time_eval_upper.json
```

---

### 五、2D GT 这次到底怎么修

#### 1. 二值化规则
对 raw mask 的正式规则不再一刀切，而是按数据集来源区分：

##### JPEG / 灰度 mask
- `tn3k`
- `tg3k`
- `kvasirseg`

正式规则：
- 默认使用 `>= 128` 二值化
- 禁止继续用统一 `>0`

##### 纯二值 mask
- `cvc_clinicdb`
- `ddti`
- `otu_2d`
- `ph2`

正式规则：
- 保持原有前景定义
- 不做额外扩张

#### 2. 系统性非病灶组件处理
当前正式版会把下列已确认的系统性伪组件从 organized label 与 prompt 中去掉：

##### `tg3k`
- 左上固定小白点
- 顶边孤立小点

##### `kvasirseg`
- 固定角点 / 极小 JPEG 残留

其他数据集不做一刀切删除，只按审计结果处理。

#### 3. 审计文件
每个 2D organized 数据集都会输出：
- `label_source_audit.json`
- `label_source_audit.csv`
- `label_source_audit_summary.json`

任何后续关于“GT 为什么变粗 / 为什么有边界噪点”的排查，都应先看这三个文件。

---

### 六、prompt 这次怎么改

#### 1. 仍然保持 tight box 正式原则
- 只由 GT 自动生成
- 只在 native space 定义
- 不允许 expansion / margin / jitter / random perturbation

#### 2. 当前新增的正式规则
##### 2D
- 一真实病灶一框
- 严禁一个大框套多个离散病灶

##### 3D 多类别
- 同类多个离散连通域分别出框
- 不再使用 `per_class_union` 把同一器官多个离散区域合并成一个大框

#### 3. 当前正式 3D 规则口径
- `btcv / synapse / acdc / prostate158`：`per_class_component`

#### 4. 当前正式 2D 规则口径
- `kvasirseg / cvc_clinicdb / tn3k / tg3k / ddti / otu_2d / monuseg / ph2`：`instance`

---

### 七、pseudo 与 student 怎么接

#### 1. teacher pseudo
teacher tri pseudo 仍按正式 train-only 主线生成：

```text
processed/<dataset>/fold_0/pseudo_teacher/tri_train/
processed/<dataset>/fold_0/pseudo_student/tri_train/
```

#### 2. student 训练标签
正式仍只保留两组：

##### baseline
- 训练标签：`pseudo_student/tri_train`
- `255` 作为 ignore

##### upper
- 训练标签：`student_gt`

#### 3. 这次为什么 student 也要跟着重跑
因为：
- 2D GT 修复会改变 organized label 与 processed GT
- prompt 修复会改变 tight box
- teacher pseudo 会随之变化

所以当前正式版要求：
- 对受影响的 2D 数据集，从 organize 之后整条正式链路重跑
- student baseline / upper 都在固定目录中覆盖重写

---

### 八、统一可视化怎么做

#### 1. 唯一正式入口
```text
/storage/baiyuting/data/Swin-UMamba-main/pipeline/screen_cases.py
```

#### 2. Teacher 输出
写到：

```text
MedSAM-main/data/vis/pseudo/<dataset>/
```

主要内容：
- `activation/`：正式 bad pseudo 审阅图
- `preview/`：每个数据集 3 张预览样片
- `index.json`
- `preview_index.json`
- `activation_summary.json`
- 各类 ranking 文件

#### 3. Student 输出（当前暂缓批量生成）
student 端可视化先不作为当前阶段闸门，后续再统一生成，暂定目录为：

```text
Swin-UMamba-main/work_dir/vis/baseline/<dataset>/
Swin-UMamba-main/work_dir/vis/upper/<dataset>/
```

#### 4. 当前 preview 机制
正式全量覆盖前，先给每个数据集生成 3 张 Teacher 样片：
- `clean`
- `split`
- `artifact_sensitive`

写到：

```text
MedSAM-main/data/vis/pseudo/<dataset>/preview/
```

只有在样片确认后，才继续正式 teacher 覆盖和 student 学习链重跑。

---

### 九、正式执行顺序

当前正式顺序固定为：

```text
2D raw mask audit
→ organized label rebuild
→ processed
→ generate_prompts
→ generate_pseudo_labels (train only)
→ teacher previews (all 12 datasets, 3 per dataset)
→ teacher full review outputs
→ test_student_patch_dataset
→ run_train_all.sh
→ run_infer_eval_all.sh
→ eval summary
→ timer summary
```

说明：
- 不是重做 split
- 只是沿用当前固定 split 名单，在固定目录里覆盖重写

---

### 十、运行环境

#### MedSAM 侧
- 环境：`medsam310`
- 操作目录：`/storage/baiyuting/data/MedSAM-main`

#### Swin-UMamba 侧
- 环境：`swin_umamba`
- 操作目录：`/storage/baiyuting/data/Swin-UMamba-main`

正式执行命令必须显式激活对应环境后再运行。

---

### 十一、覆盖重写原则
当前正式版继续沿用 baseline 硬规则：

- 只允许覆盖固定目录
- 不允许自动新建 `debug / v2 / timestamp` 等新层级
- 不允许把 debug 结果混入正式统计

也就是说：
- `organized / processed / data/vis / work_dir`
  都是在原固定路径内覆盖重写
- 不并行保留多个正式版本目录

---

### 十二、2026-04 可视化一致性修复补充（GT 对齐与路径收口）

#### 1. 触发问题
在 teacher 伪标签审阅图中，部分 2D 样本（例如 `ph2/IMD090`）出现“第二列 GT 与原 GT 不匹配”的异常，表现为对角白带和角部白块。

#### 2. 根因判定
根因定位为**可视化渲染层 bug**，不是 GT 数据链错误：
- 当 GT 连通域贴边时，`find_contours` 会返回开口轮廓；
- 旧实现把开口轮廓直接 `ax.fill`，Matplotlib 会自动闭合端点，导致斜向伪多边形（看起来像“GT 错位”）。

#### 3. 链路核查结论（全 2D）
已对 `kvasirseg / cvc_clinicdb / tn3k / tg3k / ddti / otu_2d / monuseg / ph2` 完成 organized 与 processed 一致性审计：
- `organized label` vs `processed native_gt`：全部 `mismatch=0`
- 结论：GT 数据本身与 processed 正式链一致，问题仅在旧版可视化绘制方法

#### 4. 修复动作
- 文件：`/storage/baiyuting/data/Swin-UMamba-main/pipeline/screen_cases.py`
- 函数：`draw_binary_gt_fill`
- 变更：由“轮廓多边形填充”改为“按栅格 mask 直接填充（nearest）”，杜绝贴边开口轮廓导致的斜向闭合伪影

#### 5. 输出目录收口修正
teacher 可视化 canonical 路径正式改为：

```text
/storage/baiyuting/data/MedSAM-main/data/vis/pseudo/<dataset>/
```

说明：
- 该目录承载 teacher 伪标签审阅（activation/preview/index/ranking）
- 旧 `data/vis/teacher/<dataset>` 保留历史产物，不再作为正式口径

#### 6. 当前阶段边界
- teacher（伪标签筛查）继续作为当前审阅主线；
- student（eval/test 预测可视化）当前按你的要求暂缓批量生成，后续在 `work_dir/vis/baseline|upper/<dataset>` 落地。

---

### 十三、2026-04-12 评估与时间总表（12 数据集）

本节总表来源于以下自动汇总文件：
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_metrics_2d.csv`
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_metrics_3d.csv`
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_metrics_3d_per_organ.csv`
- `/storage/baiyuting/data/Swin-UMamba-main/work_dir/summary_time_12datasets_hdotm.csv`

#### 1. 2D 总指标表（8 datasets）

| dataset | baseline_dice | upper_dice | delta_dice_upper_minus_baseline | baseline_iou | upper_iou | delta_iou_upper_minus_baseline | baseline_mae_fg | upper_mae_fg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kvasirseg | 0.8426 | 0.9242 | 0.0816 | 0.7416 | 0.8703 | 0.1287 | 0.0547 | 0.0215 |
| cvc_clinicdb | 0.8394 | 0.9008 | 0.0614 | 0.7498 | 0.8548 | 0.1049 | 0.0270 | 0.0078 |
| tn3k | 0.8004 | 0.8318 | 0.0314 | 0.6924 | 0.7371 | 0.0447 |  |  |
| tg3k | 0.5790 | 0.7706 | 0.1916 | 0.4334 | 0.6855 | 0.2521 |  |  |
| ddti | 0.7641 | 0.7990 | 0.0349 | 0.6521 | 0.6960 | 0.0438 |  |  |
| otu_2d | 0.8153 | 0.8493 | 0.0340 | 0.7206 | 0.7733 | 0.0527 |  |  |
| monuseg | 0.7313 | 0.7367 | 0.0054 | 0.5774 | 0.5838 | 0.0065 |  |  |
| ph2 | 0.8985 | 0.9487 | 0.0503 | 0.8176 | 0.9052 | 0.0876 |  |  |

#### 2. 3D 总指标表（4 datasets）

| dataset | baseline_macro_dice | upper_macro_dice | delta_dice_upper_minus_baseline | baseline_macro_hd95_mm | upper_macro_hd95_mm | baseline_macro_assd_mm | upper_macro_assd_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| btcv | 0.5213 | 0.7906 | 0.2694 | 28.669 | 13.001 | 6.641 | 2.944 |
| synapse | 0.7263 | 0.8564 | 0.1301 | 17.020 | 9.531 | 3.376 | 1.856 |
| acdc | 0.5834 | 0.8813 | 0.2980 | 14.109 | 3.616 | 3.979 | 0.842 |
| prostate158 | 0.7570 | 0.8204 | 0.0634 | 4.615 | 4.583 | 1.433 | 1.074 |

#### 3. 3D 分器官指标表（class-wise）

| dataset | class_id | label_name | baseline_dice | upper_dice | delta_dice_upper_minus_baseline | baseline_hd95_mm | upper_hd95_mm | delta_hd95_upper_minus_baseline | baseline_assd_mm | upper_assd_mm | delta_assd_upper_minus_baseline | n_cases_baseline | n_cases_upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| btcv | 1 | spleen | 0.7984 | 0.9385 | 0.1401 | 19.594 | 10.694 | -8.900 | 4.695 | 2.167 | -2.527 | 6 | 6 |
| btcv | 2 | right_kidney | 0.6520 | 0.7501 | 0.0981 | 23.776 | 17.315 | -6.461 | 8.660 | 5.999 | -2.661 | 6 | 6 |
| btcv | 3 | left_kidney | 0.7170 | 0.8108 | 0.0937 | 21.995 | 16.759 | -5.236 | 4.740 | 3.123 | -1.617 | 6 | 6 |
| btcv | 4 | gallbladder | 0.1126 | 0.6338 | 0.5212 | 28.656 | 12.632 | -16.024 | 6.834 | 3.754 | -3.080 | 6 | 6 |
| btcv | 5 | esophagus | 0.6099 | 0.7947 | 0.1848 | 20.995 | 7.465 | -13.530 | 3.610 | 1.444 | -2.166 | 6 | 6 |
| btcv | 6 | liver | 0.7879 | 0.9652 | 0.1773 | 38.790 | 30.631 | -8.158 | 9.197 | 3.187 | -6.010 | 6 | 6 |
| btcv | 7 | stomach | 0.7081 | 0.8680 | 0.1599 | 32.744 | 13.253 | -19.491 | 7.072 | 2.758 | -4.314 | 6 | 6 |
| btcv | 8 | aorta | 0.8928 | 0.9239 | 0.0311 | 37.560 | 11.018 | -26.542 | 5.516 | 1.366 | -4.150 | 6 | 6 |
| btcv | 9 | ivc | 0.7259 | 0.8707 | 0.1448 | 11.483 | 7.270 | -4.213 | 2.872 | 1.585 | -1.288 | 6 | 6 |
| btcv | 10 | portal_vein_splenic_vein | 0.2351 | 0.7328 | 0.4977 | 56.191 | 10.825 | -45.366 | 13.875 | 3.073 | -10.802 | 6 | 6 |
| btcv | 11 | pancreas | 0.5371 | 0.7011 | 0.1640 | 23.572 | 12.595 | -10.977 | 5.980 | 3.131 | -2.848 | 6 | 6 |
| btcv | 12 | right_adrenal_gland | 0.0000 | 0.7242 | 0.7242 |  | 5.799 |  |  | 1.067 |  | 6 | 6 |
| btcv | 13 | left_adrenal_gland | 0.0000 | 0.5646 | 0.5646 |  | 12.762 |  |  | 5.617 |  | 6 | 6 |
| synapse | 1 | aorta | 0.8652 | 0.9165 | 0.0513 | 8.199 | 7.111 | -1.088 | 1.389 | 1.119 | -0.270 | 12 | 12 |
| synapse | 2 | gallbladder | 0.4116 | 0.6587 | 0.2471 | 10.383 | 14.832 | 4.449 | 3.075 | 2.801 | -0.273 | 12 | 12 |
| synapse | 3 | left_kidney | 0.7990 | 0.8966 | 0.0977 | 10.698 | 4.394 | -6.304 | 2.348 | 0.988 | -1.359 | 12 | 12 |
| synapse | 4 | right_kidney | 0.7767 | 0.8671 | 0.0904 | 24.655 | 7.038 | -17.618 | 4.281 | 2.163 | -2.118 | 12 | 12 |
| synapse | 5 | liver | 0.8020 | 0.9623 | 0.1603 | 32.558 | 10.641 | -21.917 | 6.038 | 1.553 | -4.485 | 12 | 12 |
| synapse | 6 | pancreas | 0.5862 | 0.7381 | 0.1519 | 16.061 | 12.310 | -3.751 | 3.227 | 2.210 | -1.018 | 12 | 12 |
| synapse | 7 | spleen | 0.7806 | 0.9360 | 0.1554 | 17.593 | 7.328 | -10.265 | 3.601 | 1.711 | -1.890 | 12 | 12 |
| synapse | 8 | stomach | 0.7892 | 0.8760 | 0.0869 | 16.010 | 12.593 | -3.417 | 3.047 | 2.305 | -0.742 | 12 | 12 |
| acdc | 1 | rv | 0.7590 | 0.8859 | 0.1270 | 8.552 | 4.373 | -4.180 | 2.138 | 0.989 | -1.149 | 100 | 100 |
| acdc | 2 | myo | 0.5484 | 0.8422 | 0.2938 | 11.666 | 3.087 | -8.579 | 3.156 | 0.716 | -2.440 | 100 | 100 |
| acdc | 3 | lv | 0.4427 | 0.9158 | 0.4731 | 22.109 | 3.388 | -18.721 | 6.643 | 0.820 | -5.823 | 100 | 100 |
| prostate158 | 1 | central_gland | 0.8302 | 0.8788 | 0.0486 | 4.720 | 4.786 | 0.066 | 1.503 | 1.158 | -0.345 | 19 | 19 |
| prostate158 | 2 | peripheral_zone | 0.6838 | 0.7620 | 0.0782 | 4.511 | 4.380 | -0.130 | 1.363 | 0.989 | -0.374 | 19 | 19 |

#### 4. 时间统计表（单位：小时.分钟）

`h.mm` 解释：`2.08` 表示 2 小时 08 分钟。

| dataset | preprocess_h.mm | prompts_h.mm | pseudo_train_h.mm | baseline_train_h.mm | baseline_infer_h.mm | baseline_eval_h.mm | upper_train_h.mm | upper_infer_h.mm | upper_eval_h.mm | total_h.mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kvasirseg | 0.05 | 0.00 | 0.06 | 0.23 | 0.00 | 0.00 | 0.23 | 0.00 | 0.00 | 0.58 |
| cvc_clinicdb | 0.00 | 0.00 | 0.04 | 0.14 | 0.00 | 0.00 | 0.14 | 0.00 | 0.00 | 0.33 |
| tn3k | 0.07 | 0.00 | 0.19 | 0.50 | 0.01 | 0.00 | 0.48 | 0.01 | 0.00 | 2.08 |
| tg3k | 0.04 | 0.00 | 0.47 | 0.55 | 0.01 | 0.00 | 0.54 | 0.01 | 0.00 | 2.41 |
| ddti | 0.02 | 0.00 | 0.05 | 0.09 | 0.00 | 0.00 | 0.09 | 0.00 | 0.00 | 0.24 |
| otu_2d | 0.07 | 0.00 | 0.09 | 0.17 | 0.01 | 0.00 | 0.17 | 0.01 | 0.00 | 0.52 |
| monuseg | 0.00 | 0.00 | 0.01 | 0.02 | 0.00 | 0.00 | 0.02 | 0.00 | 0.00 | 0.05 |
| ph2 | 0.00 | 0.00 | 0.01 | 0.03 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.08 |
| btcv | 0.05 | 0.01 | 0.27 | 7.09 | 0.02 | 0.16 | 7.06 | 0.02 | 0.20 | 15.29 |
| synapse | 0.13 | 0.01 | 0.17 | 5.11 | 0.01 | 0.25 | 5.15 | 0.01 | 0.25 | 11.50 |
| acdc | 0.02 | 0.00 | 0.17 | 2.04 | 0.01 | 0.01 | 2.10 | 0.01 | 0.01 | 4.37 |
| prostate158 | 0.08 | 0.01 | 0.13 | 3.51 | 0.01 | 0.00 | 3.49 | 0.01 | 0.00 | 8.05 |

---
