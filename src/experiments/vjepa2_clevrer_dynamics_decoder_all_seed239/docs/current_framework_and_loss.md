# 当前框架与 Loss 路径

本文档总结当前代码实际执行的 `V-JEPA2 -> UniversalDynamicsDecoder -> CLEVRERShallowProbes` 数据流，并标出参与 `structured_probe_loss` 的分支。

## 维度约定

```text
B：batch size
T：decoder temporal steps，当前 T=16
K：dynamics decoder slots，当前 K=8
N：CLEVRER object slots，当前 N=6
D：decoder hidden dimension，当前 D=256
P：每个 temporal step 的空间 patch 数，P=16×16=256
```

当前 cache 的实际 latent 协议是：

```text
context_tokens: [B, 12, 256, 1280]
future_tokens:  [B,  4, 256, 1280]
tokens:         [B, 16, 256, 1280]
```

predictive 模式的 `source_ids` 为 `[0]*12 + [1]*4`；non-predictive 模式为 `[0]*16`。部分旧文档写成 `8+8` 和 `z_dyn=[B,8,8,256]`，但当前 `cache_full_latents.py`、`model.py` 和 manifest 的实现均对应 `12+4` 与 `z_dyn=[B,16,8,256]`。

## 总体数据流与 Loss

```mermaid
flowchart TD
    A["CLEVRER 视频<br/>RGB: [B,128,H,W,3]"] --> B["冻结 V-JEPA2 ViT-H Encoder<br/>tubelet=2, patch=16"]
    B --> C["Context latent<br/>[B,12,256,1280]"]
    C --> P["冻结 Naive Predictor"]
    P --> F["Predicted future latent<br/>[B,4,256,1280]"]

    C --> E["沿 temporal 维拼接"]
    F --> E
    E --> G["Concat latent<br/>[B,16,256,1280]"]
    G --> H["Linear projection<br/>1280 → 256"]
    H --> I["Spatial / temporal / source embeddings"]
    I --> J["Full-patch memory<br/>[B,16,256,256]<br/>reshape → [B,4096,256]"]

    Q["Learned decoder queries<br/>[1,T,K,D] = [1,16,8,256]"] --> K["Slot cross-attention"]
    J --> K
    K --> L["Slot features<br/>[B,T,K,D] = [B,16,8,256]"]
    L --> M["Per-slot temporal self-attention<br/>[B×K,16,256]"]
    M --> N["Temporal memory reread"]
    J --> N
    N --> O["Per-time slot interaction<br/>[B×16,K,256]"]
    O --> R["Interaction memory reread"]
    J --> R
    R --> Z["z_dyn<br/>[B,T,K,D] = [B,16,8,256]"]
    Z --> ZM["Probe memory<br/>[B,T×K,D] = [B,128,256]"]

    OQ["6 learned object queries<br/>[B,N,D] = [B,6,256]"] --> OA["Object cross-attention"]
    ZM --> OA
    OA --> OB["Object tokens<br/>[B,6,256]"]
    OB --> P1["Presence logits<br/>[B,6]"]
    OB --> P2["Color logits<br/>[B,6,8]"]
    OB --> P3["Material logits<br/>[B,6,2]"]
    OB --> P4["Shape logits<br/>[B,6,3]"]

    TE["16 learned time embeddings"] --> TQ["Object-time queries<br/>[B,6,16,256]"]
    OB --> TQ
    TQ --> TA["Object-time cross-attention"]
    ZM --> TA
    TA --> TB["Object-time features<br/>[B,6,16,256]"]
    TB --> SM["State MLP"]
    SM --> ST["Trajectory/state<br/>[B,6,16,8]"]

    OB --> PA["15 canonical pairs<br/>C(6,2)=15"]
    TB --> PT["Pair-time features"]
    ST --> PT
    PT --> PE1["Pair-time encoder<br/>[B,15,16,256]"]
    PE1 --> PD["Pair distance<br/>[B,15,16]"]
    PE1 --> CO["Contact logits<br/>[B,15,16]"]
    OB --> PO["Object pair features<br/>sum + abs difference"]
    PE1 --> TM["Temporal mean<br/>[B,15,256]"]
    PO --> PE2["Pair encoder"]
    TM --> PE2
    PE2 --> PTK["Pair tokens<br/>[B,15,256]"]
    PTK --> FC["First-contact logits<br/>[B,15,17]"]

    GT["CLEVRER GT targets<br/>object_present [B,6]<br/>attributes [B,6]<br/>state [B,6,16,8]<br/>pair/contact [B,6,6,16]<br/>first_contact [B,6,6]"] --> MATCH["Sample-local object matching"]
    P1 --> MATCH
    P2 --> MATCH
    P3 --> MATCH
    P4 --> MATCH
    ST --> MATCH
    MATCH --> LOSS["Permutation-invariant multi-task loss"]
    P1 --> LOSS
    P2 --> LOSS
    P3 --> LOSS
    P4 --> LOSS
    ST --> LOSS
    PD --> LOSS
    CO --> LOSS
    FC --> LOSS
    GT --> LOSS

    classDef normal fill:#e8eef7,stroke:#52739a,color:#17202a;
    classDef frozen fill:#e5e5e5,stroke:#777,color:#222;
    classDef lossPred fill:#ffe0b2,stroke:#d97706,stroke-width:2px,color:#4a2700;
    classDef target fill:#d9f2d9,stroke:#3f8f4f,stroke-width:2px,color:#153b1d;
    classDef lossCore fill:#ffc7c7,stroke:#c0392b,stroke-width:3px,color:#5b1010;

    class A,B,C,F,G,H,I,J,Q,K,L,M,N,O,R,Z,ZM,OQ,OA,OB,TE,TQ,TA,TB,SM,PA,PT,PE1,PO,TM,PE2,PTK normal;
    class P frozen;
    class P1,P2,P3,P4,ST,PD,CO,FC lossPred;
    class GT target;
    class MATCH,LOSS lossCore;
```

## Loss 分支

object slots 没有固定 CLEVRER object ID。训练前，代码根据 presence、颜色、材质、形状和有效轨迹代价，为每个样本寻找最优 slot permutation；匹配过程在 `no_grad` 下执行。

匹配后计算：

```text
L = L_presence
  + L_attributes
  + 2.0 * L_center
  + 1.0 * L_geometry
  + 0.5 * L_velocity
  + 0.1 * L_log_area
  + 1.0 * L_distance
  + 2.0 * L_contact
  + 1.0 * L_first_contact
```

直接参与 loss 的模型输出为：

```text
presence_logits
color_logits / material_logits / shape_logits
trajectory_2d
pair_distance_2d
contact_gt_event
first_contact_logits
```

`z_dyn`、`object_tokens` 和 `pair_tokens` 是中间表示，不直接计算监督损失，但通过各 prediction head 接收梯度。

## 关键输出

```text
object_tokens:        [B,6,256]
trajectory_2d:        [B,6,16,8]
pair_tokens:          [B,15,256]
pair_distance_2d:     [B,15,16]
contact_gt_event:     [B,15,16]
first_contact_logits: [B,15,17]
```

其中 17 个 first-contact 类别表示 16 个时间 bin 加 1 个 no-contact 类别。QA 导出阶段会冻结该 decoder/probe，并将 object、pair、object-time、pair-time 表示作为结构化 token 接入 QA Transformer。
