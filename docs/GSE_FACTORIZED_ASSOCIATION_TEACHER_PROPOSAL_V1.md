# Factorized GSE-Graph 关联 Teacher 修订提案 V1

状态：`DATA_TEACHER_DEFECT_CONFIRMED_MATERIAL_REBUILD_NOT_STARTED`

## 已发现的缺陷

正式 inventory 证明连续几何和完整 token 数据可用，但旧 open-set pair cache 不适合 decision-node association：

- fit decision pairs：`23,359` positives / `27` negatives；
- selection：`8,083` positives / `71` negatives；
- fit只有S03/S07/S10含负对；selection只有S01/S07/S08/S09含负对；
- 其余family的“全正”并不证明安全，只说明16 m同world近邻生成器没有产生不同节点候选。

如果直接训练，模型几乎只学习“接受”；如果按family报告precision，多数family没有负样本，安全门槛不可识别。这个问题属于association Teacher/sample unit，不是LiDAR、token、geometry profile或系统故障。

## 自动推荐方案 A

建立一个**identity-balanced structural alias Teacher**，同时保留两个不同证据域：

1. runtime candidate域保持不变：同world、严格过去、3D距离`<=16 m`，用于报告真实在线尝试；
2. structural alias域按decision identity平衡构造，用于训练和严格检验表示是否混淆相似节点。

每个junction/terminal identity确定性产生：

- 一个positive：优先选择不同physical edge的两个有效观测；无不同edge时使用同edge反向/最大观测间隔；仅对客观上只有一条末端观测的短terminal使用固定环向旋转增强；
- 一个hard negative：不同identity、相同event type、相同incident degree，按objective inbound geometry profile距离最近；
- Teacher geometry只用于负样本选择和标签，禁止进入学生特征；
- fit只在C01--C06内部构造，selection只在C07--C08内部构造，禁止跨split配对；
- 若同event/degree无唯一候选，按identity字典序决胜；若完全无候选，显式记录unavailable，不替换类别或跨split借样本。

学生候选输入固定为：

```text
完整exit token集合
+ place descriptor
+ 当前决策不确定性
+ 已实际穿越的incoming/incident edge geometry profile
+ measured spatial distance（只在runtime域）
```

## 不采用的方案

- 不扩大16 m runtime radius来制造负样本：会改变在线图候选域并引入新的图超参。
- 不复制27/71个负对做class balancing：重复样本不能增加结构多样性。
- 不把不同world身份作为可输入特征：world ID只用于split和Teacher审计。
- 不退回强制最近距离loop merge：会放弃不确定性拒绝和学习关联贡献。

## 实施前必须冻结的证据

- 792个fit、274个selection decision identities逐identity正/负pair manifest；
- event、degree、same/different-edge、family分层；
- objective geometry hard-negative距离分布；
- 15个fit和5个selection缺少5帧inbound profile的identity清单及明确mask；
- 每个输入列的部署可用性和Teacher隔离审计；
- 预注册capacity proof的precision、false accept、recall和per-family门槛。

该提案不授权直接训练；下一材料操作应先生成只读pair manifest proof和Data Card。若identity-balanced Teacher仍缺乏每类负样本或产生跨split/未来泄漏，停止学习关联路线。

## Singleton短terminal修订（2026-08-27）

只读实现诊断发现5个terminal identity（fit 4、selection 1）恰好只有一条terminal标注观测。它们位于短边，剩余1 m采样点客观上属于相邻junction，无法产生第二个真实terminal视角。删除这些身份会造成选择偏差，把同一张量重复计为重访也不成立。

因此只对这5条具有完整5帧历史的terminal序列增加最终fallback：原序列与固定180-bin环向平移后的序列构成正样本。该变换不读取objective geometry或identity作为学生输入，并在结果中单独标为augmentation，不冒充physical revisit。其余1,061个身份仍必须来自不同edge、反向视角或不同观测。hard negative和split隔离合同保持不变。
