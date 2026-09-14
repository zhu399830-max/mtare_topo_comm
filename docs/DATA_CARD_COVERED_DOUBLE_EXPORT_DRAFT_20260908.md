# 原12声明五帧诊断导出卡草案

状态：待绑定机器运行规格和导出校验器；此草案不是已执行导出，也不是训练许可。常规范围按用户持续授权推进，不要求用户进行人工标注。

目的：将已封存的正确距离记录转换成统一五帧输入，供后续来源/结构监督核验；不再执行原生求交。

输入run：`gate3_20260908_gse_double_population_covered_v1_seed20260906`。封条SHA-256：`687cd803127e479a88cd1605c3c35bd4ffb5cc2dfa19649af8e80864fa3474e4`。导出前必须流式核对封条和实际逐条文件，不能仅信summary中的PASS。

精确声明：`double_junction__{circle,ellipse,rounded_rectangle}__view{0,1,2,3}`的笛卡尔积，共12个；不是模糊路径访问，执行器应由冻结清单列出全部case_id。每个声明frame0—4，共60帧、691200条射线、12个五帧观察。

独立性：1种双路口构造拓扑、3种截面实现、每实现4个视点；不能称12个独立世界。每个历史位置间隔标称0.1米（精确float64坐标以冻结清单为准）；方位0.5度、俯仰-15至15度每2度，共16×720。没有 train/validation/test 划分，因为这只是固定合成诊断；不读取C01—C10及机器人测试世界。

输出分离：

- student：五帧ranges、valid_mask、相对平移及相对yaw。
- diagnostic_only：绝对pose、reported退出表面来源代码、codebook；不得进入模型前向。
- source：声明、五个帧索引、输入run与文件SHA。
- qualification：training_eligible=false、structure_labels_present=false、source_completeness_qualified=false；距离证据需导出执行器绑定封存输入，不能由组装函数自称通过。

本批不运行teacher，不产生结构锚点、开口或归属训练答案。旧来源“每条射线完整生成基元集合”与新记录“已报告退出表面来源”不能悄悄等同。

现有 `v3_gse_synthetic_corrective_card_v1/v2` 校验器强绑定旧4观察/20帧/20无效位点修正，不适用于此次12观察/60帧全结果转换。不得修改旧card的数量来冒用，也不得将此数据导出改称infrastructure绕过预检。下一采用单独、精确限制的sensor转换卡校验入口，保留旧合同；绑定manifest、来源seal、代码与环境后正常preflight/create执行。

预期检查：准确60帧顺序、11520射线/帧、不删除或重复；未知整帧失败；float32存储误差单独报告；多来源代码往返一致；五帧student不含绝对pose或构造身份；失败不污染旧run/共享编码表。全部源记录和旧图片保留。
