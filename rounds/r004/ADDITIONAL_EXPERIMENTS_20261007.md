# r004补充提案：从现有结果追问六个遗漏

## 最新状态与范围

2026-10-07检查：CVPR main仍为12a9252feaac60e69e39087b9d994f99a0162243；
Research Autopilot作者main仍为a8343aeb4f51303e2eb651081d4fe51c24c5ed3f。
上次交付之后没有新结果。此文件是proposed_unimplemented，所有执行/科学门均pending。
使用已核对的最新skill规则：每项数学分析、M路线现有方法审计、完整原生比较、
真实资源和剩余预算、Web/Local边界。没有重新凑20→15原创候选。

历史不是“只有两个idea”：20张历史卡中12个已实现原型；旧18臂运行完成17臂
（11原型+6控制），I01阻塞，另外8卡parked。I09/I13是较明显开发信号，
不意味着其他全部被证实失败；修复后的三任务结果仍缺失。
已有E01–E08、25新增配置及原18条全部保留；此处另提6个问题、29配置。
若未来实现并合并，两批共72条登记（含I01），不是72个原创idea，更不是72次已授权尝试。

## 数学共同定义与公平边界

使用冻结的真实训练包，q_i,c_j为PCA32单位向量，
P_i为native positives，N_i为allowed且非positive的候选。每个实际训练行必须
有正负例；不从未标注的全局候选池擅自造负例。
x_ij=vec(q_i(c_p-c_j)^T)/τ，w0=vec(I)，z=x^T w，
t=sigmoid(teacher_p-teacher_j)，τ=.1，λ=.02。
CE(z,t)=log(1+exp(z))-tz；锚定正则λ||w-w0||²/2。
梯度为Σa x(sigmoid(z)-t)+λ(w-w0)，
Hessian=Σa sigmoid(z)(1-sigmoid(z))xx^T+λI≽λI（a非负且归一）。
因此相应线性双线性头的目标强凸，但优化器到界或未收敛仍不得声称求得最优。
训练teacher是同一冻结2B的全维分数；不是更大reranker，不声称复现官方蒸馏训练。

### E09 配对顺序是否制造结论（5配置，优先1）

源代码pair_data使用第一个positive和第一个allowed negative。
这使学习依赖候选文件的次序；即使评估保留完整原生候选，训练信息可能只用任意一对。
新增：
1. B_PAIR_ALL：每行所有真实positive×native-negative组合，目标
   n^-1 Σ_i (|P_i||N_i|)^-1 Σ_pΣ_j CE(x_ipj^T w,t_ipj)+正则。
   每个query等权，避免候选多的行支配。梯度按相同权重累计。
2. B_PAIR_TEACHER_HARD：保持第一个positive，只选teacher分数最大的native negative；
   teacher相同分数按原候选次序固定tie-break，不看test。
3. B_PAIR_RANDOM_42/104729/314159：保持第一个positive，对native negatives均匀选一个。
   先按稳定native candidate ID排序再做固定种子选择，并保存选择digest。

必需对照A_CE_LBFGS；λ/τ、拟合约束、归一化、评估完全一致。
三个随机种子只代表训练pair抽样，不能声称独立数据集/encoder训练复现。
报告每行正负数、pair总数、抽样重合率、teacher margin和成本。
若全部仅有一正一负，多臂数学等价，应标为冗余条件，不把重复臂算独立证据。
all-pairs使用更多已有训练信息，不是纯优化器消融；主要内存按chunk流式梯度，
不用显式建立Σ|P||N|×1024矩阵。若native负例标注语义不支持硬负例先阻塞。

判定：同目标充分优化后仍对pair选择敏感，原I09相对CE控制结论需注明配对依赖；
若选择一致或排名不变，则该解释缺乏支持。hard-negative有假负例风险，保留原标签，
不得凭teacher修改ground truth。

### E10 teacher与真实标签的折中（3配置，优先2）

t_α=(1-α)+αt，α=.25,.5,.75；端点0=A_ERM_LBFGS，1=A_CE_LBFGS。
CE(z,t_α)=(1-α)CE(z,1)+αCE(z,t)，同一正则仅加一次；
沿用共同强凸性与解析梯度，不能把“混合后更好”自动归因于更强模型。
固定首pair以隔离target变化，不与E09同时改变配对。
输出逐group teacher margin<0比例、软target分布、训练loss/梯度及三任务paired差值。

假设：同2B teacher在PCA32重排上有不可靠或过软的监督，硬标签混合可能缓解。
否证：充分优化后各α不稳定或不优于两个端点/强基线。
这是开发敏感性网格，不能挑test最优α再在同一test宣称显著成功。
α选择需要独立前瞻确认。

### E11 训练来源对小头的贡献（6配置，优先5）

I09/I13/A_CE_LBFGS分别只用ScienceQA或只用A-OKVQA训练行，共6臂；
对照各自使用全部两来源的parent，同维PCA32、C_PROJECTED及强控制。
筛选rows后候选池为这些行allowed候选的并集；q、c、teacher、
positive/allowed mask同步重映射，不改标签、分母或候选含义。
否则I13仍会从另一来源的候选统计取信息，所谓source-only不成立。

重要范围：已冻结PCA仍由完整原两来源训练包拟合，encoder也不变。
所以这只分析head更新来源，不能声称整个pipeline完全未见另一来源，
不能称held-out-domain generalization。单来源行数也更少，来源与样本量混杂；
必须报告实际unique query/candidate数，参照E12，不能单独当纯来源因果证明。
I09分析所选pair Hessian/teacher冲突，I13分析所选候选统计量，
CE沿用共同凸性。若包没有可信source/row身份，阻塞，不按数组猜分组。

### E12 小头监督量的真实曲线（9配置，优先6）

I09/I13/A_CE_LBFGS × 每来源64/128/256 unique query。
按SHA256(seed42,group,native query_id)稳定排序，取嵌套前缀，
同一subset供三头共用；native query_id不足或冲突先阻塞。
候选池按E11 allowed并集缩减，只缩减训练，不缩减原生test。
完整parent约1019 unique，以manifest为准，不把“512 released行/来源”当512 unique。

它是head-supervision scaling；PCA已使用完整原始训练包，
不能用曲线宣称端到端64样本data efficiency或从零训练样本复杂度。
报告实际q/c/pair数及总encoder/PCA共享成本、head边际成本。
三头分别沿原数学定义，不通过更改λ或用test调参补救小样本。
嵌套样本引入相关性；一套seed只能是条件性开发曲线，不提供训练抽样方差。
否证：效果不随样本量稳定或全量仍不胜强控制，则更多head监督未解决主要瓶颈。

### E13 白化到底在估计谁的统计（2配置，优先4）

原I13对stack(q,unique c)等向量加权，query质量权重为n/(n+m)，
候选数多会让candidate分布支配；新增两臂固定原parent共同中心μ：
S_bal=.5 n^-1Σ(q_i-μ)(q_i-μ)^T+.5 m^-1Σ(c_j-μ)(c_j-μ)^T；
a_j=n^-1Σ_i 1[j∈P_i]/|P_i|，
S_pos=.5 n^-1Σ(q_i-μ)(q_i-μ)^T+.5Σ_j a_j(c_j-μ)(c_j-μ)^T。
Σa_j=1（每行至少一positive），两个S均PSD。
A=[(1-.2)S+.2 tr(S)/d I+.02 I]^-1/2；两侧映射后单位化，
与原I13、C_WHITEN、PCA32比较；固定λ、α、μ与cos。

μ固定后这是anchored second moment，不是绕新加权均值的协方差。
这样先隔离权重；不得偷偷重新估计μ再声称只有一个因子改变。
positive-weighted臂明确使用训练标签，不是无监督白化。
记录q/c权重、eigen谱、条件数、zero-norm及训练候选出现频率；
若收益只在一个任务或被MRL强基线超过，保留异质性和局限。
λ保证逆平方根可定义，不保证retrieval泛化。

### E14 I02保护约束是否压低可学容量（4配置，优先3）

protected rank=2/4/16，原I02 rank8；另固定seed42高斯矩阵QR得到
random orthogonal rank8作为容量匹配对照（生成算法参数，不生成评估数据）。
保持同一ScienceQA保护来源、MSE目标、λ和pair选择，
对照C_KD（无保护，rank0端点）、原I02、PCA32及强控制。

R为正交列，P=I-RR^T，M=I+PΔ；
优化||X vec(PΔ)-y||²/n+λ||PΔ||_F²。
用已有sample-space ridge求解，记录实际rank与R^T(M-I)残差。
对任意q在span(R)，q^T(M-I)c=0；但真实ScienceQA query通常不全在span(R)，
因此不能声称整个受保护任务预测/准确率保持不变。
报告保护query能量比例||R^Tq||²/||q||²、约束残差、train loss和全部task差值。

若random rank8与SVD rank8相当，优选子空间解释受削弱；
若减rank改善外部任务但损伤ScienceQA，说明真实保护/适应权衡；
若rank0仍差，则不能把失败仅归因保护约束。
随机QR需记录算法、NumPy版本、seed、R digest，不把一个seed当分布性证明。

## 完整G01提案与执行限制

每臂都覆盖ScienceQA/ChartQA/MSCOCO_i2t全部原生test、原候选/labels/次序、
官方scorer与原分母；沿用r004锁定模型/训练revision、leakage和native producer。
先同版本重放parent/required controls；不能拿旧不同代码的数字直接比较。
原E01全维2048和原生MRL前沿仍是表示强对照，不能只击败一个弱头。

primary：E09四种pair策略分别对首pairCE（3随机臂汇报分布，保留单臂）；
E10每α分别对软/硬两个端点；E11/E12每头各subset对同版本全量parent；
E13各权重对原I13；E14每rank/random对原I02及无保护KD。
每个对比×三任务，均预登记、全量返回；不挑表现最好的结果。
旧r004的required historical controls保留。每项同时报告点差、实际cluster数、
2000次image-SHA/query输入cluster bootstrap条件性95%CI，以及排名改变比例。
bootstrap仅描述固定candidate corpus下不确定性，不能以多重比较中的最优CI
作confirmatory显著性或科学PASS。无实际unit identity则CI pending。
缺raw/live replay、未收敛、单任务缺失、精度不足分别标记，不统一叫失败。

前置：先恢复修复后父方法结果和E01/E04强对照；本提案优先E09→E10→E14→E13，
E11/E12延后至source/unique query身份和成本合格。
29新增配置若全做是至少87个native group jobs，还需配套parent/control jobs，
不是29次已授权attempt；同参数重复fit不算独立seed。
不扩张旧18次/累计8h/prepare限制，不清锁、不重置计数，不启动并行harness。
现有CPU/GPU/RAM/剩余时间未知，dispatch_ready=false。
兼容缓存的小头拟合/评分请求0 GPU；新prepare需要独立真实1GPU准入且计入总预算。
all-pairs CPU成本O(pair_count*d²*iterations)，流式内存仍含原teacher/masks；
单来源/样本量改变会改变候选统计与fit成本，须实测peak RAM/时间。
Web只写分析与源，不运行项目/测试/模型，不下载数据；Local沿原Conda/SSH，无Docker。

当前43-arm registry不能接受B_*，新JSON是设计登记，不是可执行配置。
精确实现缺口、Local验收和交付见LOCAL_EXTENSION_HANDOFF.md。
不改变Parent/Natural Gate0/IPCG/Gate A状态，不承诺论文贡献。
