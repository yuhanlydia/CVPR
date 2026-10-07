# 旧prepare缓存：单个EOF空行兼容修复

Web状态generated_unexecuted；没有访问GPU主机的旧receipt，没有跑测试/实验。
Local提供的观察是“看起来只有末尾换行不同”；实际legacy SHA仍需在真实receipt核对。
当前main准备脚本Git blob为8ca1a3d46c7e87059e0f738583dec1e51f7db4c3，
16089字节。Web只对源字节计算身份（不是运行项目）：
- 当前SHA256：ea6c615a51ee1e0f900146553c0c4372625cebe5360dd1e51e55a6b09c43443d。
- 去掉最后单个LF后16088字节的SHA256：
  94b2d4466cfc1c5f753d528aaa1a2c933d94c6f560a39e0cd0be97ec71b00928。

## 为什么可兼容，允许什么

原脚本已以LF结束，当前多出的唯一字节也是LF，即EOF的一个空行。
其余所有字节完全相同，不改变有效Python语句、参数、数据/模型、
投影/teacher或输出构造。该论证只覆盖明确这一对源文件，不是任意空白等价。

bundle.py保留严格的completed/exit0/prepare trial/上游身份/原layout/
manifest输出hash检查，以及validate_manifest全部cache refs/leakage checks。
prepare code ref要求恰好一个：
1. recorded SHA与当前文件SHA完全相等：照旧接受。
2. 否则只允许上述current/legacy SHA对，当前必须确实以两个LF结束，
   并重新计算current_bytes[:-1]的SHA等于原记录。记录policy与两个不同原始SHA。
3. 所有其他内容、空白、CRLF、第二个空行、缺失或重复code ref一律拒绝。

不把旧receipt里的hash改成新hash，不改旧manifest/cache，不冒称旧prepare用新版运行；
不strip/rstrip整个文件，不跳过producer检查，不自动重新跑prepare。
result.json和draft jobcards中的preparation_code_binding会明确记录
REVIEWED_SINGLE_EOF_BLANK_LINE或EXACT_SOURCE_BYTES、原/当前SHA与policy。
如果实际旧receipt不是94b2...，该修复不宣称解锁；Local返回精确SHA及实际源diff，
先证据审查，不再猜“只有换行”或降低guard。

## Local验收

合并本次修复到保留10个I09提交的隔离checkout，记录实际full SHA/dirty diff；
活动workspace和旧results不更新。所有源码/runtime/setup接受沿现有harness/pool/预算。
在已有获准的软件接受任务中（0 GPU、实际Conda解释器），工作目录为实际远端checkout：

~~~bash
"$NATIVE_PY" tools/qualify_prepare_compatibility.py --bundle "$AUDIT_BUNDLE" --out out/prepare-compatibility.json
~~~

这是源码/缓存接受，不是新增方法实验，也不授予额外额度。
工具读真实producer/manifest，验证实际cache refs，
软件性质检查exact、指定单LF可接受，语义变化/另一个空行/缺失/重复ref拒绝；
这些反例只是在内存测试source-binding函数，不生成或保存假的native receipt。
报告确认实际旧receipt未改变；保留真实stdout/stderr/exit/source/input/env refs。
即使PROPERTY_CHECKS_PASS，仍需后续原生评分/软件/G01/resource接受；
没有新的科学结果。异常也需真实harness stderr保留。

若当前checkout准备脚本有本地额外变化而不是上述current SHA，不覆盖本地改动；
核对actual diff与实际producer，未审查的版本不享有这一兼容例外。
兼容检查失败不代表方法失败。

## 预算仍单独阻塞

原18次方法/对照额度用完的事实保持。此次修复不清marker、不新建pool、
不重置累计时钟、不启动r004。缓存可复用后，Local估算完整第一批
（含同版本parent/强对照和三任务）的真实成本/有限新额度，提交给用户决定。
源码检查297文件、缓存兼容接受、native评分、方法实验分别报告完成情况。
