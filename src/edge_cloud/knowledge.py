"""Small demonstration knowledge base for the production-line use case.

These entries only drive the *mock* backends used by the local demo and tests.
Real deployments plug ``TransformersBackend`` / ``OpenAICompatibleBackend`` in
instead; the rest of the system (routing, confidence, flywheel) is unchanged.

Each entry models the intended split of labour:
- ``edge_answer``: what the small on-premise model can answer alone, often
  short or incomplete for long-tail questions.
- ``cloud_answer``: the large cloud model's enhanced answer.
- qualities: synthetic per-token probability used to exercise the confidence
  estimator deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KBEntry:
    key: str
    keywords: tuple[str, ...]
    task: str
    difficulty: str
    edge_answer: str
    cloud_answer: str
    edge_quality: float
    cloud_quality: float


ENTRIES: tuple[KBEntry, ...] = (
    KBEntry(
        key="reflow-profile",
        keywords=("回流焊", "峰值温度", "焊接温度", "温度曲线"),
        task="process_qa",
        difficulty="easy",
        edge_answer="回流焊峰值温度通常控制在 235–245℃，217℃ 以上维持 60–90 秒。",
        cloud_answer=(
            "回流焊峰值温度通常控制在 235–245℃，217℃ 以上维持 60–90 秒。"
            "实际设定还需结合锡膏厂商推荐曲线、PCB 厚度与器件热容量；"
            "首次导入建议用炉温测试仪实测并留存曲线。"
        ),
        edge_quality=0.94,
        cloud_quality=0.98,
    ),
    KBEntry(
        key="fault-e104",
        keywords=("E-104", "e-104", "真空压力", "吸嘴"),
        task="fault_diagnosis",
        difficulty="easy",
        edge_answer="E-104 表示贴片头真空压力不足：先检查真空管路是否漏气，再更换过滤棉。",
        cloud_answer=(
            "E-104 表示贴片头真空压力不足。处理顺序：1）检查吸嘴是否堵塞或磨损；"
            "2）检查真空管路接头与过滤棉；3）测量真空压力是否达到设备规格（通常 -60 kPa 以上）；"
            "4）以上正常仍报警，检查真空传感器零点。"
        ),
        edge_quality=0.91,
        cloud_quality=0.97,
    ),
    KBEntry(
        key="solder-void",
        keywords=("空洞", "焊点", "25%", "返修"),
        task="quality_verdict",
        difficulty="easy",
        edge_answer="判定：返修。焊点空洞面积超过 25% 上限，不满足放行条件。",
        cloud_answer=(
            "判定：返修。依据 IPC-A-610 Class 2，焊点空洞总面积不应超过焊点面积的 25%；"
            "若空洞集中在界面处或影响长期可靠性，即使未超限也应升级评审。"
        ),
        edge_quality=0.9,
        cloud_quality=0.97,
    ),
    KBEntry(
        key="ipc-class3",
        keywords=("Class 3", "class 3", "虚焊", "IPC-A-610"),
        task="standard_qa",
        difficulty="hard",
        edge_answer="Class 3 对虚焊要求更严格，通常不允许影响可靠性的虚焊。",
        cloud_answer=(
            "IPC-A-610 Class 3 要求焊点形成连续、完整的润湿，不允许未润湿、退润湿或明显虚焊；"
            "对空洞的限制也更严：通常焊点空洞总面积不超过 10%，且不得位于焊点与引脚界面。"
            "Class 3 产品还需按抽样方案做 X-Ray 与切片验证。"
        ),
        edge_quality=0.42,
        cloud_quality=0.96,
    ),
    KBEntry(
        key="dppm-calc",
        keywords=("DPPM", "dppm", "200 个焊点", "6 个缺陷"),
        task="process_calc",
        difficulty="hard",
        edge_answer="约为 3000 DPPM。",
        cloud_answer=(
            "30000 DPPM。计算：6 ÷ 200 × 1,000,000 = 30,000 DPPM。"
            "如果 6 个缺陷分布在 200 个产品上而不是 200 个焊点上，口径不同，需先确认分母。"
        ),
        edge_quality=0.3,
        cloud_quality=0.98,
    ),
    KBEntry(
        key="fault-e217",
        keywords=("E-217", "e-217", "主轴", "冷却液"),
        task="fault_diagnosis",
        difficulty="hard",
        edge_answer="E-217 建议先重启设备，观察是否恢复。",
        cloud_answer=(
            "E-217 表示主轴冷却液流量不足，不应直接重启。处理：1）检查冷却液液位与管路折弯；"
            "2）清洗流量传感器滤网；3）确认泵体是否气蚀；4）恢复流量后再复位报警，并记录主轴温度趋势。"
        ),
        edge_quality=0.34,
        cloud_quality=0.95,
    ),
    KBEntry(
        key="oee",
        keywords=("OEE", "oee", "92%", "设备综合效率"),
        task="improvement",
        difficulty="hard",
        edge_answer="建议先从设备点检和员工培训入手，逐步提升 OEE。",
        cloud_answer=(
            "按 OEE = 可用率 × 性能 × 良率拆解：先测出三类损失各自占比，再选损失最大的单项做改善。"
            "例如换型时间过长优先做 SMED；小停机频繁先做 Andon 与点检；良率损失则回到工艺窗口和来料。"
            "目标 92% 通常需要可用率 ≥95%、性能 ≥97%、良率 ≥99.5%，建议先设定 4 周基线再定改善项目。"
        ),
        edge_quality=0.38,
        cloud_quality=0.93,
    ),
    KBEntry(
        key="estop-reset",
        keywords=("急停", "复位", "安全"),
        task="safety",
        difficulty="easy",
        edge_answer="先确认危险源已排除、人员撤离，再按急停复位按钮并保持 3 秒，然后重新启动。",
        cloud_answer=(
            "先确认危险源已排除、人员撤离并完成上锁挂牌确认，再按急停复位按钮并保持 3 秒，"
            "然后按启动流程复位安全回路。若复位后仍无法启动，检查安全门与光栅回路，禁止短接安全信号。"
        ),
        edge_quality=0.93,
        cloud_quality=0.98,
    ),
)

DEFAULT_ENTRY = KBEntry(
    key="unknown",
    keywords=(),
    task="general",
    difficulty="unknown",
    edge_answer="端侧知识库未覆盖该问题，建议转人工或请求云端增强。",
    cloud_answer=(
        "该问题超出端侧知识库范围。云端建议：先补充设备型号、工艺参数与报警上下文，"
        "再按“现象—数据—可能原因—验证步骤”给出结论；无法确认时转交工艺工程师。"
    ),
    edge_quality=0.4,
    cloud_quality=0.82,
)


def lookup(prompt: str) -> KBEntry:
    """Return the best matching entry by keyword hits (longest keyword wins ties)."""

    lowered = prompt.lower()
    best: KBEntry | None = None
    best_score = 0
    best_keyword_len = 0
    for entry in ENTRIES:
        hits = [kw for kw in entry.keywords if kw.lower() in lowered]
        score = len(hits)
        keyword_len = max((len(kw) for kw in hits), default=0)
        if score > best_score or (score == best_score and score > 0 and keyword_len > best_keyword_len):
            best = entry
            best_score = score
            best_keyword_len = keyword_len
    return best or DEFAULT_ENTRY
