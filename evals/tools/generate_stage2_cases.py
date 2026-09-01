from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "evals" / "stage2"


def _write(name: str, payload: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def planning_cases() -> list[dict[str, Any]]:
    knowledge = [
        "什么是北极星指标？",
        "How does continuous discovery work?",
        "产品市场匹配通常有哪些判断信号？",
        "What is a retention cohort?",
        "机会解决方案树解决什么问题？",
        "How should a product team define activation?",
        "双钻模型包含哪些阶段？",
        "What makes an interview question leading?",
        "如何区分输出指标和结果指标？",
        "What is a product strategy?",
        "定价访谈应该避免哪些问法？",
        "How is churn different from contraction?",
        "什么是可逆决策？",
        "What does product-led growth mean?",
        "路线图和承诺计划有什么区别？",
    ]
    comparison = [
        "比较北极星指标和 OMTM 的使用场景。",
        "Compare continuous discovery with a fixed research sprint.",
        "产品愿景与产品战略有什么不同？",
        "Compare qualitative interviews and product analytics.",
        "机会解决方案树与普通功能清单有什么区别？",
        "Compare leading and lagging indicators.",
        "自下而上定价和基于价值定价有什么差别？",
        "Compare problem validation with solution validation.",
        "项目型路线图与结果型路线图如何取舍？",
        "Compare a beta launch with a full launch.",
        "用户分群与用户画像有什么不同？",
        "Compare an A/B test with a holdout test.",
        "增长团队和核心产品团队的目标如何区分？",
        "Compare a product review with a design critique.",
        "月留存与周留存分别适合什么产品？",
    ]
    scenario_missing = [
        "我们的激活率下降了，我该先做什么？",
        "We need to improve retention. What should the team prioritize?",
        "老板要求下季度增长 30%，产品该怎么拆？",
        "Our roadmap is overloaded. What should I cut?",
        "新功能上线后使用率很低，下一步怎么做？",
        "We are considering a price increase. How should we proceed?",
        "销售不断插需求，我应该如何排序？",
        "Our onboarding is too long. What should change first?",
        "团队对北极星指标有分歧，怎么推进？",
        "A competitor launched a similar feature. Should we respond?",
    ]
    scenario_complete = [
        "B2B 试用产品的新用户在连接数据源前流失；目标是本月提高首次连接率，工程只有一周，先查什么？",
        "Our weekly retention is stable, but activation fell after a required setup step was added. What should we test first?",
        "付费转化稳定，但高价值客户在续约前 30 天减少使用；我们能触达客户成功团队，如何诊断？",
        "We have 200 interview notes showing the same reporting pain and two engineers for a two-week prototype. What is a sensible next step?",
        "移动端结账完成率低于网页端，漏斗显示地址输入流失最多，本季度目标是提升成交，优先做什么？",
        "Our enterprise buyers request SSO, five signed deals are blocked, and security review takes six weeks. How should we sequence the work?",
        "内容产品首周留存下降，新增用户来源从自然搜索转为广告，产品体验没有改动，该怎么排查？",
        "A pricing test increased ARPU but reduced conversion; the goal is total gross profit and the sample is powered. How should we decide?",
        "团队每周交付功能但核心任务完成率不变，当前有可用性录像和事件数据，下一步怎么做？",
        "Our beta has 40 target users, 70% complete the core workflow, and support tickets cluster around permissions. What should the next sprint focus on?",
    ]
    followups = [
        ("那第二点具体怎么做？", "assistant: 先检查激活定义，再看分群差异。"),
        ("Can you turn that into a first-week checklist?", "assistant: Start with the activation funnel and interview five recent drop-offs."),
        ("如果工程只有三天呢？", "assistant: 建议先验证地址输入是否为主要摩擦点。"),
        ("Which assumption is riskiest?", "assistant: The recommendation depends on whether usage decline predicts renewal risk."),
        ("这个结论来自哪一类资料？", "assistant: The answer cited interview and analytics guidance."),
        ("能再比较一下第二种方法吗？", "assistant: 两种方法的主要差异是反馈频率。"),
        ("What would make you change that recommendation?", "assistant: Prioritize the permissions issue in the next sprint."),
        ("跳过追问，直接给建议。", "assistant: Which customer segment matters most?"),
        ("把建议缩成两步。", "assistant: 先分群，再验证关键摩擦。"),
        ("Does that still apply to a marketplace?", "assistant: Use a retention metric tied to durable customer value."),
    ]

    cases: list[dict[str, Any]] = []
    for intent, questions in (("knowledge", knowledge), ("comparison", comparison)):
        for index, question in enumerate(questions, 1):
            cases.append(
                {
                    "case_id": f"plan-{intent}-{index:02d}",
                    "question": question,
                    "conversation_context": "",
                    "completed_clarification_rounds": 0,
                    "skip_clarification": False,
                    "expected_intent": intent,
                    "expected_clarification": False,
                    "expected_source_scope": "broad",
                }
            )
    for index, question in enumerate(scenario_missing, 1):
        cases.append(
            {
                "case_id": f"plan-scenario-missing-{index:02d}",
                "question": question,
                "conversation_context": "",
                "completed_clarification_rounds": 0,
                "skip_clarification": False,
                "expected_intent": "scenario_advice",
                "expected_clarification": True,
                "expected_source_scope": "broad",
            }
        )
    for index, question in enumerate(scenario_complete, 1):
        cases.append(
            {
                "case_id": f"plan-scenario-complete-{index:02d}",
                "question": question,
                "conversation_context": "",
                "completed_clarification_rounds": 0,
                "skip_clarification": False,
                "expected_intent": "scenario_advice",
                "expected_clarification": False,
                "expected_source_scope": "broad",
            }
        )
    for index, (question, context) in enumerate(followups, 1):
        cases.append(
            {
                "case_id": f"plan-follow-up-{index:02d}",
                "question": question,
                "conversation_context": context,
                "completed_clarification_rounds": 1 if index == 8 else 0,
                "skip_clarification": index == 8,
                "expected_intent": "follow_up",
                "expected_clarification": False,
                "expected_source_scope": "broad",
            }
        )
    assert len(cases) == 60
    return cases


def claim_support_cases() -> list[dict[str, Any]]:
    scenarios = [
        ("activation", "new self-serve users", "review the setup funnel", "activation drops after setup"),
        ("retention", "weekly active teams", "compare cohorts by acquisition source", "retention differs across sources"),
        ("pricing", "prospective buyers", "test willingness to pay", "price sensitivity is uncertain"),
        ("interviews", "recently churned customers", "ask about the last real event", "the team needs behavioral detail"),
        ("roadmaps", "a platform team", "rank work by desired outcome", "the backlog is overloaded"),
        ("experiments", "mobile checkout users", "test the address-entry step", "the largest drop occurs there"),
        ("strategy", "an enterprise product", "name the target segment", "resources are constrained"),
        ("discovery", "a product trio", "interview customers every week", "the problem space changes quickly"),
        ("metrics", "a subscription product", "track durable customer value", "the team needs a north-star metric"),
        ("launch", "a closed beta", "resolve permission errors first", "support tickets cluster around permissions"),
        ("growth", "paid-acquisition users", "separate channel mix from product changes", "retention fell after channel mix shifted"),
        ("churn", "enterprise accounts", "review usage before renewal", "usage declines in the final 30 days"),
        ("onboarding", "invited workspace members", "measure first collaborative action", "solo completion is not the product value"),
        ("research", "buyers who rejected a proposal", "ask how the purchase decision was made", "the buying process is unclear"),
        ("prioritization", "a small engineering team", "limit work in progress", "five initiatives are running concurrently"),
        ("segmentation", "a two-sided marketplace", "measure each side separately", "supply and demand behave differently"),
        ("positioning", "security teams", "test the problem framing", "prospects misunderstand the category"),
        ("adoption", "existing administrators", "observe the first configuration task", "feature discovery is low"),
        ("quality", "high-volume API clients", "track error rate by endpoint", "aggregate reliability hides variation"),
        ("planning", "a quarterly product review", "state assumptions next to bets", "evidence strength varies"),
        ("analytics", "a new recommendation feature", "define an exposure event", "the denominator is ambiguous"),
        ("sales requests", "strategic accounts", "separate reusable needs from one-off requests", "custom work is increasing"),
        ("prototyping", "five design partners", "test the riskiest workflow first", "only two weeks are available"),
        ("localization", "new Japanese users", "review task completion by locale", "overall conversion masks a locale gap"),
        ("notifications", "weekly active managers", "measure action after notification", "open rate alone is not the outcome"),
    ]
    cases: list[dict[str, Any]] = []
    for scenario_index, (topic, segment, action, condition) in enumerate(scenarios, 1):
        evidence_id = f"synthetic:{topic}:{scenario_index:02d}"
        evidence = (
            f"For {segment}, the playbook recommends to {action} when {condition}. "
            "It describes this as a diagnostic step and explicitly does not claim "
            "that the step guarantees a business outcome."
        )
        variants = (
            ("supported", f"The playbook recommends that {segment} {action} when {condition}."),
            ("supported", f"The recommendation is scoped to {segment}."),
            ("supported", "The evidence does not say that the diagnostic step guarantees a business outcome."),
            ("insufficient_context", f"Every product team should always {action}."),
            ("insufficient_context", f"The step will increase revenue by 30 percent."),
            ("insufficient_context", f"The team must run the step for exactly 14 days."),
            ("contradicted", f"The playbook tells {segment} not to {action} when {condition}."),
            ("contradicted", "The evidence proves that the diagnostic step guarantees a business outcome."),
        )
        for variant_index, (verdict, claim) in enumerate(variants, 1):
            cases.append(
                {
                    "case_id": f"support-{scenario_index:02d}-{variant_index}",
                    "evidence": {"evidence_id": evidence_id, "text": evidence},
                    "claim": {
                        "claim_id": f"claim-{scenario_index:02d}-{variant_index}",
                        "text": claim,
                        "evidence_ids": [evidence_id],
                    },
                    "expected_verdict": verdict,
                }
            )
    assert len(cases) == 200
    return cases


def human_review_pack(claim_cases: list[dict[str, Any]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for verdict, count in (("supported", 8), ("insufficient_context", 6), ("contradicted", 6)):
        selected.extend(
            case
            for case in claim_cases
            if case["expected_verdict"] == verdict
        )
        selected = selected[: sum((8, 6, 6)[: ("supported", "insufficient_context", "contradicted").index(verdict) + 1])]
    assert len(selected) == 20
    claim_pairs = [
        {
            "review_id": f"human-{index:02d}",
            "case_id": case["case_id"],
            "evidence": case["evidence"],
            "claim": case["claim"],
            "human_verdict": None,
            "notes": "",
        }
        for index, case in enumerate(selected, 1)
    ]
    e2e = [
        ("e2e-01", "快速模式", "什么是北极星指标？", "资料结论可回溯，且没有无证据事实"),
        ("e2e-02", "深入模式", "比较北极星指标和 OMTM 的使用场景。", "跨文档比较覆盖主要差异"),
        ("e2e-03", "快速模式", "我们的激活率下降了，我该先做什么？", "只追问一个会改变建议的关键信息"),
        ("e2e-04", "快速模式", "跳过追问，基于现有信息继续。", "建议保留不确定性并标明依据类型"),
        ("e2e-05", "快速模式", "如果资料里没有答案，给我一套通用的 PM 复盘提纲。", "通用建议明确标注且不伪装成资料结论"),
        ("e2e-06", "快速模式", "资料中是否记录了 2030 年的公司收入？", "没有证据时拒绝编造事实"),
        ("e2e-07", "深入模式", "把上一条建议缩成两步，并说明各自依据。", "正确使用会话上下文和来源"),
        ("e2e-08", "深入模式", "给出一个结论、一个用户前提和一个可执行建议。", "三类内容在界面中清楚分开"),
    ]
    return {
        "schema_version": "stage2-human-review.v1",
        "status": "awaiting_human_review",
        "instructions": "请填写 human_verdict（supported、insufficient_context 或 contradicted）以及每个端到端案例的 rating（pass 或 fail）。全部完成并签名后，才能把界面校准状态改为 minimal。",
        "claim_pairs": claim_pairs,
        "e2e_cases": [
            {
                "review_id": case_id,
                "mode": mode,
                "question": question,
                "acceptance": acceptance,
                "rating": None,
                "notes": "",
            }
            for case_id, mode, question, acceptance in e2e
        ],
        "reviewer": "",
        "reviewed_at": None,
    }


def main() -> None:
    plan = planning_cases()
    support = claim_support_cases()
    _write(
        "question-planning.v1.json",
        {"schema_version": "stage2-question-planning.v1", "cases": plan},
    )
    _write(
        "claim-support.v1.json",
        {"schema_version": "stage2-claim-support.v1", "cases": support},
    )
    _write("human-review.v1.json", human_review_pack(support))


if __name__ == "__main__":
    main()
