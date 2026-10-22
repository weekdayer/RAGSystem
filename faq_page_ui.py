"""独立的 FAQ 管理页面；界面只依赖 FAQ 服务的公开接口。"""

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent


def _render_management(faq, show_error) -> None:
    if st.button("加载客服示例"):
        try:
            result = faq.import_json((ROOT / "examples" / "faq.json").read_bytes())
            st.success(f"新增 {result['added']} 条，跳过 {result['skipped']} 条重复问答。")
        except Exception as error:
            show_error(error, "示例加载失败")

    st.subheader("添加标准问答")
    with st.form("faq_add_form"):
        question = st.text_input("标准问题", max_chars=4000)
        answer = st.text_area("标准答案", max_chars=12000)
        aliases = st.text_area("相似问法（可选，每行一个）")
        submitted = st.form_submit_button("添加 FAQ", type="primary")
    if submitted:
        try:
            faq.add(question, answer, [line.strip() for line in aliases.splitlines() if line.strip()])
            st.success("FAQ 已保存，可以回到聊天页面提问。")
        except Exception as error:
            show_error(error, "FAQ 保存失败")

    with st.expander("批量导入"):
        st.caption("JSON 数组格式，每条包含 question、answer，可选 aliases。")
        upload = st.file_uploader(
            "导入 FAQ（JSON，最多 1 MiB）", type=["json"], key="faq_upload"
        )
        if st.button("导入 FAQ", disabled=upload is None):
            try:
                if upload.size > 1024 * 1024:
                    raise ValueError("FAQ 文件最多 1 MiB。")
                result = faq.import_json(upload.getvalue())
                st.success(f"新增 {result['added']} 条，跳过 {result['skipped']} 条重复问答。")
            except Exception as error:
                show_error(error, "FAQ 导入失败")

    st.subheader("已有问答")
    try:
        items = faq.list()
        st.caption(f"已保存 {len(items)} 条 FAQ")
        if not items:
            st.info("可以先加载客服示例，也可以添加自己的问答。")
        for item in items:
            with st.expander(item["question"]):
                st.markdown(item["answer"])
                alias_draft = st.text_area(
                    "补充相似问法（每行一个）", key=f"faq_aliases_{item['id']}"
                )
                if st.button("保存相似问法", key=f"save_faq_aliases_{item['id']}"):
                    try:
                        new_aliases = [line.strip() for line in alias_draft.splitlines() if line.strip()]
                        if not new_aliases:
                            raise ValueError("请先填写需要补充的相似问法。")
                        item = faq.add_aliases(item["id"], new_aliases)
                        st.success("相似问法已保存，后续提问立即生效。")
                    except Exception as error:
                        show_error(error, "相似问法保存失败")
                if item["aliases"]:
                    st.caption("相似问法：" + "；".join(item["aliases"]))
                if st.button("删除这条 FAQ", key=f"delete_faq_{item['id']}"):
                    faq.delete(item["id"])
                    st.rerun()
    except Exception as error:
        show_error(error, "FAQ 暂不可用")


def _render_settings(faq, show_error) -> None:
    try:
        settings = faq.get_settings()
    except Exception as error:
        show_error(error, "FAQ 设置读取失败")
        return

    st.caption("调整语义门槛，控制哪些问题直接使用 FAQ 标准答案。")
    with st.form("faq_matching_settings"):
        threshold_points = st.number_input(
            "FAQ 匹配门槛（0–100 分）", min_value=0, max_value=100,
            value=round(settings["threshold"] * 100), step=1,
            help="达到语义门槛后，候选还需要通过领先分检查。",
        )
        with st.expander("高级设置"):
            margin_points = st.number_input(
                "候选领先分（0–100 分）", min_value=0, max_value=100,
                value=round(settings["min_margin"] * 100), step=1,
                help="第一名和第二名的混合分差距过小时，继续原有 RAG 流程。",
            )
        saved = st.form_submit_button("保存匹配设置", type="primary")
    if saved:
        try:
            settings = faq.update_settings(
                threshold=threshold_points / 100, min_margin=margin_points / 100
            )
            st.success("匹配设置已保存，后续提问立即生效。")
        except Exception as error:
            show_error(error, "FAQ 设置保存失败")

    st.info(
        "70 分表示语义相关度达到 0.70，不代表答案有 70% 的正确率。"
        "BM25 关键词分只用于候选排序，不能绕过语义门槛。"
        "标准问题或已保存的相似问法精确命中时，直接返回标准答案。"
    )
    st.caption(
        f"当前生效：语义门槛 {settings['threshold'] * 100:g} 分，"
        f"候选领先分 {settings['min_margin'] * 100:g} 分。"
    )


def _points(value):
    if value is None:
        return None
    return round(max(0.0, min(1.0, float(value))) * 100, 1)


def _render_preview(faq, show_error) -> None:
    st.caption("只检查 FAQ 匹配过程，不生成模型回答。未命中时，正式聊天会继续原有 RAG 流程。")
    with st.form("faq_match_preview"):
        question = st.text_input("测试问题", max_chars=4000)
        submitted = st.form_submit_button("测试匹配", type="primary")
    if not submitted:
        return
    try:
        with st.spinner("正在检查 FAQ 匹配…"):
            result = faq.preview(question)
    except Exception as error:
        show_error(error, "FAQ 匹配测试失败")
        return

    match = result.get("match")
    if result["matched"] and match:
        method = "精确匹配" if match["match_method"] == "exact" else "语义与关键词匹配"
        st.success(f"命中 FAQ（{method}）")
        st.markdown(f"**标准问题：** {match['question']}")
        st.markdown(match["answer"])
    else:
        reasons = {
            "empty": "还没有 FAQ。",
            "below_threshold": "语义分未达到门槛。",
            "ambiguous": "候选得分过于接近，无法确定标准答案。",
            "no_candidates": "没有找到可用候选。",
        }
        st.info("未命中 FAQ，将继续 RAG / 模型回答。" + reasons.get(result["reason"], ""))
    st.caption(
        f"本次设置：语义门槛 {result['threshold'] * 100:g} 分，"
        f"候选领先分 {result['min_margin'] * 100:g} 分。"
    )
    st.caption("语义分与混合分不是正确率；负相关按 0 分显示。")
    candidates = result.get("candidates", [])
    if candidates:
        st.dataframe(
            [
                {
                    "候选问题": candidate["question"],
                    "语义分（0–100）": _points(candidate["semantic_score"]),
                    "BM25 原始分": round(float(candidate["bm25_score"]), 4),
                    "混合分（0–100）": _points(candidate["hybrid_score"]),
                }
                for candidate in candidates
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.caption("精确命中或没有候选时，无需展示候选排序。")


def render_faq_page(service, show_error) -> None:
    """在主内容区域展示 FAQ 管理，不改变聊天历史或连接设置。"""
    st.title("FAQ 管理")
    st.caption("维护标准答案、调整匹配门槛，并测试不同问法。")
    management, settings, preview = st.tabs(["问答管理", "匹配设置", "匹配测试"])
    with management:
        _render_management(service.faq, show_error)
    with settings:
        _render_settings(service.faq, show_error)
    with preview:
        _render_preview(service.faq, show_error)
