"""Run hierarchical relation-guided reasoning over a Freebase endpoint."""
from __future__ import annotations

import argparse
import os
import json
import re
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from freebase_func import build_triplets, fetch_relations, filter_relations_by_super, set_sparql_url
from utils import (
    LLMConfig,
    SuperRelationTable,
    direct_cot,
    ensure_dir,
    prepare_dataset,
    reasoning_with_llm,
    reflection_llm,
    save_jsonl,
    select_fine_relations_batch,
    select_super_relations,
    split_subobjectives,
    retrieve_top_docs,
    set_sbert_path,
    parse_llm_json,
    prune_triplets_by_information_gain,
    merge_ent_rel_dicts,
    score_candidate_answer  # [新增] 导入评分函数
)


# ============================================================================
# 辅助函数
# ============================================================================
def subgoal_coverage(memory_state: Dict[str, Any]) -> float:
    st = memory_state.get("sub_objective_status") or {}
    if not st:
        return 0.0
    ok = 0
    total = 0
    for _, v in st.items():
        total += 1
        s = str(v).lower()
        # 认为 Unknown 部分为空/none/n-a 才算完成
        if "unknown" not in s:
            ok += 1
        elif re.search(r"unknown:\s*(none|no|n/?a|nil|empty|-)\b", s):
            ok += 1
    return ok / max(total, 1)


def parse_reasoning_output(text: str):
    data = parse_llm_json(text)
    if not isinstance(data, dict):
        return "", "", False

    if "A" in data and isinstance(data["A"], dict):
        ans = data["A"].get("Answer", "")
        sufficient = data["A"].get("Sufficient", "")
    else:
        ans = data.get("Answer", "")
        sufficient = data.get("Sufficient", "")

    # 归一化处理
    ans_str = str(ans).strip()
    is_sufficient = str(sufficient).lower() in ["yes", "true", "1", "sufficient"]
    
    # 过滤无效答案
    if ans_str.lower() in ["none", "null", "unknown", "need more information", "i don't know"]:
        ans_str = ""
        
    return ans_str, str(data.get("R", "")), is_sufficient


def limit_relations_by_similarity(
    relations: Sequence[Dict[str, Any]],
    sub_objectives: Sequence[str],
    top_k: int
) -> List[Dict[str, Any]]:
    """
    使用 SBERT 过滤关系。
    [优化]：如果关系数量巨大，先进行随机采样，防止计算 SBERT 耗时过长。
    """
    total_rels = len(relations)
    if total_rels <= top_k:
        return list(relations)
    
    # 性能保护：SBERT 计算量随 N 线性增长，N>2000 时非常慢
    MAX_SBERT_INPUT = 2000
    
    if total_rels > MAX_SBERT_INPUT:
        # 随机采样一部分进行计算
        candidates_to_score = random.sample(list(relations), MAX_SBERT_INPUT)
    else:
        candidates_to_score = list(relations)
    
    names = [r["relation"] for r in candidates_to_score]
    # retrieve_top_docs 内部使用 SBERT
    top_names, _ = retrieve_top_docs(" ".join(sub_objectives), names, top_k=top_k)
    
    # === [修改策略] ===
    # 改为100% SBERT，0% random
    sbert_quota = top_k

    sbert_keep = set(top_names[:sbert_quota])

    final_keep = sbert_keep # 不加random

    return [r for r in relations if r["relation"] in final_keep]


def collect_next_frontier(ent_rel_ent_dict: Dict[str, Any], visited: set[str], width: int) -> List[str]:
    """
    Next frontier should only contain *expandable KB node ids*.
    We keep literals (numbers/dates/text) in triplets for reasoning, but we do NOT expand them.
    Also filter out CWQ internal question nodes like 'rabj/store/questions/...'.
    """
    import re

    def is_literal_like(x: str) -> bool:
        if not x:
            return True
        s = str(x).strip()
        # numbers
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return True
        # dates / datetimes
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", s):
            return True
        # spaces are usually literal text
        if " " in s:
            return True
        return False

    def is_bad_internal_node(s: str) -> bool:
        s = s.strip()
        if "store/questions" in s or s.startswith("rabj/store/"):
            return True
        return False

    def is_expandable_node_id(x: str) -> bool:
        if not x:
            return False
        s = str(x).strip()
        if is_literal_like(s):
            return False
        if s.startswith("http://") or s.startswith("https://"):
            return False
        if is_bad_internal_node(s):
            return False
        # Strong allow: standard mids/gids
        if re.fullmatch(r"[mg]\.[A-Za-z0-9_]+", s):
            return True
        # Allow key-like ids and slash ids (will be queried via full IRI safely)
        if re.fullmatch(r"[a-z][a-z0-9_]*\.[A-Za-z0-9_]+", s):
            return True
        if "/" in s:
            return True
        # Avoid expanding weird numeric-leading ids like "422$002F..." (often noisy) to save time
        if re.fullmatch(r"\d+.*", s):
            return False
        return True

    candidates: List[str] = []
    if not ent_rel_ent_dict:
        return []

    for _, dir_dict in ent_rel_ent_dict.items():
        for _, rel_dict in dir_dict.items():
            for _, ent_list in rel_dict.items():
                for eid in ent_list:
                    if not is_expandable_node_id(eid):
                        continue
                    if eid not in visited and eid not in candidates:
                        candidates.append(eid)

    return candidates[:width] if width > 0 else candidates

def confidence_based_fallback(
    current_relations: Sequence[Dict[str, Any]],
    selected_super: Sequence[str],
    super_table: "SuperRelationTable",
    width: int = 3,
) -> Tuple[str, List[str]]:
    pool = super_table.candidate_pool()
    if not pool:
        return "flat_relation_search", []

    used = set(selected_super)
    remaining = [sr for sr in pool if sr not in used]

    if remaining:
        return "super_relation_expansion", remaining[:width]
    else:
        return "flat_relation_search", []


# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="TDReasoning: Freebase knowledge graph question answering")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="grailqa")
    parser.add_argument("--data_dir", type=str, default=str(Path(__file__).resolve().parent / "data"))
    
    # Search
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=10)
    
    # LLM
    parser.add_argument("--max_length", type=int, default=768) 
    parser.add_argument("--temperature_exploration", type=float, default=0.1)
    parser.add_argument("--temperature_reasoning", type=float, default=0.0)
    parser.add_argument("--LLM_type", type=str, required=True, help="Model ID supported by your API provider.")
    parser.add_argument("--openai_api_keys", type=str, default="", help="Legacy option; prefer the OPENAI_API_KEY environment variable.")
    parser.add_argument("--api_base", type=str, default=os.environ.get("OPENAI_BASE_URL") or None, help="Optional OpenAI-compatible API base URL.")

    # Output & Config
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--sparql_url", type=str, default=os.environ.get("SPARQL_URL", "http://localhost:8890/sparql"))
    parser.add_argument("--enable_reflection", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sbert_path", type=str, default=os.environ.get("SBERT_MODEL", "sentence-transformers/msmarco-distilbert-base-tas-b"))
    parser.add_argument("--limit", type=int, default=500)

    parser.add_argument("--sample_size", type=int, default=None, help="Randomly sample subset of questions.")
    parser.add_argument("--sample_seed", type=int, default=None, help="Random seed for sampling.")

    args = parser.parse_args()

    random.seed(args.sample_seed)

    
    # ========================================================================
    # 初始化
    # ========================================================================
    api_key = args.openai_api_keys or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        parser.error("Set OPENAI_API_KEY before running inference.")
    
    from tqdm import tqdm

    base_url = args.api_base

    # 规划配置
    llm_plan = LLMConfig(
        engine=args.LLM_type, 
        temperature=args.temperature_exploration,
        max_tokens=min(args.max_length, 512), 
        api_key=api_key, 
        base_url=base_url
    )
    
    # 推理配置 (Token 需留足)
    reasoning_max_tokens = 2048 
    llm_reason = LLMConfig(
        engine=args.LLM_type, 
        temperature=args.temperature_reasoning,
        max_tokens=reasoning_max_tokens,
        api_key=api_key, 
        base_url=base_url
    )
    
    if args.verbose:
        print(f"LLM Config: Model={args.LLM_type}, Base={base_url}")

    if args.sparql_url: set_sparql_url(args.sparql_url)
    if args.sbert_path: set_sbert_path(args.sbert_path)
    
    try:
        datas, question_field = prepare_dataset(args.dataset, args.data_dir)
    except Exception as e:
        parser.error(f"Failed to load dataset: {e}")
    
    if args.sample_size is not None and args.sample_size > 0:
        if args.sample_size < len(datas):
            random.seed(args.sample_seed)
            datas = random.sample(datas, args.sample_size)
            print(f"Sampled {len(datas)} questions.")
    
    if args.limit and args.limit > 0:
        datas = datas[:args.limit]
    
    output_path = Path(args.output_dir)
    ensure_dir(output_path)
    output_file = output_path / f"PoG_SR_{args.dataset}_{args.LLM_type.replace('/', '_')}.jsonl"
    
    # ========================================================================
    # 主循环
    # ========================================================================
    
    for idx, data in enumerate(tqdm(datas, desc="Processing")):
        question = data[question_field]
        topic_entity = data.get("topic_entity", {})
        
        call_counter = {"count": 0}
        token_stats = {"total": 0, "input": 0, "output": 0}
        start_time = time.time()
        
        # Step 0: 子目标拆分
        sub_objectives = split_subobjectives(question, llm_plan, token_stats, call_counter)
        
        memory_state = {
            "sub_objectives": sub_objectives,
            "sub_objective_status": {},
            "super_relation_history": [],
            "key_entities": []
        }
        
        # [修改 1] 初始化候选答案池 (列表)，而非单个变量
        candidate_answer_pool = [] 

        # 如果无 Topic Entity，直接 CoT
        if not topic_entity:
            # 扁平化 cluster_triplets (虽然这里通常是空的，但保持一致性)
            all_found_triplets = [] 
            answer = direct_cot(question, llm_reason, token_stats, call_counter, triplets=all_found_triplets)
            record = {
                question_field: question, "results": parse_llm_json(answer),
                "reasoning_chains": [], "super_relation_history": [],
                "call_num": call_counter["count"], "token_usage": token_stats,
                "time": time.time() - start_time, "answer_source": "direct_cot"
            }
            save_jsonl(output_file, record)
            continue
        
        entid_name = {eid: name for eid, name in topic_entity.items()}
        name_entid = {name: eid for eid, name in topic_entity.items()}
        visited_entities = set(topic_entity.keys())
        frontier = list(topic_entity.keys())
        super_table = SuperRelationTable()
        cluster_triplets = []
        stop_flag = False
        flat_search_tried = False
        
        for depth in range(1, args.depth + 1):

            if args.verbose: print(f"Depth {depth}/{args.depth}, Frontier: {len(frontier)}")
            
            # Step 1: 获取关系
            current_relations = []
            for eid in frontier:
                # Skip CWQ internal "question nodes" (e.g., rabj/store/questions/...) to avoid wasting queries
                if isinstance(eid, str) and ("store/questions" in eid or eid.startswith("rabj/store/")):
                    continue
                rels = fetch_relations(eid, remove_unnecessary=True)
                super_table.ingest_relations([r["relation"] for r in rels])
                current_relations.extend(rels)

            # Step 1.1: 初始无关系检测
            if not current_relations:
                if not flat_search_tried:
                    flat_search_tried = True
                    current_relations = []
                    for eid in frontier:
                        current_relations.extend(
                            fetch_relations(eid, remove_unnecessary=False)
                        )
                    if not current_relations:
                        break
                else:
                    break

            # Step 2: 选择超关系
            # [修改 2] 修正 tried_sr 逻辑：只避免“直接上一轮”重复，而不是永久拉黑历史 SR
            tried_sr = []
            if depth > 1 and memory_state["super_relation_history"]:
                last_history = memory_state["super_relation_history"][-1]
                tried_sr = last_history.get("selected_super_relations", [])
            
            selected_super = select_super_relations(
                sub_objectives, super_table, llm_plan, token_stats,
                tried=tried_sr, top_k=20, call_counter=call_counter
            )

            # 并且如果 LLM 选的太少，强制补全
            if len(selected_super) < 2: 
                selected_super.extend(super_table.candidate_pool()[:2])

            if not selected_super: selected_super = super_table.candidate_pool()[:3]
            
            memory_state["super_relation_history"].append({
                "step": depth, "selected_super_relations": selected_super, "reason": "auto"
            })
            
            # Step 3: 过滤
            filtered = filter_relations_by_super(current_relations, selected_super)
            
            # Step 4: 细粒度选择
            relations_by_sr = {}
            for r in filtered:
                relations_by_sr.setdefault(r["super_relation"], []).append(r["relation"])
            
            fine_rels = select_fine_relations_batch(
                selected_super, sub_objectives, relations_by_sr,
                llm_plan, token_stats, max_per_sr=15, call_counter=call_counter  
            )    
            
            if fine_rels:
                filtered = [r for r in filtered if r["relation"] in fine_rels]
            elif len(filtered) > args.width* 4:
                # SBERT 限流
                filtered = limit_relations_by_similarity(filtered, sub_objectives, top_k=args.width* 4)
            
            # Step 5: 构建三元组
            triplets, ent_rel_ent_dict = build_triplets(filtered, entid_name, name_entid)
            triplets = prune_triplets_by_information_gain(question, sub_objectives, triplets, entid_name)
            
            # === Fallback 逻辑 ===
            if triplets:
                cluster_triplets.append(triplets)
                super_table.mark_success(selected_super)
            else:
                if args.verbose: print(f"Weak or no triplets ({len(triplets)}), triggering fallback...")
                super_table.mark_failed(selected_super)
                
                # 最后一层跳过耗时搜索
                if depth == args.depth:
                    if args.verbose: print("At max depth, skipping expensive flat search.")
                    strategy = "stop"
                else:
                    strategy, fallback_srs = confidence_based_fallback(
                        current_relations, selected_super, super_table, width=args.width
                    )
                
                if strategy == "stop":
                    break 
                
                if strategy == "super_relation_expansion" and fallback_srs:
                    new_relations = filter_relations_by_super(current_relations, fallback_srs)
                    triplets, ent_rel_ent_dict = build_triplets(new_relations, entid_name, name_entid)
                    triplets = prune_triplets_by_information_gain(question, sub_objectives, triplets, entid_name)
                    if triplets:
                        cluster_triplets.append(triplets)
                        super_table.mark_success(fallback_srs)
                    else:
                        super_table.mark_failed(fallback_srs)
                        
                elif strategy == "flat_relation_search":
                    # Fallback 时更加保守，减少 top_k 数量
                    fallback_width = args.width* 3
                    flat_rel = limit_relations_by_similarity(current_relations, sub_objectives, top_k=fallback_width)
                    triplets, ent_rel_ent_dict = build_triplets(flat_rel, entid_name, name_entid)
                    if triplets:
                        cluster_triplets.append(triplets)
            
            # Step 6: 联合记忆更新 + 推理
            current_step_triplets = triplets if triplets else []
            
            if current_step_triplets or depth == args.depth:
                
                result_text = reasoning_with_llm(
                    question, sub_objectives, memory_state, current_step_triplets,
                    llm_reason, token_stats, call_counter
                )
                ans, reason, sufficient = parse_reasoning_output(result_text)

                # [修改 3] 答案打分与入池
                if ans:
                    score = score_candidate_answer(
                        ans, 
                        sufficient, 
                        current_step_triplets, 
                        memory_state.get("key_entities", [])
                    )
                    
                    if args.verbose:
                        print(f"  --> Found Answer: '{ans}', Sufficient: {sufficient}, Score: {score}")

                    candidate_answer_pool.append({
                        "Answer": ans,
                        "R": reason,
                        "Sufficient": "Yes" if sufficient else "No",
                        "Score": score,
                        "Step": depth,
                        "final_memory": memory_state.copy(),
                        "reasoning_chains": list(cluster_triplets) # 浅拷贝当前链条状态
                    })

                cov = subgoal_coverage(memory_state)
                # conjunction/superlative/comparative 更严格，避免“只满足一半”就停
                strict_cov = 1.0 if any(k in question.lower() for k in [" and ", " both ", "least", "most", "largest", "smallest", "higher", "lower", "more than", "less than"]) else 0.8
                if ans and sufficient and score >= 2.5 and cov >= strict_cov:
                    stop_flag = True
                    break

            # Step 8: 反思 (Reflection)
            if args.enable_reflection and depth < args.depth:
                current_frontier_preview = list(entid_name.keys())[:15]
                reflection = reflection_llm(
                    question, sub_objectives, memory_state, current_frontier_preview,
                    llm_plan, token_stats, call_counter
                )
                
                action = reflection.get("action")
                
                if action == "backtrack":
                    backtrack_to = reflection.get("backtrack_to", [])
                    if backtrack_to:
                        new_frontier_ids = []
                        for n in backtrack_to:
                            if n in name_entid: new_frontier_ids.append(name_entid[n])
                        if new_frontier_ids:
                            frontier = new_frontier_ids
                            continue 
                
                elif action == "expand_super_relation":
                    tried_all = set()
                    for h in memory_state["super_relation_history"]:
                        tried_all.update(h.get("selected_super_relations", []))
                    
                    fallback_strategy_name, new_srs = confidence_based_fallback(
                        current_relations, list(tried_all), super_table, width=args.width
                    )
                    
                    if fallback_strategy_name == "super_relation_expansion" and new_srs:
                        new_filtered = filter_relations_by_super(current_relations, new_srs)
                        if len(new_filtered) > args.width* 5:
                            new_filtered = limit_relations_by_similarity(new_filtered, sub_objectives, top_k=args.width* 5)
                        
                        new_triplets, new_ent_dict = build_triplets(new_filtered, entid_name, name_entid)
                        new_triplets = prune_triplets_by_information_gain(question, sub_objectives, new_triplets, entid_name)
                        
                        if new_triplets:
                            cluster_triplets.append(new_triplets)
                            super_table.mark_success(new_srs)
                            merge_ent_rel_dicts(ent_rel_ent_dict, new_ent_dict)
                            memory_state["super_relation_history"].append({
                                "step": depth, "selected_super_relations": new_srs, "reason": "reflection_expansion"
                            })
                        else:
                            super_table.mark_failed(new_srs)
            
            # Step 9: 下一轮 Frontier
            next_frontier = collect_next_frontier(ent_rel_ent_dict, visited_entities, args.width)
            if not next_frontier:
                if args.verbose: print("No more frontier entities.")
                break
            visited_entities.update(next_frontier)
            frontier = next_frontier
        
        # [修改 4] 循环结束后的最终仲裁 (Final Arbitration)
        
        # 定义一个 helper function 来保存结果
        def save_record(final_ans_obj, source_tag):
            record = {
                question_field: question,
                "results": {"A": {"Answer": final_ans_obj["Answer"], "Sufficient": final_ans_obj["Sufficient"]}, "R": final_ans_obj["R"]},
                "reasoning_chains": final_ans_obj.get("reasoning_chains", cluster_triplets),
                "super_relation_history": memory_state["super_relation_history"],
                "call_num": call_counter["count"],
                "token_usage": token_stats,
                "time": time.time() - start_time,
                "answer_source": source_tag,
                "score": final_ans_obj.get("Score", 0)
            }
            save_jsonl(output_file, record)

        if candidate_answer_pool:
            # 按 Score 降序排列 (分数最高的在前面)
            candidate_answer_pool.sort(key=lambda x: x["Score"], reverse=True)
            best_candidate = candidate_answer_pool[0]
            
            if stop_flag:
                # 即使是因为 sufficient=True 退出的，我们也再次确认最高分（通常是一样的，因为+1加成）
                if args.verbose: print(f"Stopped early. Best Answer: {best_candidate['Answer']} (Score: {best_candidate['Score']})")
                save_record(best_candidate, "kg_reasoning_confident")
            else:
                # 遍历完深度后，选取最高分
                if args.verbose: print(f"Depth limit reached. Best Candidate: {best_candidate['Answer']} (Score: {best_candidate['Score']})")
                save_record(best_candidate, "kg_reasoning_best_scored")

        else:
            # 2. 实在没有 KG 答案，走 CoT Fallback
            if args.verbose: print("No KG candidate found. Running Direct CoT.")
            
            # 扁平化 cluster_triplets
            flat_triplets = [t for batch in cluster_triplets for t in batch]
            
            answer = direct_cot(question, llm_reason, token_stats, call_counter, triplets=flat_triplets)
            
            # 解析 CoT 结果以构建统一格式
            parsed_res = parse_llm_json(answer)
            if isinstance(parsed_res, dict) and "A" in parsed_res:
                ans_str = parsed_res["A"].get("Answer", "")
                suff_str = parsed_res["A"].get("Sufficient", "No")
                reason_str = parsed_res.get("R", "")
            else:
                ans_str, reason_str, suff_bool = parse_reasoning_output(answer)
                suff_str = "Yes" if suff_bool else "No"
                
            fallback_obj = {
                "Answer": ans_str,
                "Sufficient": suff_str,
                "R": reason_str,
                "Score": 0,
                "reasoning_chains": cluster_triplets
            }
            save_record(fallback_obj, "direct_cot_fallback")

    print(f"\nProcessing complete! Results: {output_file}")

if __name__ == "__main__":
    main()
