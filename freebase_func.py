"""Freebase SPARQL retrieval with full IRIs and literal-aware triplet handling."""
from __future__ import annotations

import re
import time
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import quote

from utils import normalize_super_relation

SPARQL_URL = "http://localhost:8890/sparql"


def set_sparql_url(url: str) -> None:
    global SPARQL_URL
    SPARQL_URL = url
    print(f"SPARQL URL set to: {url}")


# -----------------------------------------------------------------------------
# ID / IRI helpers
# -----------------------------------------------------------------------------
_MID_RE = re.compile(r"^[mg]\.[A-Za-z0-9_]+$")
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*\.[A-Za-z0-9_]+$")


def is_mid_or_gid(x: str) -> bool:
    return bool(_MID_RE.fullmatch(str(x)))


def is_kb_node_id(x: str) -> bool:
    s = str(x).strip()
    if not s:
        return False
    if s.startswith("http://") or s.startswith("https://"):
        return True
    if " " in s or "\t" in s or "\n" in s:
        return False
    return bool(_MID_RE.fullmatch(s) or _KEY_RE.fullmatch(s) or ("/" in s))


def ns_iri(x: str) -> str:
    s = str(x).strip()
    if s.startswith("<") and s.endswith(">"):
        return s
    if s.startswith("http://") or s.startswith("https://"):
        return f"<{s}>"
    enc = quote(s, safe="/._-$:")
    return f"<http://rdf.freebase.com/ns/{enc}>"


# -----------------------------------------------------------------------------
# SPARQL templates (placeholders expect *full IRI strings* like <http://...>)
# -----------------------------------------------------------------------------
sparql_head_relations = """
SELECT DISTINCT ?relation
WHERE {
  %s ?relation ?x .
}
LIMIT 2000
"""

sparql_tail_relations = """
SELECT DISTINCT ?relation
WHERE {
  ?x ?relation %s .
}
LIMIT 2000
"""

sparql_tail_entities_extract = """
SELECT ?tailEntity
WHERE {
  %s %s ?tailEntity .
}
LIMIT 300
"""

sparql_head_entities_extract = """
SELECT ?tailEntity
WHERE {
  ?tailEntity %s %s .
}
LIMIT 300
"""

sparql_batch_names = """
PREFIX ns: <http://rdf.freebase.com/ns/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?entity ?name
WHERE {
  FILTER (?entity IN (%s))
  {
    ?entity ns:type.object.name ?name .
    FILTER (lang(?name) = "en")
  }
  UNION
  {
    ?entity rdfs:label ?name .
    FILTER (lang(?name) = "en")
  }
  UNION
  {
    ?entity ns:type.object.name ?name .
  }
}
"""

sparql_id = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?tailEntity
WHERE {
  {
    ?entity ns:type.object.name ?tailEntity .
    FILTER(?entity = %s)
  }
  UNION
  {
    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .
    FILTER(?entity = %s)
  }
}
LIMIT 50
"""


# -----------------------------------------------------------------------------
# SPARQL executor
# -----------------------------------------------------------------------------
def execute_sparql(sparql_query: str, max_retries: int = 3):
    from SPARQLWrapper import JSON, SPARQLWrapper

    sparql = SPARQLWrapper(SPARQL_URL)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    for attempt in range(max_retries):
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"SPARQL query failed after {max_retries} attempts: {e}\n\nSPARQL query:\n{sparql_query}")
                return []
    return []


def replace_relation_prefix(relations):
    return [
        relation["relation"]["value"].replace("http://rdf.freebase.com/ns/", "")
        for relation in relations
    ]


def replace_entities_prefix(entities):
    out = []
    for ent in entities:
        v = ent["tailEntity"]["value"]
        out.append(v.replace("http://rdf.freebase.com/ns/", ""))
    return out


def abandon_rels(relation: str) -> bool:
    if relation in {"type.object.type", "type.object.name"}:
        return True
    if relation.startswith("common.") or relation.startswith("freebase."):
        return True
    if "sameAs" in relation:
        return True
    return False


def id2entity_name_or_type(entity_id: str) -> str:
    if not is_kb_node_id(entity_id):
        return "UnName_Entity"
    q = sparql_id % (ns_iri(entity_id), ns_iri(entity_id))
    results = execute_sparql(q)
    if not results:
        return "UnName_Entity"
    return results[0]["tailEntity"]["value"]


def fetch_relations(entity_id: str, remove_unnecessary: bool = True) -> List[Dict]:
    relations: List[Dict] = []
    if not is_kb_node_id(entity_id):
        return relations

    eiri = ns_iri(entity_id)

    head_rel = replace_relation_prefix(execute_sparql(sparql_head_relations % eiri))
    tail_rel = replace_relation_prefix(execute_sparql(sparql_tail_relations % eiri))

    if remove_unnecessary:
        head_rel = [r for r in head_rel if not abandon_rels(r)]
        tail_rel = [r for r in tail_rel if not abandon_rels(r)]

    for rel in set(head_rel):
        relations.append({"entity": entity_id, "relation": rel, "head": True, "super_relation": normalize_super_relation(rel)})

    for rel in set(tail_rel):
        relations.append({"entity": entity_id, "relation": rel, "head": False, "super_relation": normalize_super_relation(rel)})

    return relations


def filter_relations_by_super(relations: Sequence[Dict], allowed_super: Iterable[str]) -> List[Dict]:
    allowed = set(allowed_super)
    return [r for r in relations if r.get("super_relation") in allowed]


def entity_search(entity_id: str, relation: str, head: bool = True) -> List[str]:
    if not is_kb_node_id(entity_id):
        return []
    eiri = ns_iri(entity_id)
    riri = ns_iri(relation)

    if head:
        q = sparql_tail_entities_extract % (eiri, riri)
    else:
        q = sparql_head_entities_extract % (riri, eiri)

    entities = execute_sparql(q)
    return replace_entities_prefix(entities)


def detect_is_english(text: str) -> bool:
    try:
        text.encode("utf-8").decode("ascii")
        return True
    except UnicodeDecodeError:
        return False


def provide_triple(entity_ids: Sequence[str], relation: str) -> Tuple[List[str], List[str]]:
    if not entity_ids:
        return [], []

    LIMIT_COUNT = 60
    target_ids = list(entity_ids)[:LIMIT_COUNT]

    kb_ids = [eid for eid in target_ids if is_kb_node_id(eid)]
    id2name: Dict[str, str] = {}

    if kb_ids:
        sparql_ids = ", ".join([ns_iri(eid) for eid in kb_ids])
        query = sparql_batch_names % sparql_ids
        results = execute_sparql(query)

        for item in results:
            eid_uri = item["entity"]["value"]
            eid = eid_uri.replace("http://rdf.freebase.com/ns/", "")
            name = item.get("name", {}).get("value", "")
            if eid not in id2name or (name and detect_is_english(name) and not detect_is_english(id2name.get(eid, ""))):
                id2name[eid] = name

    final_names, final_ids = [], []
    for eid in target_ids:
        if is_kb_node_id(eid):
            name = id2name.get(str(eid))
            if not name:
                name = f"Entity:{eid}"
            final_ids.append(str(eid))
            final_names.append(name)
        else:
            final_ids.append(str(eid))
            final_names.append(str(eid))

    return final_names, final_ids


def build_triplets(
    current_entity_relations: Sequence[Dict],
    entid_name: Dict[str, str],
    name_entid: Dict[str, str],
) -> Tuple[List[Tuple[str, str, str]], Dict]:
    triplets: List[Tuple[str, str, str]] = []
    ent_rel_ent_dict: Dict = {}

    for ent_rel in current_entity_relations:
        ent = ent_rel["entity"]
        rel = ent_rel["relation"]
        head_flag = ent_rel["head"]

        candidates_id = entity_search(ent, rel, head=head_flag)
        if not candidates_id:
            continue

        if len(candidates_id) > 50:
            candidates_id = candidates_id[:50]

        candidate_names, candidates_id = provide_triple(candidates_id, rel)

        # Update maps ONLY for KB nodes
        for cid, cname in zip(candidates_id, candidate_names):
            if is_kb_node_id(cid):
                entid_name[cid] = cname
                name_entid[cname] = cid

        dir_key = "head" if head_flag else "tail"
        ent_rel_ent_dict.setdefault(ent, {}).setdefault(dir_key, {}).setdefault(rel, [])

        for cid in candidates_id:
            if cid not in ent_rel_ent_dict[ent][dir_key][rel]:
                ent_rel_ent_dict[ent][dir_key][rel].append(cid)

        entity_name = entid_name.get(ent, ent)
        for cname in candidate_names:
            if head_flag:
                triplets.append((entity_name, rel, cname))
            else:
                triplets.append((cname, rel, entity_name))

    return triplets, ent_rel_ent_dict


def get_relation_count(entity_id: str) -> int:
    return len(fetch_relations(entity_id, remove_unnecessary=False))


def print_entity_info(entity_id: str) -> None:
    name = id2entity_name_or_type(entity_id)
    relations = fetch_relations(entity_id, remove_unnecessary=True)

    print(f"\nEntity: {entity_id}")
    print(f"Name: {name}")
    print(f"Relations: {len(relations)}")
    super_rel_counts = {}
    for rel in relations:
        sr = rel["super_relation"]
        super_rel_counts[sr] = super_rel_counts.get(sr, 0) + 1
    print("\nSuper-relations:")
    for sr, count in sorted(super_rel_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sr}: {count} relations")
