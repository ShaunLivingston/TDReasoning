# TDReasoning

Research code for knowledge graph question answering over Freebase. The pipeline
breaks questions into sub-objectives, selects super-relations (`domain.type`) and
fine-grained relations, retrieves evidence triples, and reasons with an LLM.
It includes semantic retrieval, memory updates, optional reflection, and a direct
LLM fallback when graph reasoning does not produce a candidate answer.

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then load it explicitly:

```bash
set -a
source .env
set +a
```

Inference requires an OpenAI-compatible chat-completions API, a supported model
ID, and a running SPARQL endpoint serving Freebase RDF. Set `OPENAI_BASE_URL` for
a custom provider and `SPARQL_URL` for your endpoint (default:
`http://localhost:8890/sparql`). Model weights are downloaded on first semantic
retrieval; use `SBERT_MODEL` or `--sbert_path` to specify a local model directory.
The Freebase dump, endpoint setup, datasets, and model weights are not bundled.

## Data

Prepare preprocessed WebQSP, ComplexWebQuestions (CWQ), or GrailQA JSON files in
`data/`; see [data formats](data/README.md). Questions must include linked
`topic_entity` IDs to enable graph retrieval. Entity linking is not implemented.
Optional evaluation resources are described in [alias resources](cope_alias/README.md).
Local datasets and aliases are excluded from Git; obtain the appropriate data
and preserve its original attribution and usage terms.

## Run

Replace `YOUR_MODEL_ID` with the model identifier accepted by your provider:

```bash
python main_freebase_sampling.py \
  --dataset webqsp \
  --LLM_type YOUR_MODEL_ID \
  --limit 10 \
  --sample_seed 42 \
  --output_dir outputs/webqsp
```

Key options: `--depth 4`, `--width 10`, `--enable_reflection`,
`--sample_size N`, `--sample_seed SEED`, and `--data_dir PATH`.
The default limit is 1000; pass `--limit 0` to process all selected records.
Sampling happens before the limit is applied. `--max_length` controls planning
calls (capped at 512 tokens); reasoning calls use a 2,048-token limit.
Use `--help` for the complete argument list.

## Evaluate

```bash
python eval.py \
  --dataset webqsp \
  --output_file outputs/webqsp/PoG_SR_webqsp_YOUR_MODEL_ID.jsonl \
  --quiet
```

## Repository

- `main_freebase_sampling.py`: search and reasoning pipeline.
- `freebase_func.py`: Freebase SPARQL retrieval.
- `utils.py`: active prompts, LLM calls, retrieval, and reasoning utilities.
- `eval.py`: answer matching and aggregate statistics; standard library only.
- `tests/`: offline regression tests (`python -m unittest discover -s tests -v`).

