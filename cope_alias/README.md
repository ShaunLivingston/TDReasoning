# Optional evaluation aliases

`eval.py` looks here by default; override with `--alias_dir` or a specific
resource flag. These JSON files are local inputs and are excluded from Git.

| File | Expected content |
| --- | --- |
| `cwq_aname_dict.json` | Question string to list of answer names |
| `CWQ_aliase_data31158.json` | Answer name to list of aliases |
| `ComplexWebQuestions_test_wans.json` | List of objects with `question` and `answers`; each answer may contain `answer`, `answer_id`, and `aliases` |
| `WQSP_aliase_data.json` | Answer name to list of aliases |

CWQ uses the first three resources; WebQSP uses the fourth. GrailQA does not
use these alias resources. Missing files produce a notice and evaluation
continues without that expansion. Scores with and without aliases are not
directly interchangeable. Use resources corresponding to your experiment and
preserve upstream attribution and usage terms when redistributing them.
