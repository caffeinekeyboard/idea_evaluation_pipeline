# Idea Evaluation Pipeline

## Abstract
The Idea Evaluation Pipeline is a framework designed to automate the generation and quantitative evaluation of solutions to complex technical and engineering prompts. Utilizing Large Language Models (LLMs) and dense vector embeddings, the pipeline produces multiple candidate solutions for a given task and evaluates them across five principal dimensions: Novelty, Diversity Contribution, Depth, Utility, and Feasibility. 

## Pipeline Architecture

The system is bifurcated into a generation module and a tripartite evaluation module consisting of the Vector Engine, Logic Engine, and Judge Engine.

### 1. Idea Generation (`generation/generate_ideas.py`)
Candidate solutions are generated using a localized LLM via the `ollama` library. For a given prompt and context, the generation module samples $N$ ideas across a defined temperature band $[T_{min}, T_{max}]$. This ensures a spectrum of outputs ranging from highly deterministic solutions to highly exploratory ones. 

### 2. Evaluation Engines

#### A. Vector Engine (`engines/vector_engine.py`)
The Vector Engine maps textual ideas into a dense $d$-dimensional vector space using a pre-trained `SentenceTransformer` model (e.g., `BAAI/bge-large-en-v1.5`). Let $E = \{e_1, e_2, \dots, e_N\}$ denote the set of $L_2$-normalized embeddings for a batch of $N$ generated ideas, where $e_i \in \mathbb{R}^d$ and $\|e_i\|_2 = 1$.

* **Novelty:** The novelty of an idea $i$ is defined as its cosine distance from the centroid of the batch embeddings. Let the centroid be $C = \frac{1}{N} \sum_{j=1}^N e_j$, normalized such that $C \leftarrow \frac{C}{\|C\|_2}$. The novelty score $\mathcal{N}_i$ is computed as:
    $$\mathcal{N}_i = \max(0, 1 - e_i \cdot C^T)$$

* **Diversity Contribution:** The diversity contribution $\mathcal{D}_i$ of an idea $i$ represents its average distance from all other generated ideas within the same prompt batch:
    $$\mathcal{D}_i = \frac{1}{N-1} \sum_{j \neq i} \max(0, 1 - e_i \cdot e_j^T)$$

#### B. Logic Engine (`engines/logic_engine.py`)
The Logic Engine employs a rigorous, rule-based approach to constraint satisfaction.
1.  **Constraint Extraction:** Given a context $C_{text}$ and a task $T_{text}$, the LLM extracts a set of $K$ atomic, verifiable constraints $C = \{c_1, c_2, \dots, c_K\}$.
2.  **Depth Evaluation:** For a given idea, the engine acts as a boolean logic gate, evaluating the satisfaction of each constraint. Let $I_k \in \{0, 1\}$ be the indicator function returning $1$ if the idea satisfies constraint $c_k$, and $0$ otherwise. The depth score $\mathcal{Z}$ is computed as the ratio of satisfied constraints:
    $$\mathcal{Z} = \frac{1}{K} \sum_{k=1}^K I_k$$

#### C. Judge Engine (`engines/judge_engine.py`)
The Judge Engine utilizes a large language model (e.g., `gemma3:27b`) acting as an impartial adjudicator. It evaluates an idea against the context and task to produce:
1.  **Reasoning:** A chain-of-thought justification for the scores.
2.  **Raw Scores:** A utility score $S_u \in \{1, 2, 3, 4, 5\}$ and a feasibility score $S_f \in \{1, 2, 3, 4, 5\}$.

During the final pipeline execution (`main.py`), these scores are min-max normalized to the interval $[0, 1]$:
$$S_{norm} = \frac{S - 1}{5 - 1}$$

## Execution

**1. Idea Generation:**
Execute the generation script to produce candidate ideas from the input CSV of prompts.
```bash
python -m generation.generate_ideas --model <model_name> --input generation/prompts.csv --output_jsonl generation/outputs/generated_ideas.jsonl --n 5
```

### 2. Evaluation Pipeline:

```bash
python main.py
```

This script coordinates the Vector, Logic, and Judge engines, outputting the final multidimensional metrics to `evaluations.jsonl`.


## Dependencies
- `torch` 
- `numpy`
- `sentence_transformers`
- `ollama`
- `pydantic`
- `tqdm`

## License
This software is provided under the MIT License. See the `LICENSE` file for full details.