import json
import csv
import torch
from pathlib import Path
from tqdm import tqdm

from engines.vector_engine import VectorEngine
from engines.logic_engine import LogicEngine
from engines.judge_engine import JudgeEngine

def load_prompts(csv_path: str) -> dict:
    prompts = {}
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts[row['prompt_id']] = {
                "task": row['prompt'],
                "context": row['context'],
                "umbrella": row['umbrella']
            }
    return prompts

def load_generated_ideas(jsonl_path: str) -> dict:
    grouped_ideas = {}
    with open(jsonl_path, mode='r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            pid = data['prompt_id']
            if pid not in grouped_ideas:
                grouped_ideas[pid] = []
            grouped_ideas[pid].append(data)
    return grouped_ideas

def normalize_score(score: int, min_val: int = 1, max_val: int = 5) -> float:
    score = max(min_val, min(score, max_val))
    return float((score - min_val) / (max_val - min_val))

def run_pipeline(prompts_csv: str, ideas_jsonl: str, output_jsonl: str):
    print("Loading data...")
    prompts = load_prompts(prompts_csv)
    grouped_ideas = load_generated_ideas(ideas_jsonl)
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    print("\nInitializing Engines...")
    vector_engine = VectorEngine(device="cuda")
    logic_engine = LogicEngine(model_name="gemma:instruct")
    judge_engine = JudgeEngine(model_name="gemma:instruct")
    
    with open(output_jsonl, mode='a', encoding='utf-8') as out_f:
        
        for prompt_id, ideas in tqdm(grouped_ideas.items(), desc="Processing Prompts"):
            if prompt_id not in prompts:
                continue
                
            prompt_data = prompts[prompt_id]
            context = prompt_data['context']
            task = prompt_data['task']
            print(f"\n--- Extracting Constraints for Prompt {prompt_id} ---")
            constraints = logic_engine.extract_constraints(context, task)
            idea_texts = [idea['idea_text'] for idea in ideas]
            
            if len(idea_texts) > 0:
                embeddings = vector_engine.embed(idea_texts)
                sim_matrix = torch.mm(embeddings, embeddings.T)
                dist_matrix = 1.0 - sim_matrix
            
            for i, idea in enumerate(ideas):
                idea_text = idea['idea_text']
                idea_id = idea['idea_id']
                
                if len(idea_texts) > 1:
                    novelty = vector_engine.calculate_novelty(idea_text, embeddings)
                    div_contrib = (torch.sum(dist_matrix[i]) / (len(idea_texts) - 1)).item()
                    
                else:
                    novelty = 0.0
                    div_contrib = 0.0
                
                depth_score, _ = logic_engine.evaluate_depth(idea_text, constraints)
                judge_result = judge_engine.evaluate_quality(context, task, idea_text)
                utility_norm = normalize_score(judge_result.get('utility', 0))
                feasibility_norm = normalize_score(judge_result.get('feasibility', 0))
                final_payload = {
                    "idea_id": idea_id,
                    "metrics": {
                        "novelty": round(max(0.0, float(novelty)), 4),
                        "diversity_contribution": round(max(0.0, float(div_contrib)), 4),
                        "depth": round(depth_score, 4),
                        "utility": round(utility_norm, 4),
                        "feasibility": round(feasibility_norm, 4)
                    },
                    "metadata": {
                        "judge_reasoning": judge_result.get('reasoning', ''),
                        "prompt_id": prompt_id,
                        "temperature_used": idea.get('temperature', 0.0)
                    }
                }
                out_f.write(json.dumps(final_payload) + "\n")
                out_f.flush()

if __name__ == "__main__":
    PROMPTS_CSV = "generation/prompts.csv"
    IDEAS_JSONL = "generation/outputs/generated_ideas.jsonl"
    OUTPUT_JSONL = "evaluations.jsonl"
    
    run_pipeline(PROMPTS_CSV, IDEAS_JSONL, OUTPUT_JSONL)