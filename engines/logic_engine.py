import json
import ollama
from typing import List, Dict, Tuple

class LogicEngine:
    def __init__(self, model_name: str = "gemma:instruct"):
        self.model_name = model_name
        print(f"Logic Engine initialized with model: '{self.model_name}'")

    def extract_constraints(self, context_text: str, task_text: str) -> List[str]:
        system_prompt = (
            "You are a strict technical requirements engineer. "
            "Read the context and the task, and extract the mandatory technical constraints "
            "and requirements that a valid solution MUST satisfy. "
            "Output ONLY a valid JSON object with a single key 'constraints' containing a list of strings. "
            "Keep constraints atomic and verifiable."
        )
        user_prompt = f"Context: {context_text}\n\nTask: {task_text}"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                format='json',
                options={'temperature': 0.0}
            )
            output = json.loads(response['message']['content'])
            return output.get('constraints', [])
            
        except Exception as e:
            print(f"Error extracting constraints: {e}")
            return []

    def evaluate_depth(self, idea_text: str, constraints: List[str]) -> Tuple[float, Dict[str, bool]]:
        if not constraints:
            return 0.0, {}

        system_prompt = (
            "You are a rigid validation logic gate. "
            "Evaluate if the provided 'Idea' successfully addresses each of the listed 'Constraints'. "
            "Output ONLY a valid JSON object with a single key 'evaluations'. "
            "The value must be a dictionary where the keys are the exact constraint text, "
            "and the values are strictly boolean true or false."
        )
        user_prompt = f"Constraints:\n{json.dumps(constraints, indent=2)}\n\nIdea:\n{idea_text}"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                format='json',
                options={'temperature': 0.0}
            )
            output = json.loads(response['message']['content'])
            evaluations = output.get('evaluations', {})
            true_count = 0
            clean_evaluations = {}
            
            for constraint in constraints:
                passed = bool(evaluations.get(constraint, False)) 
                clean_evaluations[constraint] = passed
                if passed:
                    true_count += 1
                    
            depth_score = float(true_count) / len(constraints)
            return depth_score, clean_evaluations
            
        except Exception as e:
            print(f"Error evaluating depth: {e}")
            return 0.0, {c: False for c in constraints}


if __name__ == "__main__":
    engine = LogicEngine(model_name="gemma:instruct")
    context = "Pune is a historical city... 80% of traffic compressed onto 270KM of major arterial roads. Widening them is a litigation nightmare. RFID readers exist on traffic lights."
    task = "Please find a way to reduce traffic during peak hours in the city of Pune."
    print("\n--- Step 1: Extracting Constraints ---")
    constraints = engine.extract_constraints(context, task)
    for i, c in enumerate(constraints):
        print(f"{i+1}. {c}")
        
    print("\n--- Step 2: Evaluating Ideas ---")
    idea_strong = "Utilize the existing RFID readers to implement a dynamic, algorithmic toll pricing system during peak hours, discouraging non-essential travel without requiring any road widening."
    idea_weak = "Just build a massive 10-lane underground highway system beneath the entire city using tunnel boring machines."
    score_strong, eval_strong = engine.evaluate_depth(idea_strong, constraints)
    score_weak, eval_weak = engine.evaluate_depth(idea_weak, constraints)
    
    print(f"\nStrong Idea Depth Score: {score_strong:.2f}")
    for c, passed in eval_strong.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status} {c}")
        
    print(f"\nWeak Idea Depth Score: {score_weak:.2f}")
    for c, passed in eval_weak.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status} {c}")