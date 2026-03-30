import json
import ollama
from pydantic import BaseModel, Field, ValidationError

class JudgeEvaluation(BaseModel):
    reasoning: str = Field(
        description="Chain-of-thought reasoning evaluating the idea against the core problem. Must be generated FIRST."
    )
    utility_score: int = Field(
        description="Score from 1 to 5: 1=Irrelevant/Harmful, 5=Directly solves the root cause perfectly.",
        ge=1, le=5
    )
    feasibility_score: int = Field(
        description="Score from 1 to 5: 1=Impossible/Breaks physics, 5=Can be built immediately with existing tools.",
        ge=1, le=5
    )

class JudgeEngine:
    def __init__(self, model_name: str = "gemma3:27b"):
        self.model_name = model_name
        print(f"Judge Engine initialized with local model: '{self.model_name}'")

    def evaluate_quality(self, context_text: str, task_text: str, idea_text: str) -> dict:
        system_prompt = (
            "You are an impartial, expert engineering judge. Your job is to evaluate "
            "a proposed solution (Idea) against a specific Task and Context.\n\n"
            "Evaluate strictly on two dimensions:\n"
            "1. UTILITY (1-5): How effectively does it solve the core problem?\n"
            "2. FEASIBILITY (1-5): How realistic is implementation?\n\n"
            "You MUST output ONLY a valid JSON object matching this exact format:\n"
            "{\n"
            '  "reasoning": "your step-by-step logic here",\n'
            '  "utility_score": <integer 1-5>,\n'
            '  "feasibility_score": <integer 1-5>\n'
            "}"
        )
        user_prompt = f"Context: {context_text}\n\nTask: {task_text}\n\nIdea to Evaluate: {idea_text}"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format='json',
                options={'temperature': 0.0}
            )
            raw_content = response['message']['content']
            parsed_json = json.loads(raw_content)
            evaluation = JudgeEvaluation(**parsed_json)
            
            return {
                "utility": evaluation.utility_score,
                "feasibility": evaluation.feasibility_score,
                "reasoning": evaluation.reasoning
            }
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Validation Error during Judge Engine evaluation: {e}")
            return {
                "utility": 0,
                "feasibility": 0,
                "reasoning": f"FORMATTING FAILED: Model failed to return valid schema."
            }

        except Exception as e:
            print(f"Execution Error during Judge Engine evaluation: {e}")
            return {
                "utility": 0,
                "feasibility": 0,
                "reasoning": f"EVALUATION FAILED: {str(e)}"
            }


if __name__ == "__main__":
    engine = JudgeEngine(model_name="gemma3:27b")
    context = "A researcher needs to fine-tune a 7B LLM locally without Out of Memory (OOM) errors."
    task = "Devise a strategy to fine-tune the model."
    idea_strong = "Utilize QLoRA. Load the base model in 4-bit precision using bitsandbytes, and only train the injected adapter weights."
    print("\n--- Evaluating Strong Idea ---")
    print("Please wait, 27B models running partially on system RAM will take time...")
    res_strong = engine.evaluate_quality(context, task, idea_strong)
    print(f"\nReasoning: {res_strong['reasoning']}")
    print(f"Utility: {res_strong['utility']}/5")
    print(f"Feasibility: {res_strong['feasibility']}/5")