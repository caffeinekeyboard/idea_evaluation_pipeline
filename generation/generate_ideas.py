import argparse
import csv
import json
import uuid
from pathlib import Path
from tqdm import tqdm
import ollama

def load_prompts(csv_path: str) -> list[dict]:
    """Reads prompts from a CSV file."""
    prompts = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts

def generate_ideas(model_name: str, input_csv: str, output_jsonl: str, output_csv: str, num_ideas_per_prompt: int, min_temp: float, max_temp: float):
    prompts = load_prompts(input_csv)
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting generation using model: {model_name}")
    print(f"Temperature band: {min_temp} to {max_temp} across {num_ideas_per_prompt} generations")
    print(f"Total prompts to process: {len(prompts)}")
    csv_headers = ["prompt_id", "umbrella", "idea_id", "model_used", "temperature", "idea_text"]
    csv_exists = Path(output_csv).exists()
    
    with open(output_jsonl, mode='a', encoding='utf-8') as f_jsonl, open(output_csv, mode='a', encoding='utf-8', newline='') as f_csv:
        csv_writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
        
        if not csv_exists:
            csv_writer.writeheader()
            
        for prompt_data in tqdm(prompts, desc="Processing Prompts"):
            prompt_id = prompt_data.get('prompt_id', 'unknown_id')
            task = prompt_data.get('prompt', '')
            context = prompt_data.get('context', '')
            umbrella = prompt_data.get('umbrella', 'General Engineering')
            
            if not task:
                continue

            full_prompt = f"Context: {context}\n\nTask: {task}"

            if num_ideas_per_prompt == 1:
                temps = [min_temp]
            else:
                step = (max_temp - min_temp) / (num_ideas_per_prompt - 1)
                temps = [min_temp + (i * step) for i in range(num_ideas_per_prompt)]

            for i, temp in enumerate(temps):
                current_temp = round(temp, 2)
                unique_hash = uuid.uuid4().hex[:8]
                idea_id = f"{prompt_id}_{unique_hash}"
                
                try:
                    response = ollama.chat(
                        model=model_name, 
                        messages=[
                            {
                                'role': 'system',
                                'content': f"You are an expert ideation assistant specializing in {umbrella}. Provide a highly specific, concrete solution to the user's task based on the provided context. Output only the technical solution, without conversational filler or introductory fluff."
                            },
                            {
                                'role': 'user',
                                'content': full_prompt
                            }
                        ],
                        options={
                            'temperature': current_temp
                        }
                    )
                    idea_text = response['message']['content'].strip()
                    payload = {
                        "prompt_id": prompt_id,
                        "umbrella": umbrella,
                        "idea_id": idea_id,
                        "model_used": model_name,
                        "temperature": current_temp,
                        "idea_text": idea_text
                    }
                    
                    f_jsonl.write(json.dumps(payload) + "\n")
                    csv_writer.writerow(payload)
                    
                    f_jsonl.flush()
                    f_csv.flush()
                    
                except Exception as e:
                    print(f"\nError generating idea for {prompt_id} at temp {current_temp}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ideas from prompts using Ollama.")
    parser.add_argument("--model", type=str, required=True, help="Name of the Ollama model (e.g., llama3, mistral)")
    parser.add_argument("--input", type=str, default="prompts.csv", help="Path to input CSV")
    parser.add_argument("--output_jsonl", type=str, default="outputs/generated_ideas.jsonl", help="Path to output JSONL file")
    parser.add_argument("--output_csv", type=str, default="outputs/generated_ideas.csv", help="Path to output CSV file")
    parser.add_argument("--n", type=int, default=5, help="Number of ideas to generate per prompt")
    parser.add_argument("--min_temp", type=float, default=0.1, help="Minimum temperature boundary")
    parser.add_argument("--max_temp", type=float, default=0.9, help="Maximum temperature boundary")
    
    args = parser.parse_args()
    generate_ideas(args.model, args.input, args.output_jsonl, args.output_csv, args.n, args.min_temp, args.max_temp)