import os
import sys
import os.path as osp
import traceback

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # Add parent dir to path

from inference import Arguments, run_inference_single_prompt
from fetch_material import fetch_materials

def main():
    # Get text input from command line arguments
    if len(sys.argv) < 2:
        print("[Error]: Please provide a text query as a command line argument")
        print("Usage: python single_prompt_example.py <text_query>")
        return
    
    text_input = " ".join(sys.argv[1:])
    
    # Example usage
    args = Arguments(
        test_data_path="",
        output_dir="./single_output",
        model_path=None,
        model_base='llava-hf/llama3-llava-next-8b-hf',
        device_id=[],
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        max_length=2048
    )
    
    # Fetch material using the text query
    print(f"Fetching material for query: {text_input}")
    try:
        results = fetch_materials(
            text_input,
            api_url="http://brahmastra.ucsd.edu:3001/search",
            top_k=1,
            cache_dir=".material_cache",
        )
        
        if not results:
            print("[Error]: No results returned from fetch_materials")
            return
        
        # Get the image path from the first result
        image_path = results[0].get("image_path_local")
        
        if not image_path:
            print("[Error]: No image path in the results")
            return
            
        print(f"Retrieved image: {image_path}")
        print(f"Score: {results[0].get('score', 0):.3f}")
        
    except Exception as e:
        print(f"[Error] fetching material: {e}")
        traceback.print_exc()
        return
    
    text_prompt = "Write a Python function with Blender API to create a material node graph for this image."
    
    if not os.path.exists(image_path):
        print(f"[Error]: Cannot find {image_path}")
        return
    
    try:
        print(f"Image Path: {image_path}")
        print(f"Prompt: {text_prompt}")

        response = run_inference_single_prompt(args, image_path, text_prompt)
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(response)
        print("="*50)
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "response.txt")
        with open(output_file, 'w') as f:
            f.write(response)
        print(f"\n Saved to {output_file}")
        
    except Exception as e:
        print(f"error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
