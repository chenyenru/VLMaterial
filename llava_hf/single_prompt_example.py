import os
import sys
import os.path as osp
import traceback

sys.path.append(osp.dirname(osp.abspath(__file__)))

from inference import Arguments, run_inference_single_prompt

def main():
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
    
    image_path = "material_dataset_filtered/B3DMatPack1.2/Basic_Dented/transpiled_render.jpg"
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
