import os
import sys
import os.path as osp
import traceback
from flask import Flask, request, jsonify
import socket

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # Add parent dir to path

from inference import Arguments, SinglePromptEngine
from fetch_material import fetch_materials

app = Flask(__name__)

# Preload the processor and model once at startup
# You can tweak these defaults or make them configurable via env vars
GLOBAL_ARGS = Arguments(
    test_data_path="",
    output_dir="./single_output",
    model_path="/app/llava_hf/checkpoints_pretrained/llava-llama3-8b-sllm-p10/checkpoint-epoch5",  # set to a checkpoint path if you have one
    model_base=os.environ.get('LLAVA_MODEL_BASE', 'llava-hf/llama3-llava-next-8b-hf'),
    device_id=[int(x) for x in os.environ.get('CUDA_VISIBLE_IDS', '').split(',') if x.strip().isdigit()],
    temperature=0.6,
    top_k=50,
    top_p=0.9,
    max_length=2048,
    num_processes=4,
    mode='gen',
)
ENGINE = SinglePromptEngine(GLOBAL_ARGS)

@app.route('/fetch_material', methods=['POST'])
def fetch_material_route():
    data = request.get_json()
    text_input = data.get('query')

    if not text_input:
        return jsonify({'error': 'No query provided'}), 400
    print(f"Received query: {text_input}")

    # Call the existing main functionality
    try:
        results = fetch_materials(
            text_input,
            api_url="http://brahmastra.ucsd.edu:3001/search",
            top_k=1,
            cache_dir=".material_cache",
        )
        if not results:
            return jsonify({'error': 'No results found'}), 404

        image_path = results[0].get("image_path_local")
        if not image_path:
            return jsonify({'error': 'No image path in results'}), 404

        text_prompt = "Write a Python function with Blender API to create a material node graph for this image."
        # Use preloaded engine to avoid reloading the model each request
        response = ENGINE.generate(image_path, text_prompt)
        return jsonify({'response': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# app.run(host='0.0.0.0', port=5000)
if __name__ == '__main__':
    PROMPT = (
    'Write a Python function with Blender API to create a material node graph '
    'for this image.'
    )
    text_prompt = PROMPT


    # image_path = "/app/llava_hf/results/llava-llama3-8b-sllm-p10/test-single-eval-epoch5/015-blenderkit-procedural_chees_a256b8f2-6d4a-49ad-8a6a-32520eeab5f6-blender/input.jpg"
    # image_path = "/app/checkerboard.png"
    image_path = "/app/rusty_texture.jpg"
    response = ENGINE.generate(image_path, text_prompt)
    print("---")
    print(response)