import os
import sys
import os.path as osp
import traceback
from flask import Flask, request, jsonify
import socket

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # Add parent dir to path

from inference import Arguments, run_inference_single_prompt
from fetch_material import fetch_materials

app = Flask(__name__)

@app.route('/fetch_material', methods=['POST'])
def fetch_material_route():
    data = request.get_json()
    text_input = data.get('query')

    if not text_input:
        return jsonify({'error': 'No query provided'}), 400

    # Call the existing main functionality
    try:
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
        response = run_inference_single_prompt(args, image_path, text_prompt)
        return jsonify({'response': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

app.run(host='0.0.0.0', port=5000)
