import os
import os.path as osp

from llava_hf.inference import Arguments, render_single_code, response_to_code


def main():
    # Read the response containing a code block
    with open("working_sample.txt", "r") as f:
        response = f.read()

    code = response_to_code(response)

    # Build minimal Arguments needed by render_single_code
    out_dir = "test_outputs"
    os.makedirs(out_dir, exist_ok=True)

    args = Arguments(
        test_data_path="",
        output_dir=out_dir,
        # model_base and other defaults are fine; override blender path if needed
        blender_path=osp.join(
            osp.dirname(osp.abspath(__file__)), "infinigen", "blender", "blender"
        ),
        image_folder=osp.join(
            osp.dirname(osp.abspath(__file__)), "material_dataset_filtered_v2"
        ),
        info_dir=osp.join(osp.dirname(osp.abspath(__file__)), "material_dataset_info"),
        mode="render",
        num_processes=1,
        num_samples=1,
        max_samples=1,
        display_id=99,
        device_id=[0],
    )

    # Blender requires an absolute filepath for saving .blend files
    blend_save = osp.abspath(osp.join(out_dir, "render_single.blend"))
    rendered = render_single_code(
        code=code, args=args, example_dir=out_dir, save_path=blend_save
    )
    if rendered:
        print(f"Rendered image written to: {rendered}")
    else:
        print("Rendering failed. Check logs in the output directory.")


if __name__ == "__main__":
    main()
