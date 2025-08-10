import subprocess
import os
import sys
import shutil
import click
from pathlib import Path
from glyphdata.utils.io_utils import validate_positive_int


@click.command()
@click.option(
    "--image-dir", "-c",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory that contains the content images which shall be adapted in their style."
)
@click.option(
    "--style-dir", "-t",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory that contains the a handful of image files that shall serve as style references."
)
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory where stylized images will be saved."
)
@click.option(
    '--image-size', "-s",
    required=False,
    default=128,
    type=int,
    callback=validate_positive_int,
    help='Square image size as a single integer (e.g., 128)'
)
@click.option(
    '--max-results', "-m",
    required=False,
    default=1000000,
    type=int,
    callback=validate_positive_int,
    help='Maximum number of output files.'
)
def run_stylization_process(image_dir, style_dir, output_dir, image_size, max_results):
    content_path = Path(image_dir).absolute().as_posix()    
    style_path = Path(style_dir).absolute().as_posix()
    results_dir = Path(output_dir)
    temp_results_dir = Path(output_dir) / "_temp"
    temp_results_dir.mkdir(parents=True, exist_ok=True)
    temp_results_path = temp_results_dir.absolute().as_posix()
    results_name = 'AdaAttN_glyphs'

    submodule_script = os.path.join(os.getcwd(), 'adaattn', 'test.py')
    arguments = ['--content_path', content_path, 
                 '--style_path', style_path,
                 '--results_dir', temp_results_path,
                 '--name', results_name,
                 '--model', 'adaattn',
                 '--dataset_mode', 'unaligned',
                 '--load_size', str(image_size),
                 '--crop_size', str(image_size),
                 '--image_encoder_path', 'checkpoints/vgg_normalised.pth',
                 '--gpu_ids', '0',
                 '--skip_connection_3', 
                 '--shallow_layer',
                 '--num_test', str(max_results)]
    
    command = ['python', submodule_script] + arguments
    submodule_dir = os.path.dirname(submodule_script)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    with subprocess.Popen(command,
                          cwd=submodule_dir,  
                          text=True, 
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, 
                          env=env
        ) as style_proc:
        pass

        for stdout_line in iter(style_proc.stdout.readline, ""):
            print(stdout_line, end="") 
        for stderr_line in iter(style_proc.stderr.readline, ""):
            print(stderr_line, end="", file=sys.stderr)

        style_proc.stdout.close()
        style_proc.stderr.close()
        style_proc.wait()

    # move relevant contents in temp structure to output directory
    cs_dir = temp_results_dir / results_name / "test_latest" / "images"
    for fn in cs_dir.glob('*_*s.png'):
        destination_path = results_dir / fn.name
        shutil.move(fn.absolute().as_posix(), destination_path.absolute().as_posix())
    
    # remove temporary directory
    try:
        shutil.rmtree(temp_results_path)
        print(f"Directory {temp_results_path} and all its contents have been removed.")
        print(f"--> The stylized images can be found in {results_dir.absolute().as_posix()}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_stylization_process()
