import os
import subprocess
import argparse
from glob import glob

def run_mitsuba(xml_path, exr_folder):
    exr_path = os.path.join(exr_folder, os.path.splitext(os.path.basename(xml_path))[0] + ".exr")
    try:
        print(f"Running Mitsuba on {xml_path}...")
        subprocess.run(["mitsuba", xml_path, "-o", exr_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Mitsuba on {xml_path}: {e}")

def convert_exr_to_png(exr_path, png_folder):
    png_path = os.path.join(png_folder, os.path.splitext(os.path.basename(exr_path))[0] + ".png")
    try:
        print(f"Converting {exr_path} to {png_path}...")
        subprocess.run(["ffmpeg", "-y", "-i", exr_path, png_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {exr_path} to PNG: {e}")

def main(folder, exr_folder, png_folder):
    # Step 1: Run Mitsuba on all .xml files
    xml_files = glob(os.path.join(folder, "*.xml"))
    for xml_file in xml_files:
        run_mitsuba(xml_file, exr_folder)

    # Step 2: Convert all .exr files to .png
    exr_files = glob(os.path.join(exr_folder, "*.exr"))
    for exr_file in exr_files:
        convert_exr_to_png(exr_file, png_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render .xml files with Mitsuba and convert .exr to .png using ffmpeg.")
    parser.add_argument("--xml", type=str, help="Path to the folder containing .xml files")
    parser.add_argument("--exr", type=str, help="Path to the folder containing .exr files")
    parser.add_argument("--png", type=str, help="Path to the folder to save .png files")
    args = parser.parse_args()

    main(args.xml, args.exr, args.png)