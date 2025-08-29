import argparse
import csv
import shutil
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm

"""
requires `ffmpeg` for reading mp4 metadata,
uses `aria2` for quick download
videos from https://github.com/NJU-PCALab/OpenVid-1M/tree/main
"""


def get_dimensions(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def get_part_directories(output_directory, part):
    part_dir = os.path.join(output_directory, f"part_{part}")
    zip_dir = os.path.join(part_dir, "zip")
    video_dir = os.path.join(part_dir, "video")
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    return part_dir, zip_dir, video_dir


def get_download_cmd(file_path, url):
    out_dir = os.path.dirname(file_path)
    out_file = os.path.basename(file_path)
    # command = ["wget", "-c", "-O", file_path, url]
    return ["aria2c", "-c", "-d", out_dir, "-o", out_file, "-x", "16", "-s", "16", url]


def download_part(i, output_directory, error_log_path):
    _, zip_dir, video_dir = get_part_directories(output_directory, i)
    url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}.zip"
    file_path = os.path.join(zip_dir, f"OpenVid_part{i}.zip")

    command = get_download_cmd(file_path, url)
    unzip_command = ["unzip", "-o", "-j", "-qq", file_path, "-d", video_dir]
    try:
        subprocess.run(command, check=True)
        subprocess.run(unzip_command, check=True)
    except subprocess.CalledProcessError as e:
        error_message = f"file {url} download failed: {e}\n"
        with open(error_log_path, "a") as error_log_file:
            error_log_file.write(error_message)

        part_urls = [
            f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partaa",
            f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{i}_partab",
        ]

        for part_url in part_urls:
            part_file_path = os.path.join(zip_dir, os.path.basename(part_url))
            if not os.path.exists(part_file_path):
                part_command = get_download_cmd(part_file_path, part_url)
                subprocess.run(part_command, check=True)

        cat_command = f"cat {os.path.join(zip_dir, f'OpenVid_part{i}_part*')} > {file_path}"
        os.system(cat_command)
        subprocess.run(unzip_command, check=True)

    return i


def process_file(filepath):
    return get_dimensions(filepath)


def analyze_part(i, output_directory, analyzer_workers=4):
    _, _, video_dir = get_part_directories(output_directory, i)
    mp4_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    part_counts = Counter()

    with ProcessPoolExecutor(max_workers=analyzer_workers) as pool:
        results = list(
            tqdm(
                pool.map(process_file, mp4_files),
                desc=f"Analyzing part {i}",
                total=len(mp4_files),
            )
        )

    for dim in results:
        if dim:
            part_counts[dim] += 1
    return i, part_counts


def clean_part(i, output_directory):
    part_dir, _, _ = get_part_directories(output_directory, i)
    if os.path.exists(part_dir):
        shutil.rmtree(part_dir)


def download_data_files(output_directory):
    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv",
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        command = ["wget", "-O", data_path, data_url]
        subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--output_directory",
        type=str,
        help="Path to the dataset directory",
        default="/data/jeffjma/",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=8,
        help="Max number of concurrent downloads/analyses",
    )
    parser.add_argument(
        "--analyzer_workers",
        type=int,
        default=50,
        help="Number of processes per analyzer task",
    )
    args = parser.parse_args()

    output_directory = args.output_directory
    work_dir = os.path.join(output_directory, "processing")
    os.makedirs(work_dir, exist_ok=True)
    error_log_path = os.path.join(work_dir, "download_log.txt")

    dimension_counts = Counter()
    num_parts = 186
    max_concurrent = args.max_concurrent
    analyzer_workers = args.analyzer_workers

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        download_map = {}
        analyze_map = {}
        all_futures = set()

        # Start initial downloads
        for i in range(min(max_concurrent, num_parts)):
            df = executor.submit(download_part, i, work_dir, error_log_path)
            download_map[df] = i
            all_futures.add(df)

        next_part_to_download = max_concurrent
        completed_parts = 0

        while completed_parts < num_parts:
            done_future = next(as_completed(all_futures))
            all_futures.remove(done_future)

            if done_future in download_map:
                # Download completed, immediately start analyzing
                part_index = download_map.pop(done_future)
                af = executor.submit(analyze_part, part_index, work_dir, analyzer_workers)
                analyze_map[af] = part_index
                all_futures.add(af)

            else:
                # Analysis completed
                part_index = analyze_map.pop(done_future)
                _, part_counts = done_future.result()
                dimension_counts.update(part_counts)
                clean_part(part_index, work_dir)
                completed_parts += 1

                # If there are more parts to download, start them
                if next_part_to_download < num_parts:
                    df = executor.submit(download_part, next_part_to_download, work_dir, error_log_path)
                    download_map[df] = next_part_to_download
                    all_futures.add(df)
                    next_part_to_download += 1

    # Clean up if empty
    if os.path.exists(work_dir) and not os.listdir(work_dir):
        os.rmdir(work_dir)

    download_data_files(output_directory)

    with open("stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for dimension, count in dimension_counts.items():
            writer.writerow([dimension, count])


if __name__ == "__main__":
    main()
