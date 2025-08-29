import sys
import argparse
import os
import re


def template_test(batch_size, seq_len, sm_num, block_size):
    with open("__overlap__.cu") as inf:
        source_template = inf.read()
    # source_template = source_template.replace("__batch_size__", str(batch_size))
    # source_template = source_template.replace("__seq_len__", str(seq_len))
    source_template = source_template.replace("__SM_num__", str(sm_num))
    source_template = source_template.replace("__block_size__", str(block_size))
    return source_template


def parse_output(file_name):
    with open(file_name) as f:
        data = f.read()
    pattern = r"Rank (\d+), time: ([\d.]+) ms, energy: ([\d.]+) mJ"
    matches = re.findall(pattern, data)
    parsed_data = {}
    for rank, time, energy in matches:
        parsed_data[int(rank)] = {"time": float(time), "energy": float(energy)}
    return parsed_data


if __name__ == "__main__":
    # freq = str(sys.argv[1])
    freq = "default"
    batch_size = 4
    seq_len = 4096

    with open("overlap_{}.csv".format(freq), "w") as f:
        f.write(
            "sm_num,block_size,rank0 time (ms),rank0 energy (mJ),rank1 time (ms),rank1 energy (mJ),rank2 time (ms),rank2 energy (mJ),rank3 time (ms),rank3 energy (mJ),max time (ms),total energy (mJ)\n"
        )

    for sm_num in range(3, 31, 3):
        for block_size in [128, 256, 512, 1024]:
            os.system("mpirun --allow-run-as-root -np 4 build/test_overlap {} {} > tmp 2>&1".format(sm_num, block_size))

            data = parse_output("tmp")
            max_time = max([data[i]["time"] for i in range(4)])
            total_energy = sum([data[i]["energy"] for i in range(4)])

            with open("overlap_{}.csv".format(freq), "a") as f:
                f.write(
                    "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        sm_num,
                        block_size,
                        data[0]["time"],
                        data[0]["energy"],
                        data[1]["time"],
                        data[1]["energy"],
                        data[2]["time"],
                        data[2]["energy"],
                        data[3]["time"],
                        data[3]["energy"],
                        max_time,
                        total_energy,
                    )
                )
            os.system("rm tmp")

    # baseline
    os.system("mpirun --allow-run-as-root -np 4 build/test_sequential > tmp 2>&1")

    data = parse_output("tmp")
    max_time = max([data[i]["time"] for i in range(4)])
    total_energy = sum([data[i]["energy"] for i in range(4)])

    with open("overlap_{}.csv".format(freq), "a") as f:
        f.write(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                "sequential",
                "baseline",
                data[0]["time"],
                data[0]["energy"],
                data[1]["time"],
                data[1]["energy"],
                data[2]["time"],
                data[2]["energy"],
                data[3]["time"],
                data[3]["energy"],
                max_time,
                total_energy,
            )
        )

    os.system("rm tmp")
