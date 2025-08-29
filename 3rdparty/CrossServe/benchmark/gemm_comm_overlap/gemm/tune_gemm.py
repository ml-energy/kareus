import sys
import os
import pandas as pd


def get_time_from_output_file(file_name):
    try:
        t_ms = 1e100
        with open(file_name) as f:
            for line in f.readlines():
                if "Error" in line or "error" in line:
                    return 1e100
                if "time" in line:
                    t_ms = float(line.rstrip()[6:-3])
        return t_ms
    except:
        return 1e100


def get_energy_from_output_file(file_name):
    try:
        e_mj = 1e100
        with open(file_name) as f:
            for line in f.readlines():
                if "energy" in line:
                    e_mj = float(line.rstrip()[8:-3])
        return e_mj
    except:
        return 1e100


def template(config):
    (
        ab_type,
        c_type,
        op_class,
        accum_type,
        cta_m,
        cta_n,
        cta_k,
        stages,
        warps_m,
        warps_n,
        warps_k,
        inst_m,
        inst_n,
        inst_k,
    ) = config

    with open("__test2__.cu") as inf:
        source_template = inf.read()

    source_template = source_template.replace("__blockM__", str(cta_m))
    source_template = source_template.replace("__blockN__", str(cta_n))
    source_template = source_template.replace("__blockK__", str(cta_k))

    source_template = source_template.replace("__warpM__", str(int(cta_m / warps_m)))
    source_template = source_template.replace("__warpN__", str(int(cta_n / warps_n)))
    source_template = source_template.replace("__warpK__", str(int(cta_k / warps_k)))

    source_template = source_template.replace("__instM__", str(inst_m))
    source_template = source_template.replace("__instN__", str(inst_n))
    source_template = source_template.replace("__instK__", str(inst_k))

    source_template = source_template.replace("__kstages__", str(stages))
    return source_template


def template_test(source_template, batch, m, n, k):
    source_template = source_template.replace("__batch__", str(batch))
    source_template = source_template.replace("__m__", str(m))
    source_template = source_template.replace("__n__", str(n))
    source_template = source_template.replace("__k__", str(k))
    return source_template


def run(config, batch, m, n, k, freq):
    source_template = template(config)
    source = template_test(source_template, batch, m, n, k)
    with open("test2.cu", "w") as outf:
        outf.write(source)

    cmd = "nvcc -arch=sm_86  -std=c++17 -I/workspaces/CrossServe/3rdparty/cutlass/tools/util/include -I/workspaces/CrossServe/3rdparty/cutlass/include -I/workspaces/CrossServe/3rdparty/cutlass/examples/common -lnvidia-ml --expt-relaxed-constexpr test2.cu -o test2"
    os.system(cmd)
    if not os.path.exists("test2"):
        return False
    os.system("export CUDA_VISIBLE_DEVICES=2 && ./test2 > tmp 2>&1".format(cmd))

    time = get_time_from_output_file("tmp")
    gflops = (2 * batch * m * n * k) / 1000000000000 / (time / 1000)
    print("time:", time, "ms")
    print("TFLOPS:", gflops)

    energy = get_energy_from_output_file("tmp")
    print("energy:", energy, "mJ")
    print("power:", energy / time, "W")
    os.system("rm tmp && rm test2 && rm test2.cu")

    (
        ab_type,
        c_type,
        op_class,
        accum_type,
        cta_m,
        cta_n,
        cta_k,
        stages,
        warps_m,
        warps_n,
        warps_k,
        inst_m,
        inst_n,
        inst_k,
    ) = config
    with open("gemm_{}_{}_{}_{}_{}.csv".format(batch, m, n, k, freq), "a") as outf:
        outf.write(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                ab_type,
                c_type,
                op_class,
                accum_type,
                cta_m,
                cta_n,
                cta_k,
                stages,
                warps_m,
                warps_n,
                warps_k,
                inst_m,
                inst_n,
                inst_k,
                time,
                gflops,
                energy,
                energy / time,
            )
        )
    return True


if __name__ == "__main__":
    batch = int(sys.argv[1])
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    k = int(sys.argv[4])
    freq = "default"

    with open("gemm_{}_{}_{}_{}_{}.csv".format(batch, m, n, k, freq), "w") as outf:
        outf.write(
            "ab_type,c_type,op_class,accum_type,cta_m,cta_n,cta_k,"
            "stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,time (ms),TFLOPS,energy (mJ),power(W)\n"
        )

    tuning_space = pd.read_csv("cutlass_tuning_space.csv")

    config_list = []
    best_time = 1e100
    for idx, config in tuning_space.iterrows():
        if config[0] == "f16" and config[1] == "f16" and config[3] == "f32" and config[2] == "tensorop":
            print("config:")
            print(config)
            ok = run(config, batch, m, n, k, freq)
