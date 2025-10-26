import os
import sys
import subprocess
import sqlite3
import pandas as pd


def get_overlap_windows():
    return [
        (-1, -1),
        (0, 8), (2, 8), (4, 8), (6, 8), (7, 8),
        # (0, 1), (2, 3), (4, 5), (6, 6), (7, 8),
        # (0, 3), (2, 5), (4, 6), (6, 8),
        # (0, 5), (2, 6), (4, 8),
        # (0, 6), (2, 8),
        # (0, 8),
    ]


def process_sqlite(metrics_file):
    conn = sqlite3.connect(metrics_file)
    cursor = conn.cursor()
    # Determine GPU with shorter total allreduceKernelEntryPointBF16 kernel time
    chosen_gpu = None
    demangled_id_row = pd.read_sql_query(
        """
        SELECT id AS demangled_id
        FROM StringIds
        WHERE value = 'allreduce1'
        LIMIT 1
        """,
        conn,
    )
    demangled_id = int(demangled_id_row["demangled_id"].iloc[0])
    duration_rows = pd.read_sql_query(
        """
        SELECT deviceId, SUM(end - start) AS total_duration
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE demangledName = ? AND deviceId IN (0, 1)
        GROUP BY deviceId
        ORDER BY total_duration ASC
        """,
        conn,
        params=(demangled_id,),
    )
    # Choose the GPU with the shorter total duration; tie-breaker on smaller deviceId
    duration_rows = duration_rows.sort_values(["total_duration", "deviceId"], ascending=[True, True])
    chosen_gpu = int(duration_rows["deviceId"].iloc[0])
    print("chosen_gpu:", chosen_gpu)

    # Fetch metrics for the chosen GPU by decoding GPU_METRICS.typeId
    # GPU ordinal = (typeId - vmId) & 0xFFFFFFFF
    # Map visible CUPTI deviceId (0..N-1) to system ordinal using CUDA_VISIBLE_DEVICES
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    visible_ordinals = [int(x) for x in visible_env.split(",") if x.strip() != ""]
    chosen_gpu_system_ordinal = visible_ordinals[chosen_gpu]

    query = """
    WITH vm(vmid) AS (
        SELECT DISTINCT vmId FROM TARGET_INFO_GPU
    )
    SELECT
        g.timestamp,
        t.metricName,
        g.value
    FROM GPU_METRICS AS g
    JOIN TARGET_INFO_GPU_METRICS AS t
      ON g.typeId = t.typeId AND g.metricId = t.metricId
    CROSS JOIN vm
    WHERE ((g.typeId - vm.vmid) & 0xFFFFFFFF) = ?
    """
    df = pd.read_sql_query(query, conn, params=(chosen_gpu_system_ordinal,))
    df = transform_metrics_to_columns(df)

    kernel_query = """
    SELECT
        MIN(start) as start_time,
        MAX(end) as end_time
    FROM (
        SELECT k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
        WHERE k.deviceId = ? AND (s.value IS NULL OR s.value NOT LIKE '%FillFunctor%')
        ORDER BY k.start
    )
    """
    kernel_time_df = pd.read_sql_query(kernel_query, conn, params=(chosen_gpu,))
    start_time = kernel_time_df['start_time'].iloc[0]
    end_time = kernel_time_df['end_time'].iloc[0]
    df = df[df['timestamp'] >= start_time]
    df = df[df['timestamp'] <= end_time]
    # df = df[df["GPU Active [Throughput %]"] > 0]

    conn.close()
    return df


def transform_metrics_to_columns(metrics_df):
    pivot_df = metrics_df.pivot(index='timestamp', columns='metricName', values='value')
    pivot_df = pivot_df.reset_index()
    return pivot_df


def calculate_metrics_stats(df):
    metrics_list = [
        "SMs Active [Throughput %]",
        "SM Instructions [Throughput %]",
        "NVLink RX Responses User Data [Throughput %]",
        "NVLink TX Responses User Data [Throughput %]",
        "DRAM Read Bandwidth [Throughput %]",
        "DRAM Write Bandwidth [Throughput %]",
        "GPC Clock Frequency [MHz]",
    ]

    if "SM Issue [Throughput %]" in df.columns and "Tensor Active [Throughput %]" in df.columns:
        df["SM Instructions [Throughput %]"] = df["SM Issue [Throughput %]"] + df["Tensor Active [Throughput %]"]
    if "GPC Clock Frequency [MHz]" in df.columns:
        df["GPC Clock Frequency [MHz]"] = df["GPC Clock Frequency [MHz]"] / 1000000

    results = []
    for metric in metrics_list:
        if metric in df.columns:
            mean_value = df[metric].mean()
            var_value = df[metric].var()
        else:
            mean_value = None
            var_value = None
        results.append({
            'Metric': metric,
            'Mean': mean_value,
            'Variance': var_value,
        })
    results_df = pd.DataFrame(results)
    return results_df


def run_cmd(cmd_str):
    os.system(cmd_str)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=4)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--seq_len", "-s", type=int, default=4096)
    parser.add_argument("--frequency", "-f", type=str, default="default")
    args = parser.parse_args()
    frequency = args.frequency

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible is not None and len(visible.strip()) > 0:
        vis_list = [x for x in visible.split(",") if x.strip() != ""]
        target_indices = vis_list
    else:
        raise ValueError("CUDA_VISIBLE_DEVICES is not set")

    os.makedirs(f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}", exist_ok=True)
    os.makedirs(f"results/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}", exist_ok=True)

    results = []

    skip = False
    overlap_windows = get_overlap_windows()
    for overlap_start, overlap_end in overlap_windows:
        for sm_num in range(1, 21):
            for block_size in [512, 1024]:
                if sm_num == 11 and block_size == 1024 and overlap_start == 0 and overlap_end == 1:
                    skip = False
                if skip:
                    continue
                output_name = f"profile_{overlap_start}_{overlap_end}_{sm_num}_{block_size}"
                nsys_report = f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}/{output_name}.nsys-rep"

                # try:
                profile_cmd = [
                    "nsys profile",
                    "--gpu-metrics-devices", ",".join(target_indices),
                    "--capture-range", "cudaProfilerApi",
                    "--force-overwrite", "true",
                    "-o", f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}/{output_name}",
                    "python", f"overlap_test_attn_individual.py",
                    "--world_size", str(args.world_size),
                    "--batch_size", str(args.batch_size),
                    "--seq_len", str(args.seq_len),
                    "--frequency", frequency,
                    "--overlap_start", str(overlap_start),
                    "--overlap_end", str(overlap_end),
                    "--sm_num", str(sm_num),
                    "--block_size", str(block_size),
                ]
                if not skip and not os.path.exists(nsys_report):
                    run_cmd(" ".join(profile_cmd))
                print(f"[✓] nsys profiling done. Output: {nsys_report}")

                csv_file = f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}/{output_name}.csv"
                metrics_file = f"profile_result/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}/{output_name}.sqlite"
                export_cmd = [
                    "nsys stats", nsys_report,
                    "--report", "cuda_gpu_kern_sum",
                    "--format", "csv",
                    "--force-export", "true",
                    "--sqlite", metrics_file,
                    "--output", csv_file,
                ]
                if not skip and not os.path.exists(metrics_file):
                    run_cmd(" ".join(export_cmd))
                print(f"[✓] nsys export done. Output: {metrics_file}")

                metrics_df = process_sqlite(metrics_file)
                metrics_df.to_csv(f"results/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}/{output_name}.csv", index=False)

                stats_df = calculate_metrics_stats(metrics_df)
                stats_df.to_csv(f"results/tp{args.world_size}-bs{args.batch_size}-seq{args.seq_len}/{frequency}/{output_name}.metrics_stats.csv", index=False)

                result = {
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "sm_num": sm_num,
                    "block_size": block_size,
                }
                for _, row in stats_df.iterrows():
                    result[row["Metric"] + " Mean"] = row["Mean"]
                results.append(result)
                print(f"[✓] Result collected for {output_name}")
                # exit()
                # except:
                #     print(f"[❌] Error processing {output_name}")
                #     continue

    # if results:
    #     results_df = pd.DataFrame(results)
    #     results_df.to_csv(f"results/{frequency}/all_results_attn_{frequency}.csv", index=False)
    #     print(f"[✓] Aggregated results written to results/{frequency}/all_results_attn_{frequency}.csv")


