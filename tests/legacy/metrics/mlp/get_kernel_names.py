#!/usr/bin/env python3
"""
Script to extract kernel names from CUPTI_ACTIVITY_KIND_KERNEL table
in NSight Systems SQLite database files.
"""

import sqlite3
import argparse
from typing import List, Tuple, Optional

def get_kernel_names(db_path: str, limit: Optional[int] = None) -> List[Tuple]:
    """
    Get kernel names from the CUPTI_ACTIVITY_KIND_KERNEL table.
    
    Args:
        db_path: Path to the SQLite database file
        limit: Optional limit on number of results
    
    Returns:
        List of tuples containing kernel execution data with names
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query to get kernel names by joining with StringIds table
    query = '''
    SELECT 
        k.start,
        k.end,
        k.deviceId,
        k.streamId,
        k.correlationId,
        s_demangled.value AS demangledKernelName,
        s_short.value AS shortKernelName,
        s_mangled.value AS mangledKernelName,
        k.gridX * k.gridY * k.gridZ AS totalBlocks,
        k.blockX * k.blockY * k.blockZ AS threadsPerBlock,
        (k.end - k.start) AS duration_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    LEFT JOIN StringIds s_demangled ON k.demangledName = s_demangled.id
    LEFT JOIN StringIds s_short ON k.shortName = s_short.id
    LEFT JOIN StringIds s_mangled ON k.mangledName = s_mangled.id
    ORDER BY k.start
    '''
    
    if limit:
        query += f' LIMIT {limit}'
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    return results

def get_unique_kernel_names(db_path: str) -> List[str]:
    """
    Get unique kernel names from the database.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        List of unique kernel names
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = '''
    SELECT DISTINCT s.value AS kernelName
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE s.value IS NOT NULL
    ORDER BY s.value
    '''
    
    cursor.execute(query)
    results = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return results

def get_kernel_statistics(db_path: str) -> List[Tuple]:
    """
    Get kernel execution statistics grouped by kernel name.
    
    Args:
        db_path: Path to the SQLite database file
    
    Returns:
        List of tuples with kernel statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = '''
    SELECT 
        s.value AS kernelName,
        COUNT(*) AS executionCount,
        AVG(k.end - k.start) AS avgDuration_ns,
        MIN(k.end - k.start) AS minDuration_ns,
        MAX(k.end - k.start) AS maxDuration_ns,
        SUM(k.end - k.start) AS totalDuration_ns
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    JOIN StringIds s ON k.demangledName = s.id
    WHERE s.value IS NOT NULL
    GROUP BY s.value
    ORDER BY totalDuration_ns DESC
    '''
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract kernel names from NSight Systems database')
    parser.add_argument('db_path', help='Path to the SQLite database file')
    parser.add_argument('--mode', choices=['list', 'unique', 'stats'], default='list',
                       help='Mode: list (all kernels), unique (unique names), stats (statistics)')
    parser.add_argument('--limit', type=int, help='Limit number of results (for list mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'list':
        print("Kernel Execution Details:")
        print("=" * 80)
        results = get_kernel_names(args.db_path, args.limit)
        for i, row in enumerate(results, 1):
            start, end, device_id, stream_id, corr_id, demangled, short, mangled, blocks, threads, duration = row
            print(f"Kernel {i}:")
            print(f"  Name: {demangled}")
            print(f"  Device: {device_id}, Stream: {stream_id}")
            print(f"  Duration: {duration:,} ns ({duration/1e6:.2f} ms)")
            print(f"  Grid: {blocks} blocks, {threads} threads/block")
            print()
    
    elif args.mode == 'unique':
        print("Unique Kernel Names:")
        print("=" * 40)
        names = get_unique_kernel_names(args.db_path)
        for name in names:
            print(f"  - {name}")
    
    elif args.mode == 'stats':
        print("Kernel Statistics:")
        print("=" * 80)
        stats = get_kernel_statistics(args.db_path)
        print(f"{'Kernel Name':<40} {'Count':<8} {'Avg (ms)':<10} {'Total (ms)':<12}")
        print("-" * 80)
        for name, count, avg_dur, min_dur, max_dur, total_dur in stats:
            print(f"{name:<40} {count:<8} {avg_dur/1e6:<10.2f} {total_dur/1e6:<12.2f}")

if __name__ == "__main__":
    main()
