cmake -B build -H.
cmake --build build
# python3 test_overlap.py
nsys profile -f true -o overlap mpirun --allow-run-as-root -np 4 build/test_overlap 3 512
nsys profile -f true -o test_overlap_cpu mpirun --allow-run-as-root -np 4 build/test_overlap_cpu 3 512
nsys profile -f true -o sequential mpirun --allow-run-as-root -np 4 build/test_sequential
