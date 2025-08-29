# rm -rf build
cmake -Bbuild -H.
cmake --build build --parallel 16
nsys profile -f true -o overlap_all2allv_gemm mpirun --allow-run-as-root -np 4 ./build/test_overlap 3 512
