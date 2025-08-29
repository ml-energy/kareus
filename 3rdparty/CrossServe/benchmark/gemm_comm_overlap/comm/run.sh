cmake -B build -H.
cmake --build build --parallel 4
python3 test_all2all.py
