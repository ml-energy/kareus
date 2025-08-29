########################################################
git submodule update --init --recursive
########################################################
# build mscclpp

cd ./3rdparty/mscclpp

# install dependencies
apt update
apt install wget
apt install libspdlog-dev
# apt install python3-pybind11
apt-get install libglib2.0-0
apt install pigz
apt-get install nlohmann-json3-dev

ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1 /usr/lib/x86_64-linux-gnu/libibverbs.so

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/mscclpp -DBUILD_PYTHON_BINDINGS=OFF ..
make -j mscclpp mscclpp_static
make install/fast
########################################################
# build the csrc/comm
cd /workspaces/CrossServe/csrc/comm
rm -rf build
cmake -Bbuild -H.
cmake --build build --parallel 4
# mpirun --allow-run-as-root -np 4 ./build/test_nccl_all2all
mpirun --allow-run-as-root -np 4 ./build/test_mscclpp 3 1024
########################################################
# build the csrc/gemm
cd /workspaces/CrossServe/csrc/gemm
rm -rf build
cmake -Bbuild -H.
cmake --build build --parallel 4
./build/test_gemm
########################################################
