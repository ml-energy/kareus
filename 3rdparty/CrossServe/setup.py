from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys
import shutil


def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][0]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except Exception as e:
        return "no_cuda"


def check_make():
    if not shutil.which("make"):
        raise RuntimeError("Make is required to build MSCCLPP. Please install Make first.")


class BuildCSRC(build_ext):
    def run(self):
        # Check build requirements
        check_make()

        # Build MSCCLPP first
        mscclpp_dir = os.path.join("3rdparty", "mscclpp")
        mscclpp_build_dir = os.path.join(mscclpp_dir, "build")
        install_prefix = os.environ.get("MSCCLPP_INSTALL_PREFIX", "/usr/local/mscclpp")

        if not os.path.exists(mscclpp_build_dir):
            os.makedirs(mscclpp_build_dir)

        try:
            # Build MSCCLPP
            subprocess.check_call(
                [
                    "cmake",
                    "-DCMAKE_BUILD_TYPE=Release",
                    f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
                    "-DBUILD_PYTHON_BINDINGS=OFF",
                    "-B" + os.path.join(mscclpp_build_dir),
                    "-H" + mscclpp_dir,
                ],
            )
            cpu_count = os.cpu_count() or 1
            subprocess.check_call(["make", f"-j{cpu_count}", "mscclpp", "mscclpp_static"], cwd=mscclpp_build_dir)
            subprocess.check_call(["make", "install/fast"], cwd=mscclpp_build_dir)

            # Build CSRC/comm
            root_dir = os.path.dirname(os.path.abspath(__file__))
            csrc_dir = os.path.join(root_dir, "csrc", "comm")
            subprocess.check_call(
                [
                    "cmake",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')",
                    "-B" + os.path.join(csrc_dir, "build"),
                    "-H" + csrc_dir,
                ],
            )
            subprocess.check_call(["cmake", "--build", os.path.join(csrc_dir, "build"), "--parallel", f"{cpu_count}"])

            # Run the standard build_ext
            build_ext.run(self)

            # Get absolute path for include_dirs
            root_dir = os.path.dirname(os.path.abspath(__file__))

            # Copy the msccl_comm module to cfuser package
            msccl_build_dir = os.path.join(root_dir, "csrc/comm/build")
            cfuser_dir = os.path.join(root_dir, "cfuser/msccl_comm")
            if os.path.exists(msccl_build_dir):
                msccl_lib = os.path.join(msccl_build_dir, "msccl_comm*.so")
                # Use glob to find the exact file name
                import glob

                msccl_files = glob.glob(msccl_lib)
                if msccl_files:
                    msccl_file = msccl_files[0]
                    shutil.copy2(msccl_file, cfuser_dir)

            # Copy the libcustom_nccl_all2all.so module to cfuser package
            custom_nccl_path = os.path.join(root_dir, "csrc/comm/build/libcustom_nccl_all2all.so")
            nccl_target_path = os.path.join(root_dir, "cfuser/core/distributed/libcustom_nccl_all2all.so")
            if os.path.exists(custom_nccl_path):
                shutil.copy2(custom_nccl_path, nccl_target_path)

        except subprocess.CalledProcessError as e:
            print(f"Error building project: {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Unexpected error during build: {e}", file=sys.stderr)
            raise


if __name__ >= "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("cfuser/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    setup(
        name="cfuser",
        author="Members of Symbiotic and Usesys Lab",
        author_email="runyulu@umich.edu",
        packages=find_packages(),
        install_requires=[
            # "torch>=2.1.0",  # Setting to >=2.5 reinstalls torch even when container has PyTorch>=2.5.0. 2.4 has important features like dist.breakpoint anyway
            # "accelerate>=0.33.0",
            # "diffusers>=0.31.0",
            # "transformers>=4.39.1",
            # "sentencepiece>=0.1.99",
            # "beautifulsoup4>=4.12.3",
            # "distvae",
            # "fastapi>=0.110.0",
            # "uvicorn>=0.30.0",
            # "flash_attn>=2.6.3",
            # "uvloop",
            # "pytest",
            # "flask",
            # "opencv-python>=4.5.5.64",
            # "black",
        ],
        extras_require={
            # "flash_attn": [
            #     "flash_attn>=2.6.3",
            # ],
        },
        cmdclass={"build_ext": BuildCSRC},
        package_data={
            "cfuser": ["msccl_comm*.so", "libcustom_nccl_all2all.so"],  # Include the .so file in the package
        },
        url="https://github.com/LRY89757/CrossServe.",
        description="CrossServe: A Cross-Request Serving Engine for Diffusion Transformers (DiTs) on multi-GPU Clusters",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
