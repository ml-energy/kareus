#! /bin/bash

# bash tests/run_A40.sh 2>&1 | tee log.log

OUTPUT_JSON="log/tests/test_overlap_perf_correctness.json"
# Array of batch sizes to test
BATCH_SIZES=(1 2 3 4)
# ULYSSES_DEGREES=(1 2 4)
ULYSSES_DEGREES=(4)
RING_DEGREE=1
NUM_STEPS=8

# Array of image sizes to test
declare -A DIMENSIONS=(
    ["1280x720"]="720 1280"
    ["1920x1080"]="1080 1920"
    ["1920x960"]="960 1920"
    ["256x256"]="256 256"
    ["512x512"]="512 512"
    ["720x720"]="720 720"
    ["1024x1024"]="1024 1024"
    ["2048x2048"]="2048 2048"
)

# Run tests with and without stream
for USE_STREAM in true; do
    STREAM_FLAG=""
    if [ "$USE_STREAM" = false ]; then
        STREAM_FLAG="--no_stream"
    fi

    # Without nsys
    for ULYSSES_DEGREE in "${ULYSSES_DEGREES[@]}"; do
        for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
            for SIZE_KEY in "${!DIMENSIONS[@]}"; do
                read HEIGHT WIDTH <<< "${DIMENSIONS[$SIZE_KEY]}"
                CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/test_overlap_perf_correctness.py \
                    --ulysses_degree $ULYSSES_DEGREE \
                    --ring_degree $RING_DEGREE \
                    --height $HEIGHT \
                    --width $WIDTH \
                    --batch_size $BATCH_SIZE \
                    --num_inference_steps $NUM_STEPS \
                    $STREAM_FLAG \
                    --output_json $OUTPUT_JSON
            done
        done
    done

    # With nsys (commented out)
    # for SIZE in "${SIZES[@]}"; do
    #     SUFFIX=""
    #     if [ "$USE_STREAM" = false ]; then
    #         SUFFIX="_no_stream"
    #     fi
    #
    #     CUDA_VISIBLE_DEVICES=0,1,2,3 nsys profile \
    #         --force-overwrite true \
    #         -w true \
    #         --capture-range=cudaProfilerApi \
    #         -o "log/tests/test_overlap_perf_correctness_bs${BATCH_SIZE}_u${ULYSSES_DEGREE}_r${RING_DEGREE}_h${SIZE}_w${SIZE}${SUFFIX}" \
    #         python tests/test_overlap_perf_correctness.py \
    #         --ulysses_degree $ULYSSES_DEGREE \
    #         --ring_degree $RING_DEGREE \
    #         --height $SIZE \
    #         --width $SIZE \
    #         --batch_size $BATCH_SIZE \
    #         --num_inference_steps $NUM_STEPS \
    #         $STREAM_FLAG
    # done
done
