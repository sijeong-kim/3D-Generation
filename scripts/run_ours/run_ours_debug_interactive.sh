export WORKING_DIR=/workspace/3D-Generation
export BASE_DIR=${WORKING_DIR}

export SWEEP_NAME=debug
export PROMPTS=("a photo of a hamburger")

mkdir -p ${BASE_DIR}/debug/${SWEEP_NAME}

CMD="python ${WORKING_DIR}/hp_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    --sweep_config ${WORKING_DIR}/configs/text_ours_debug.yaml \
    --sweep_name ${SWEEP_NAME} \
    --prompts ${PROMPTS[*]} \
    --outdir=${BASE_DIR}/debug"

echo "[RUNNING COMMAND] $CMD"
echo ""

# Run the command and save output to file in real time
eval $CMD 2>&1 | tee ${BASE_DIR}/debug/${SWEEP_NAME}/output.out

echo ""
echo "====================================="
echo "EXPERIMENT completed!"
echo "Results saved in: ${BASE_DIR}/debug"
echo "Check debug/${SWEEP_NAME}/ for experiment runs"
echo "====================================="
