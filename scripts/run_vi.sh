export WORKING_DIR=/workspace/3D-Generation
export BASE_DIR=${WORKING_DIR}

export SWEEP_NAME=sigma_weight_viz
export PROMPTS=("a photo of a hamburger")

CMD="python ${WORKING_DIR}/hp_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    --sweep_config ${WORKING_DIR}/configs/text_ours_debug.yaml \
    --sweep_name ${SWEEP_NAME} \
    --prompts ${PROMPTS[*]} \
    --outdir=${BASE_DIR}/logs"

echo "[RUNNING COMMAND] $CMD"
echo ""

# Run the command and save output to file in real time
eval $CMD 2>&1 | tee ${BASE_DIR}/logs/output.out