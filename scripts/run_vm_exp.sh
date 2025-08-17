export WORKING_DIR=/workspace/3D-Generation
export BASE_DIR=${WORKING_DIR}

########################################################
# Experiment 1: Repulsion lambda sweep with multiple prompts
########################################################

export SWEEP_NAME=exp1_repulsion_lambda_sweep

CMD="python ${WORKING_DIR}/hp_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    --sweep_config ${WORKING_DIR}/configs/text_ours_exp.yaml \
    --sweep_name ${SWEEP_NAME} \
    --outdir=${BASE_DIR}/exp"

echo "[RUNNING COMMAND] $CMD"
echo ""

# Run the command and save output to file in real time
eval $CMD 2>&1 | tee ${BASE_DIR}/exp/${SWEEP_NAME}/output.out

echo ""
echo "====================================="
echo "EXPERIMENT completed!"
echo "Results saved in: ${BASE_DIR}/exp"
echo "Check exp/${SWEEP_NAME}/ for experiment runs"
echo "====================================="


########################################################
# Experiment 1: Baseline (without repulsion)
########################################################
export SWEEP_NAME=exp1_wo_method

CMD="python ${WORKING_DIR}/hp_ours.py \
    --config ${WORKING_DIR}/configs/text_ours.yaml \
    --sweep_config ${WORKING_DIR}/configs/text_ours_exp.yaml \
    --sweep_name ${SWEEP_NAME} \
    --outdir=${BASE_DIR}/exp"

echo "[RUNNING COMMAND] $CMD"
echo ""

# Run the command and save output to file in real time
eval $CMD 2>&1 | tee ${BASE_DIR}/exp/${SWEEP_NAME}/output.out

echo ""
echo "====================================="
echo "EXPERIMENT completed!"
echo "Results saved in: ${BASE_DIR}/exp"
echo "Check exp/${SWEEP_NAME}/ for experiment runs"
echo "====================================="
