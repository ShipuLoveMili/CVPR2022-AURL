
unset OMPI_MCA_plm_rsh_agent
mpirun --allow-run-as-root -np $NODE_NUM -H $NODE_IP_LIST -map-by slot --bind-to none \
    --mca btl_openib_want_cuda_gdr 1 -mca coll_fca_enable 0 --report-bindings \
    --display-map --mca btl_openib_rroce_enable 1 --mca pml ob1 --mca btl ^openib \
    --mca btl_openib_cpc_include rdmacm  --mca coll_hcoll_enable 0  --mca plm_rsh_no_tree_spawn 1 \
    -x NCCL_SOCKET_IFNAME=eth1 -x NCCL_DEBUG=INFO -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_SL=3 -x NCCL_NET_GDR_READ=1 \
    -x NCCL_IB_DISABLE=1 \
    -x LD_LIBRARY_PATH -x PATH \
    python3.6 main.py --n_epochs 150 --bs 32 --lr 0.1 --network r2plus1d_18 --dataset kinetics2both --nopretrained --solution 0\
    2>&1|tee task_out/out.log
