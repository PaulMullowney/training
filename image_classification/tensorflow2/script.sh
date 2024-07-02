#!/bin/bash -l
batch_size=512
small=
rocblas_bench=
rprof=
rprofstats=
ngpus=1
latest=
dataset_divider=1
enable_eager=
roctx_start=1
roctx_stop=2
data_format=channels_first
for i in "$@"; do
    case "$1" in
	# Batch size per device
        -df=*|--data_format=*)
	    data_format="${i#*=}"
	    shift # past argument=value
	    ;;
	# Batch size per device
        -batch_size=*|--batch_size=*)
	    batch_size="${i#*=}"
	    shift # past argument=value
	    ;;
	# Number of GPUs to use
	-g=*|--num_gpus=*)
	    ngpus="${i#*=}"
	    shift # past argument=value
	    ;;	
	# Number of GPUs to use
	-dd=*|--dataset_divider=*)
	    dataset_divider="${i#*=}"
	    shift # past argument=value
	    ;;	
	# Value of global step to start the GPU profiling
	-roctx_start=*|--roctx_start=*)
	    roctx_start="${i#*=}"
	    shift # past argument=value
	    ;;	
	# Value of global step to stop the GPU profiling
	-roctx_stop=*|--roctx_stop=*)
	    roctx_stop="${i#*=}"
	    shift # past argument=value
	    ;;	
	# whether or not to run eagerly
	-ee|--enable_eager)
	    enable_eager="--enable_eager"
	    shift # past argument=value
	    ;;	
	# generate rocblas bench sizes
        -rbb|--rocblas_bench)
	    rocblas_bench=YES
	    shift # past argument=value
	    ;;
	# Do rocprof profiling
	-rp|--rocprof)
	    rprof=YES
	    shift # past argument=value
	    ;;
	# whether or not to use Alessandro's latest rocblas build. Need to do this better.
	-l|--latest)
	    latest=YES
	    shift # past argument=value
	    ;;	
	--)
	    shift
	    break
	    ;;
    esac
done

#echo "Clear page cache"
#sync && /sbin/sysctl vm.drop_caches=3
#export ROCR_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=24
export OMP_PLACES=cores
export OMP_PROC_BIND=master
export DATASETS_NUM_PRIVATE_THREADS=$OMP_NUM_THREADS

BATCH_SIZE=$((batch_size*ngpus))
steps_per_loop=$(((1281167/dataset_divider) / BATCH_SIZE ))
echo "steps_per_loop="${steps_per_loop}
ostr="_1_div_${dataset_divider}"
##########################################
# point to the right library and output dir
##########################################
if [[ $latest ]];
then
    ofile="_latest"
    export LD_LIBRARY_PATH=/home/afanfari/SOURCES/rocBLAS_latest/build/release/rocblas-install/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/home/pmullown/TF/MIOpen/build/lib:$LD_LIBRARY_PATH
    
else
    ofile=""
    export LD_LIBRARY_PATH=/home/afanfari/SOURCES/rocBLAS/build/release/rocblas-install/lib:$LD_LIBRARY_PATH
fi

if [[ $enable_eager ]];
then
    ofile=${ofile}"_eager"
fi

if [ "$data_format" = "channels_first" ]; then
    ostr=${ofile}${ostr}_nhwc
else
    ostr=${ofile}${ostr}_nchw
fi

##########################################
# Make the output dir
##########################################
odir="output_"${batch_size}_${ngpus}
if [[ ! -d ${odir} ]]; then
    mkdir ${odir}
fi

##########################################
# generate rocblas bench
##########################################
if [[ $rocblas_bench ]];
then
    export ROCBLAS_LAYER=2
    export ROCBLAS_LOG_BENCH_PATH=${odir}/rocblas_bench${ostr}.txt
fi

##########################################
# make the rocprof command
##########################################
if [[ $rprof ]];
then
    #--trace-start off
    #rprof_cmd="rocprof --trace-start off --roctx-trace --sys-trace -o ${odir}/results${ostr}.csv"
    rprof_cmd="rocprof --trace-start on --hsa-trace --roctx-trace --sys-trace -o ${odir}/results${ostr}.csv"
    echo ${rprof_cmd}
else
    rprof_cmd=
fi

export TF_FORCE_GPU_ALLOW_GROWTH=true
export ROCM_PATH=/opt/rocm-6.1.2/
#export GPU_MAX_HW_QUEUES=1
#export MIOPEN_FIND_ENFORCE=3

#numactl -C 0-23 -m 0 
${rprof_cmd} python3 ./resnet_ctl_imagenet_main.py \
	--base_learning_rate=8.5 \
	--batch_size=${BATCH_SIZE} \
	--clean \
	--data_dir=/home/rajarora/applications/MLPERF/benchmarks/data/resnet/train \
	--datasets_num_private_threads=24 \
	--dtype=fp32 \
	--device_warmup_steps=1 \
	--noenable_device_warmup \
	--noenable_xla \
	--epochs_between_evals=4 \
	--noeval_dataset_cache \
	--eval_offset_epochs=2 \
	--eval_prefetch_batchs=192 \
	--label_smoothing=0.1 \
	--lars_epsilon=0 \
	--log_steps=125 \
	--lr_schedule=polynomial \
	--model_dir=/home/pmullown/TF/training/image_classification/tensorflow2/output \
	--momentum=0.9 \
	--num_accumulation_steps=2 \
	--num_classes=1000 \
	--num_gpus=${ngpus} \
	--optimizer=LARS \
	--noreport_accuracy_metrics \
	--single_l2_loss_op \
	--noskip_eval \
	--steps_per_loop=${steps_per_loop} \
	--target_accuracy=0.759 \
	--notf_data_experimental_slack \
	--tf_gpu_thread_mode=gpu_private \
	--notrace_warmup \
	--train_epochs=4 \
	--notraining_dataset_cache \
	--training_prefetch_batchs=128 \
	--nouse_synthetic_data \
	--warmup_epochs=5 \
	--dataset_divider=${dataset_divider} \
	${enable_eager} \
	--roctx_start=${roctx_start} \
	--roctx_stop=${roctx_stop} \
	--data_format=${data_format} \
	--weight_decay=0.0002 &> ${odir}/tf_out${ostr}.txt
