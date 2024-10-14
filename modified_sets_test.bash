mkdir results/
for test in active over part_fwd part_bwd shift_fwd shift_bwd vary_reg vary_rnd
do
    python3.9 main.py --config test_${test}.yaml
    #epoch=  53 #$(grep -o -E "Epoch number: [0-9]+" /work_dir/temp/log.txt|grep -o -E '[0-9]+')
    date=$(date '+%d-%m')
    mkdir results/test_${test}/${date}
    mv work_dir/dhg14-28/test/* results/test_${test}/${date}/
    # mv work_dir/temp/runs-${epoch}-* results/test_${test}/${date}
    # mv work_dir/temp/epoch${epoch}_test_each_class_acc.csv results/test_${test}/${date}
    # mv work_dir/temp/log.txt results/test_${test}/${date}
    # rm work_dir/temp/*
    rm work_dir/dhg14-28/test/14joint_1/*
done

# $ Python3.9 main.py --config test_part_bwd.yaml
# $ mkdir results/test_part_bwd/${date}
# $ mv /work_dir/temp/ /results/test_part_bwd/${date}
# $ Python3.9 main.py --config test_part_fwd.yaml
# $ mkdir results/test_part_fwd/${date}
# $ mv /work_dir/temp/ /results/test_part_fwd/${date}
# $ Python3.9 main.py --config test_shift_bwd.yaml
# $ mkdir results/test_shift_bwd/${date}
# $ mv /work_dir/temp/ /results/test_shift_bwd/${date}
# $ Python3.9 main.py --config test_shift_fwd.yaml
# $ mkdir results/test_shift_fwd/${date}
# $ mv /work_dir/temp/ /results/test_shift_fwd/${date}
# $ Python3.9 main.py --config test_vary_reg.yaml
# $ mkdir results/test_vary_reg/${date}
# $ mv /work_dir/temp/ /results/test_vary_reg/${date}
# $ Python3.9 main.py --config test_vary_rnd.yaml
# $ mkdir results/test_vary_rnd/${date}
# $ mv /work_dir/temp/ /results/test_vary_rnd/${date}

