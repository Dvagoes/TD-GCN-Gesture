$ mkdir results/
$ Python3.9 main.py --config test_over.yaml
$ mkdir results/test_over/${date}
$ mv /work_dir/temp/ /results/test_over/${date}
$ Python3.9 main.py --config test_part_bwd.yaml
$ mkdir results/test_part_bwd/${date}
$ mv /work_dir/temp/ /results/test_part_bwd/${date}
$ Python3.9 main.py --config test_part_fwd.yaml
$ mkdir results/test_part_fwd/${date}
$ mv /work_dir/temp/ /results/test_part_fwd/${date}
$ Python3.9 main.py --config test_shift_bwd.yaml
$ mkdir results/test_shift_bwd/${date}
$ mv /work_dir/temp/ /results/test_shift_bwd/${date}
$ Python3.9 main.py --config test_shift_fwd.yaml
$ mkdir results/test_shift_fwd/${date}
$ mv /work_dir/temp/ /results/test_shift_fwd/${date}
$ Python3.9 main.py --config test_vary_reg.yaml
$ mkdir results/test_vary_reg/${date}
$ mv /work_dir/temp/ /results/test_vary_reg/${date}
$ Python3.9 main.py --config test_vary_rnd.yaml
$ mkdir results/test_vary_rnd/${date}
$ mv /work_dir/temp/ /results/test_vary_rnd/${date}
