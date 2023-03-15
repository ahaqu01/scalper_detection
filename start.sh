cd /workspace/huangniu_det
taskset -c 0,1,2,3,4,5 python -u app.py ./logs 6677 > out1.log #&
#taskset -c 6,7,8,9,10,11 python -u app.py ./logs2 6699 > out2.log &

