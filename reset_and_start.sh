cd /workspace/huangniu_det
rm -r Featurelibrary
rm -r visual
rm -r logs
rm out.log
taskset -c 0,1,2,3,4,5 python -u app.py > out.log

