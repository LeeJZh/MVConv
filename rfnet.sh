pip3 install --upgrade torch torchvision
export PYTHONPATH=./
export HDF5_USE_FILE_LOCKING=FALSE

python3 acnet/do_acnet.py -a src56 -b acb
sleep 60
python3 acnet/do_acnet.py -a src56 -b base

sleep 60
python3 rfnet/do_rfnet.py -a src56 -b rfb --scale 2 --alpha 0. -e 500

sleep 60
python3 rfnet/do_rfnet.py -a src56 -b rfb --scale 3 --alpha 0. -e 500

sleep 60
python3 ksnet/do_ksnet.py -a src56 -b ksb

sleep 60
python3 fanet/do_fanet.py -a src56 -b fab

sleep 60
# python -m torch.distributed.launch --nproc_per_node=1 acnet/do_acnet.py -a sres18 -b base

sleep 60
# python -m torch.distributed.launch --nproc_per_node=1 acnet/do_acnet.py -a sres18 -b acb

sleep 60
# python -m torch.distributed.launch --nproc_per_node=1 rfnet/do_rfnet.py -a sres18 -b rfb --scale 3 --alpha 0.0
