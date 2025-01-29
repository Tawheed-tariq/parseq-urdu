read : python read.py outputs/parseq/2024-12-28_09-02-01/checkpoints/epoch=19-step=623750-val_accuracy=88.8613-val_NED=96.0859.ckpt refine_iters:int=2 decode_ar:bool=false --images demo_images/urdu/test1/out/* > demo_images/urdu/test1/predicted.txt

test: python test.py outputs/parseq/2024-12-28_09-02-01/checkpoints/epoch=19-step=623750-val_accuracy=88.8613-val_NED=96.0859.ckpt refine_iters:int=2 decode_ar:bool=false --batch_size 32

generat trdg : python run.py -l ur -c 1000 -w 5 -f 64 -rk -rbl -b 3 --word_split -na 2 -t 5

create lmbd: python tools/create_lmdb_dataset.py data/out/ data/out/labels.txt data/val/custom/