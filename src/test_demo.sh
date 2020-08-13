#python demo.py ctdet --exp_id  coco_resdcn18_v1 --arch resdcn_18 --gpus 3
#python demo.py ctdet --demo /home/qinyuanze/code/center/CenterNet/images/ --load_mode /home/qinyuanze/code/center/CenterNet/models/ctdet_coco_dla_2x.pth --batch_size 1 --gpus 3 --num_workers 1
#CUDA_VISIBLE_DIVICES=3 python demo.py ctdet --demo /home/qinyuanze/code/center/CenterNet/images/ --load_mode /home/qinyuanze/code/center/CenterNet/models/ctdet_coco_dla_2x.pth --gpus 3

#python demo.py ctdet --demo  /home/qinyuanze/code/center/CenterNet/images/ --load_model  /home/qinyuanze/code/center/CenterNet/models/ctdet_coco_dla_2x.pth

python main.py ctdet --exp_id coco_res18_v1 --arch res_18 --batch_size 32 --master_batch 8 --lr 1.25e-4 --gpus 0,1  # 能运行