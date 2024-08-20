import train_linreg
import os


angles = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
angle = angles[idx]

print("##### angle = ", angle, " #####")
train_linreg.train(dim=10, epochs=580,outdir="cone-angle-sphere_lastonly{0}/".format(angle),switch_epoch=1020, 
                   pretrain_size=2**9, angle=angle)
    
