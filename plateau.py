import train_linreg


angles = [60, 75, 90]

for angle in angles:
    print("##### angle = ", angle, " #####")
    train_linreg.train(dim=10, epochs=60,outdir="cone-angle{0}/".format(angle),switch_epoch=120, 
                       pretrain_size=2**14, angle=angle)
    
