import train_linreg


dims = [5,10,15,20,25]

for dim in dims:
    print("##### dim = ", dim, " #####")
    train_linreg.train(dim=dim, epochs=250,outdir="output{0}/".format(dim))
    
