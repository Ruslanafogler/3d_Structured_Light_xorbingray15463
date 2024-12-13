import numpy as np
import argparse
import matplotlib.pyplot as plt



# def split():
#     list(imgs.transpose(2, 0, 1))

def make_binary(h, w):
    projection = np.zeros((h,w))
    num = len(bin(w-1))-2
    num2 = np.log2(w).astype(np.uint8)

    print("num is", num, "num2 is", num2)


    imgs_code = 255*np.fromfunction(lambda y,x,n: (x&(1<<(num-1-n))!=0), (h,w,num2), dtype=int).astype(np.uint8)
    for i in range(imgs_code.shape[2]):
        plt.figure(f"projector image {i}")
        plt.imshow(imgs_code[:,:,i], cmap='gray')
    print("shape of imgs_code", imgs_code.shape) 
    plt.show()

    #imlist = self.split(imgs_code)
    #return imlist    
    pass

def make_gray(h, w):
    pass

def make_xor(h, w):
    pass

def make_optimal(h, w):
    pass

def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')   
    parser.add_argument('-t', '--type') 

    args = parser.parse_args()
    print(args.type)

    h = 500
    w = 500

    if(args.type == "binary"):
        print("constructing binary patterns...")
        make_binary(h, w)
        print("done!")
    elif(args.type == "gray"):
        print("constructing gray patterns...")
        make_gray(h, w)
        print("done!")     
    elif(args.type == "xor"):
        print("constructing xor patterns...")
        make_xor(h, w)
        print("done!")   
    elif(args.type == "optimal"):
        print("constructing a la carte patterns...")
        make_optimal(h, w)
        print("done!")   
    else:
        print("unrecognized input. Not generating a pattern.")
         
    return


    


if __name__ == "__main__":
    main()