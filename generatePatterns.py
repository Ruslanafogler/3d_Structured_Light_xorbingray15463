import numpy as np
import argparse
import matplotlib.pyplot as plt
import skimage.io as ski_io

def write_image(path, data, is_float=True):
    if(is_float):
        data = np.clip(data, 0.0, 1.0)
        data = data*255
    saved = data.astype(np.uint8)
    ski_io.imsave(path, saved)

#inspiration taken from https://github.com/elerac/structuredlight/blob/master/structuredlight/binary.py
def make_binary(h, w):
    show_stripes = False
    num = len(bin(w-1))-2
    num2 = np.log2(w).astype(np.uint8)

    print("num is", num, "num2 is", num2)
    imgs_code = 255*np.fromfunction(lambda y,x,n: (x&(1<<(num-1-n))!=0), (h,w,num), dtype=int).astype(np.uint8)
    for i in range(imgs_code.shape[2]):
        write_image(f"./patterns/binary/bin_{i}", False)
        if(show_stripes):
            plt.figure(f"projector image {i}")
            plt.imshow(imgs_code[:,:,i], cmap='gray')
    print("shape of imgs_code", imgs_code.shape) 
    if(show_stripes):
        plt.show()  
    return

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