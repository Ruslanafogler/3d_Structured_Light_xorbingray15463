import numpy as np
import argparse
import matplotlib.pyplot as plt
from util import write_image


##ALLL OF THIS CODE IS INSPIRED BY https://github.com/elerac/structuredlight/blob/master/structuredlight


def make_binary(h, w):
    num = len(bin(w-1))-2
    print("number of images", num)
    #code obtained from https://github.com/elerac/structuredlight/blob/master/structuredlight/binary.py
    imgs_code = 255*np.fromfunction(lambda y,x,n: (x&(1<<(num-1-n))!=0), (h,w,num), dtype=int).astype(np.uint8)
    for i in range(imgs_code.shape[2]):
        write_image(f"./patterns2/binary/bin_{i}.tiff", imgs_code, False)

    #     plt.figure(f"projector image {i}")
    #     plt.imshow(imgs_code[:,:,i], cmap='gray')
    # print("shape of imgs_code", imgs_code.shape) 
    # plt.show()

    return


def get_gray(x, n, bit_width):
    return ((x^(x>>1))&(1<<(bit_width-1-n))!=0)


def make_gray(h, w):
    num = len(bin(w-1))-2
    print("number of images", num)
    #code obtained from https://github.com/elerac/structuredlight/blob/master/structuredlight/binary.py
    imgs_code = 255*np.fromfunction(lambda y,x,n: get_gray(x,n,num), (h,w,num), dtype=int).astype(np.uint8)
    for i in range(imgs_code.shape[2]):
        write_image(f"./patterns2/gray/gray_{i}.tiff", imgs_code[:,:,i], False)
    return

def generate_gray_code(n):
    """Generates a Gray code sequence of length 2^n."""
    gray_code = [0]
    for i in range(n):
        gray_code += [x ^ (1 << i) for x in reversed(gray_code)]
    return gray_code



def calculate_min_sw(gray_code):
    """Calculates the minimum stripe width of a Gray code sequence."""
    min_sw = float('inf')
    for i in range(len(gray_code) - 1):
        current_sw = bin(gray_code[i] ^ gray_code[i+1]).count('1')
        min_sw = min(min_sw, current_sw)
    return min_sw

def find_max_min_sw_gray_code(n):
    """Finds the Gray code with the maximum minimum stripe width for a given number of bits."""
    max_min_sw = 0
    best_gray_code = None
    for i in range(2**n):
        gray_code = generate_gray_code(n)[i:]
        min_sw = calculate_min_sw(gray_code)
        if min_sw > max_min_sw:
            max_min_sw = min_sw
            best_gray_code = gray_code
    return best_gray_code, max_min_sw


def make_gray_sw(h, w):
    num = len(bin(w-1))-2
    print("number of images", num)
    #code obtained from https://github.com/elerac/structuredlight/blob/master/structuredlight/binary.py
    #imgs_code = 255*np.fromfunction(lambda y,x,n: ((x^(x>>1))&(1<<(num-1-n))!=0), (h,w,num), dtype=int).astype(np.uint8)
    for i in range(1, num):
        gray_sw, sw = find_max_min_sw_gray_code(i)
        print("gray sw is", gray_sw)
        gray_sw = np.array(gray_sw)
        print("2 gray sw is", gray_sw)

        gray_sw = np.repeat(gray_sw[np.newaxis, :], h, axis=1)




        write_image(f"./patterns2/graysw/gray_{i}.tiff",gray_sw, False)
    return

def make_xor(h, w, doXor4):
    if(doXor4):
        xorver = 4
        last_ind = -1
    else:
        xorver = 2
        last_ind = -2

    num = len(bin(w-1))-2

    imgs_gray = 255*np.fromfunction(lambda y,x,n: ((x^(x>>1))&(1<<(num-1-n))!=0), (h,w,num), dtype=int).astype(np.uint8)
    
    # Convert gray code to xor code
    imgs_xor = np.empty((h, w, num), dtype=np.uint8)
    img_last = imgs_gray[:,:,last_ind].copy()
    for i in range(num):
        imgs_xor[:,:,i] = np.bitwise_xor(imgs_gray[:,:,i], img_last)
    imgs_xor[:,:,last_ind] = img_last.copy()
    for i in range(imgs_xor.shape[2]):
        write_image(f"./patterns2/xor{xorver}/xor{xorver}_{i}.tiff", imgs_xor[:,:,i], False)
    return


def main():
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')   
    parser.add_argument('-t', '--type') 

    args = parser.parse_args()
    print(args.type)

    h = 2048
    w = 2048

    if(args.type == "binary"):
        print("constructing binary patterns...")
        make_binary(h, w)
        print("done!")
    elif(args.type == "gray"):
        print("constructing gray patterns...")
        make_gray_sw(h, w)
        print("done!")     
    elif(args.type == "xor"):
        print("constructing xor patterns...")
        make_xor(h, w)
        print("done!")    
    else:
        print("unrecognized input. Not generating a pattern.")
         
    return


    


if __name__ == "__main__":
    main()