from convolution import convolution
from relu import relu
import os
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error




def update_parameters(filt, gradient_val, bias, biasgradient_value):
    alpha=0.001
    filt= filt - (alpha * gradient_val)
    bias=bias-(alpha * biasgradient_value)
    return filt, bias


def l2loss(param1, param2):
    loss=((param1 - param2) ** 2).mean(axis=None)
    gradient = param2 - param1
    return loss,gradient

Blurred_Img=[]
Sharp_Img=[]
level1=convolution(3, 64)
f_1=level1.filter
bias1=level1.bias

print(f_1, bias1)

level2=convolution(64, 64)
f_2=level2.filter
bias2=level2.bias



level3=convolution(64, 64)
f_3=level3.filter
bias3=level3.bias



level4=convolution(64, 64)
f_4=level4.filter
bias4=level4.bias


level5=convolution(64, 64)
f_5=level5.filter
bias5=level5.bias

level6=convolution(64, 64)
f_6=level6.filter
bias6=level6.bias


level7=convolution(64, 64)
f_7=level7.filter
bias7=level7.bias

level8=convolution(64, 3)
f_8=level8.filter
bias8=level8.bias

for subdirectory, dirs, files in os.walk("./"):
    for file_name in files:

        file_path = subdirectory + os.sep + file_name

        if file_path.endswith(".jpg") and ("Blur" in file_path):
            Blurred_Img.append(file_path)
        if file_path.endswith(".jpg") and ("sharp" in file_path):
            Sharp_Img.append(file_path)


for imag,imagSharp in zip(Blurred_Img, Sharp_Img):


    conv=convolution(3,3)
    reluo=relu()
    base_image = cv2.imread(imag)
    base_image=np.array(base_image)
    base_image=base_image[np.newaxis, ...]
    print(base_image.shape)
    Sharp_Img=cv2.imread(imagSharp)
    Sharp_Img=np.array(Sharp_Img)



    base_image, c1=conv.conv_forward(base_image, f_1, 1, 1, bias1)
    base_image=reluo.forward(base_image)

    base_image, c2 = conv.conv_forward(base_image, f_2, 1, 1, bias2)
    base_image = reluo.forward(base_image)

    base_image, c3 = conv.conv_forward(base_image, f_3, 1, 1, bias3)
    base_image = reluo.forward(base_image)

    base_image, c4 = conv.conv_forward(base_image, f_4, 1, 1, bias4)
    base_image = reluo.forward(base_image)

    base_image, c5 = conv.conv_forward(base_image, f_5, 1, 1, bias5)
    base_image = reluo.forward(base_image)

    base_image, c6 = conv.conv_forward(base_image, f_6, 1, 1, bias6)
    base_image = reluo.forward(base_image)

    base_image, c7 = conv.conv_forward(base_image, f_7, 1, 1, bias7)
    base_image = reluo.forward(base_image)

    base_image, c8 = conv.conv_forward(base_image, f_8, 1, 1, bias8)
    (inputimage, filter, stride, pad, bias) = c8
    inputimage = inputimage[np.newaxis, ...]
    cache8=(inputimage,filter,stride,pad,bias)

    loss,gradient = l2loss(base_image, Sharp_Img)
    print(loss)

    reduced_img, dW, Db=conv.conv_backward(gradient, c8)
    f_8, bias8=update_parameters(f_8, dW, bias8, Db)
    reduced_img=reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = conv.conv_backward(reduced_img, c7)
    f_7, bias7 = update_parameters(f_7, dW, bias7, Db)
    reduced_img = reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = conv.conv_backward(reduced_img, c6)
    f_6, bias6 = update_parameters(f_6, dW, bias6, Db)
    reduced_img = reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = conv.conv_backward(reduced_img, c5)
    f_5, bias5 = update_parameters(f_5, dW, bias5, Db)
    reduced_img = reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = conv.conv_backward(reduced_img, c4)
    f_4, bias4 = update_parameters(f_4, dW, bias4, Db)
    reduced_img = reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = convolution.conv_backward(reduced_img, c3)
    f_3, bias3 = update_parameters(f_3, dW, bias3, Db)
    reduced_img = reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = convolution.conv_backward(reduced_img, c2)
    f_2, bias2 = update_parameters(f_2, dW, bias2, Db)
    reduced_img = reluo.relu_backward(reduced_img)

    reduced_img, dW, Db = convolution.conv_backward(reduced_img, c1)
    f_1, bias1 = update_parameters(f_1, dW, bias1, Db)


# testing
testInput=[]
for subdirectory, dirs, files in os.walk("./result/input"):
    for file_name in files:
        #print os.path.join(subdir, file)
        file_path = subdirectory + os.sep + file_name

        if file_path.endswith(".jpg") or file_path.endswith(".png"):
            testInput.append(file_path)
for imag in testInput:
    count=1
    base_image = cv2.imread(imag)
    base_image = np.array(base_image)
    base_image = base_image[np.newaxis, ...]
    #200*200   to 500*500

    for i in range(150):
        base_image, cache1 = conv.conv_forward(base_image, f_1, 1, 1, bias1)
        base_image = reluo.forward(base_image)

        base_image, cache2 = conv.conv_forward(base_image, f_2, 1, 1, bias2)
        base_image = reluo.forward(base_image)

        base_image, cache3 = conv.conv_forward(base_image, f_3, 1, 1, bias3)
        base_image = reluo.forward(base_image)

        base_image, cache4 = conv.conv_forward(base_image, f_4, 1, 1, bias4)
        base_image = reluo.forward(base_image)

        base_image, cache5 = conv.conv_forward(base_image, f_5, 1, 1, bias5)
        base_image = reluo.forward(base_image)

        base_image, cache6 = conv.conv_forward(base_image, f_6, 1, 1, bias6)
        base_image = reluo.forward(base_image)

        base_image, cache7 = conv.conv_forward(base_image, f_7, 1, 1, bias7)
        base_image = reluo.forward(base_image)

        base_image, cache8 = conv.conv_forward(base_image, f_8, 1, 1, bias8)

        base_image = cv2.resize(imag, None, fx=1.004016, fy=1.004016, interpolation=cv2.INTER_CUBIC)
        base_image = base_image[np.newaxis, ...]

    imagname=file_path.replace(imag, "input", "output")
    cv2.imwrite(imagname, base_image)




