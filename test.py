# -*- coding:utf-8 -*-
import cv2

from caffe_extractor import CaffeExtractor
from distance import cosine_similarity

def model_mobileface(do_mirror):
    model_dir = './'
    model_proto = model_dir + 'mobilefacenet-res2-6-10-2-dim128-opencv.prototxt'
    model_path  = model_dir + 'mobilefacenet-res2-6-10-2-dim128.caffemodel'
    image_size = (112, 112)
    extractor  = CaffeExtractor(model_proto, model_path, do_mirror = do_mirror, featLayer='fc1')
    return extractor, image_size

    
def model_factory(name, do_mirror):
    model_dict = {
        'mobileface':model_mobileface, 
    }
    model_func = model_dict[name]
    return model_func(do_mirror) 
    
def crop_image(img, imsize):
    h, w, c = img.shape
    x1 = (w - imsize[0])/2
    y1 = (h - imsize[1])/2
    crop_img = img[int(y1):(int(y1)+int(imsize[1])),int(x1):int(x1+imsize[0]),:]
    return crop_img

if __name__ == '__main__':
    extractor, imsize = model_factory('mobileface', False)
    
    img1 = cv2.imread('Aaron_Eckhart_0001.jpg')
    img2 = cv2.imread('Aaron_Peirsol_0002.jpg')
    
    crop_img1 = crop_image(img1, imsize)
    crop_img2 = crop_image(img2, imsize)
    
    #cv2.imwrite('crop_img1.bmp', crop_img1)
    #cv2.imwrite('crop_img2.bmp', crop_img2)
    
    feat1 = extractor.extract_feature(crop_img1)
    feat2 = extractor.extract_feature(crop_img2)

    print(feat1)
    print(feat2)
    print('cosine_similarity:'+str(cosine_similarity(feat1, feat2)))

