import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import pywt

def conv_to_mat(im,kernel_size):
    y_d=int(np.floor(kernel_size/2))
    y_u=int(im.shape[1]-np.ceil(kernel_size/2))
    x_l=int(np.floor(kernel_size/2))
    x_r=int(im.shape[0]-np.ceil(kernel_size/2))
    wid=int(np.floor(kernel_size/2))
    mat_=np.zeros([im.shape[0],im.shape[1],kernel_size*kernel_size*im.shape[2]],dtype='complex64')
    tar_=np.zeros([im.shape[0],im.shape[1],32],dtype='complex64')    

    for ii in range(x_l,x_r):
        for jj in range(y_d,y_u):
            
            tmp_=im[range(ii-wid,ii+wid+1),:,:][:,range(jj-wid,jj+wid+1),:]
            tar_[ii,jj,:]=tmp_[int(np.floor(kernel_size/2)),int(np.floor(kernel_size/2)),:]
            tmp_[int(np.floor(kernel_size/2)),int(np.floor(kernel_size/2)),:]=0
            mat_[ii,jj,:]=tmp_.reshape([kernel_size*kernel_size*im.shape[2]])
           
    return mat_,tar_ 
 
 
def conv_to_mat_comp(im,kernel_size):
    y_d=int(np.floor(kernel_size/2.0))
    y_u=int(im.shape[1]-np.ceil(kernel_size/2))
    x_l=int(np.floor(kernel_size/2.0))
    x_r=int(im.shape[0]-np.ceil(kernel_size/2))
    wid=int(np.floor(kernel_size/2))
    mat_=np.zeros([im.shape[0],im.shape[1],kernel_size*kernel_size*im.shape[2]],dtype='complex64')
    tar_=np.zeros([im.shape[0],im.shape[1],32],dtype='complex64')    

    for ii in range(x_l,x_r):
        for jj in range(y_d,y_u):
            
            tmp_=im[range(ii-wid,ii+wid+1),:,:][:,range(jj-wid,jj+wid+1),:]
            tar_[ii,jj,:]=tmp_[int(np.floor(kernel_size/2)),int(np.floor(kernel_size/2)),:]
            #tmp_[int(np.floor(kernel_size/2)),int(np.floor(kernel_size/2)),:]=0
            mat_[ii,jj,:]=tmp_.reshape([kernel_size*kernel_size*im.shape[2]],order='F')
           
    return mat_,tar_ 
    
def conv_to_mat_msk(im,kernel_size):
    y_d=int(np.floor(kernel_size/2))
    y_u=int(im.shape[1]-np.ceil(kernel_size/2))
    x_l=int(np.floor(kernel_size/2))
    x_r=int(im.shape[0]-np.ceil(kernel_size/2))
    wid=int(np.floor(kernel_size/2))
    mat_=np.zeros([im.shape[0],im.shape[1],32,kernel_size*kernel_size*im.shape[2]],dtype='complex64')
    tar_=np.zeros([im.shape[0],im.shape[1],32,32],dtype='complex64')    
    msk=np.zeros([im.shape[0],im.shape[1],32])   
    for ii in range(x_l,x_r):
        for jj in range(y_d,y_u):
            for kk in range(32):
                tmp_=im[range(ii-wid,ii+wid+1),:,:][:,range(jj-wid,jj+wid+1),:]
                tar_[ii,jj,kk,:]=tmp_[int(np.floor(kernel_size/2)),int(np.floor(kernel_size/2)),:]
                msk[ii,jj,kk]=kk                
                #tmp_[int(np.floor(kernel_size/2)),int(np.floor(kernel_size/2)),:]=0
                mat_[ii,jj,kk,:]=tmp_.reshape([kernel_size*kernel_size*im.shape[2]],order='F')
                #mat_[ii,jj,kk,kk]=0
    aa=mat_.shape[0]*mat_.shape[1]*mat_.shape[2] 
    mat_=  mat_.reshape([aa,mat_.shape[3]])   
    tar_=  tar_.reshape([aa,tar_.shape[3]]) 
    msk=msk.reshape([aa,1])  
    return mat_,tar_,msk 

      
def transform_image_to_complex(k):
    """ Compu"""
    us_complex=np.zeros((k.shape[0],k.shape[1],k.shape[2],1),dtype=np.complex_)
    for ii in range(k.shape[0]):   
      us_complex[ii,:,:,0].real=k[ii,:,:,0]
      us_complex[ii,:,:,0].imag=k[ii,:,:,1]
    return us_complex
    
    
def centered_crop(img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)
   left = int(np.ceil((width - new_width)/2.))
   top = int(np.ceil((height - new_height)/2.))
   right = int(np.floor((width + new_width)/2.))
   bottom = int(np.floor((height + new_height)/2.))
   cImg = img[top:bottom, left:right,:]
   return cImg
   
def wavelet_thresholding(im,thr):
        coeff=pywt.dwt2(im,'db4',axes=(0, 1))
        return coeff
 
def sum_of_square(im,axis=0):
    im = np.sqrt(np.sum((im*im.conjugate()),axis=axis))
    return im
def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k