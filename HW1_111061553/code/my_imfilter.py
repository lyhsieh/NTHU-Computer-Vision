import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================

    # ============================== Start OF YOUR CODE ===============================

    output = np.zeros_like(image)
    v_offset = (imfilter.shape[0] - 1 ) // 2
    h_offset = (imfilter.shape[1] - 1 ) // 2
    # print(f'original shape = {image.shape}')
    ch0 = np.pad(image[:, :, 0], (v_offset, h_offset), mode='constant')
    ch1 = np.pad(image[:, :, 1], (v_offset, h_offset), mode='constant')
    ch2 = np.pad(image[:, :, 2], (v_offset, h_offset), mode='constant')
    # print(f'ch0.shape = {ch0.shape}')
    # r, c stand for the row and column at "upper left corner"
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            output[r][c][0] = np.sum(imfilter * ch0[r: r + imfilter.shape[0], c: c + imfilter.shape[1]])
            output[r][c][1] = np.sum(imfilter * ch1[r: r + imfilter.shape[0], c: c + imfilter.shape[1]])
            output[r][c][2] = np.sum(imfilter * ch2[r: r + imfilter.shape[0], c: c + imfilter.shape[1]])
    # print('finish')
    
    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
    
    return output