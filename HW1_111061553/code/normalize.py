def normalize(img):
    ''' Function to normalize an input array to 0-1 '''
    img_min = img.min()
    img_max = img.max()
    scale = img_max - img_min
    return (img - img_min) / scale if not scale==0 else img
