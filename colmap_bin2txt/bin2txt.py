from aa import *
 
if __name__ == '__main__':
    
    path_read = '/home/perple/czbbzc/data/nerfbusters-dataset/aloe/colmap/sparse/0/cameras.bin'
    path_write = '/home/perple/czbbzc/data/nerfbusters-dataset/aloe/colmap/sparse/0/cameras.txt'

    cameras = read_cameras_binary(path_read)
    
    write_cameras_text(cameras, path_write)
    