import os
import cv2
from PIL import Image
img = Image.open('/home/bevfusion/data/20240617-720/camera/cam_front/3428587597.861514.jpg')
print("img size:  ",img.size)
# # 原始摄像头图像路径
# input_base = 'data/20240617-720/camera'  # 替换为你的实际路径
# output_base = 'data/20240617-720/camera_resize'  # 新建的目标文件夹

# # 三个摄像头视角
# cameras = ['front_top', 'side_left', 'side_right']

# # resize目标大小
# target_size = (704, 256)  # 注意：OpenCV中是(width, height)

# for cam in cameras:
#     input_dir = os.path.join(input_base, cam)
#     output_dir = os.path.join(output_base, cam)
    
#     # 创建输出文件夹（如果不存在）
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
#     for img_file in image_files:
#         input_path = os.path.join(input_dir, img_file)
#         output_path = os.path.join(output_dir, img_file)
        
#         # 读取图像
#         img = cv2.imread(input_path)
        
#         # 调整尺寸
#         resized_img = cv2.resize(img, target_size)
        
#         # 保存图像
#         cv2.imwrite(output_path, resized_img)

# print("图像resize完成并已保存至:", output_base)