from PIL import Image
import os
from tqdm import tqdm

# 马赛克遮挡函数
def apply_mosaic(image, box):
    region = image.crop(box)
    region = region.resize((1, 1), Image.NEAREST).resize(region.size, Image.NEAREST)
    image.paste(region, box)
    return image

def clips_mosaic(original_image_dir, output_dir= '/home/pernod/data'):

    photos = [picture for picture in os.listdir(original_image_dir) if picture.endswith(('.png', '.jpg', '.jpeg'))]
    
    for photo in tqdm(photos, desc="Processing images"):
        image_path = os.path.join(original_image_dir, photo)
        image = Image.open(image_path)
        width, height = image.size 
        img_name = os.path.basename(photo).split('.')[0]
        os.makedirs(f'{output_dir}/CBLPRD-330k/{img_name}', exist_ok=True)

        for i in range(5):
            # 计算遮挡区域的起始和结束位置
            start_ratio = i * 0.2
            end_ratio = (i + 1) * 0.2
            x1 = int(start_ratio * width)
            x2 = int(end_ratio * width)

            # 修正最后一个区域的右边界
            if i == 4:
                x2 = width

            # 定义遮挡区域（覆盖整个高度）
            box = (x1, 0, x2, height)

            # 复制原始图片并应用马赛克
            img_copy = image.copy()
            img_copy = apply_mosaic(img_copy, box)
            img_name = os.path.basename(photo).split('.')[0]
            # 保存结果
            img_copy.save(f'{output_dir}/CBLPRD-330k/{img_name}/mosaic_{i + 1}.png')

            # 将图片分成5个等宽的区域进行遮挡




if __name__ == "__main__":
    # 主程序
    image_path = '/home/pernod/CBLPRD-330k_v1/CBLPRD-330k'  # 替换为你的图片路径
    output_path = '/home/pernod/data/CBLPRD_20percent'  # 替换为你想保存图片的路径
    clips_mosaic(image_path, output_path)
    print("处理完成！")