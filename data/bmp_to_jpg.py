from PIL import Image
import os
import sys

def convert_bmp_to_jpg(input_path, output_path=None, quality=95):
    """
    将BMP图像转换为JPG格式
    
    参数:
        input_path: BMP文件路径
        output_path: 输出JPG文件路径，如果为None则使用相同文件名但扩展名改为.jpg
        quality: JPG质量，范围1-100，默认95
    """
    try:
        # 如果未指定输出路径，则使用输入路径但更改扩展名
        if output_path is None:
            # 获取输入文件所在目录
            input_dir = os.path.dirname(input_path)
            # 创建jpg子目录
            jpg_dir = os.path.join(input_dir, "jpg")
            if not os.path.exists(jpg_dir):
                os.makedirs(jpg_dir)
            # 设置输出路径为jpg子目录中
            filename = os.path.basename(input_path)
            output_path = os.path.join(jpg_dir, os.path.splitext(filename)[0] + '.jpg')
        
        # 打开BMP图像
        img = Image.open(input_path)
        
        # 保存为JPG格式
        img.save(output_path, 'JPEG', quality=quality)
        
        print(f"转换成功: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"转换失败: {input_path}, 错误: {str(e)}")
        return False

def batch_convert(directory, quality=95):
    """批量转换目录中的所有BMP文件为JPG"""
    success_count = 0
    fail_count = 0
    
    # 创建jpg子目录
    jpg_dir = os.path.join(directory, "jpg")
    if not os.path.exists(jpg_dir):
        os.makedirs(jpg_dir)
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.bmp'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(jpg_dir, os.path.splitext(filename)[0] + '.jpg')
            
            if convert_bmp_to_jpg(input_path, output_path, quality):
                success_count += 1
            else:
                fail_count += 1
    
    print(f"批量转换完成: 成功 {success_count} 个, 失败 {fail_count} 个")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python bmp_to_jpg.py <BMP文件路径或目录路径> [JPG质量(1-100)]")
    else:
        path = sys.argv[1]
        quality = 100
        if len(sys.argv) >= 3:
            try:
                quality = int(sys.argv[2])
                if quality < 1 or quality > 100:
                    print("质量参数必须在1-100之间，使用默认值95")
                    quality = 95
            except:
                print("质量参数必须是整数，使用默认值95")
        
        if os.path.isdir(path):
            batch_convert(path, quality)
        elif os.path.isfile(path) and path.lower().endswith('.bmp'):
            convert_bmp_to_jpg(path, None, quality)
        else:
            print("请提供有效的BMP文件或包含BMP文件的目录")