# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.widgets import Button
import json
import os

# 配置中文字体显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti', 'SimHei', 'Heiti TC', 'LiHei Pro', 'Hei', 'Lantinghei SC', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 确保使用正确的后端
try:
    import tkinter
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except:
        pass

'''

1150 819  (36.6232, 11.1788)

857 1109(-3.3291, -39.1475)  


'''

def calculate_lat_lon(x, y, ref_points):
    """
    Calculate latitude and longitude from pixel coordinates using reference points.
    
    Args:
        x, y: pixel coordinates to convert
        ref_points: list of tuples [(x1, y1, lon1, lat1), (x2, y2, lon2, lat2)]
    
    Returns:
        tuple: (longitude, latitude)
    """
    # Reference points
    x1, y1, lon1, lat1 = ref_points[0]
    x2, y2, lon2, lat2 = ref_points[1]
    
    # Calculate the transformation using linear interpolation/extrapolation
    # Using vector-based approach
    
    # Direction vector in pixel space
    dx_pixel = x2 - x1
    dy_pixel = y2 - y1
    
    # Direction vector in geo space
    dlon_geo = lon2 - lon1
    dlat_geo = lat2 - lat1
    
    # Vector from reference point 1 to target point
    dx_target = x - x1
    dy_target = y - y1
    
    # Calculate parameter t for the linear transformation
    # We'll use a weighted approach based on both x and y distances
    pixel_distance_total = (dx_pixel**2 + dy_pixel**2)**0.5
    
    if pixel_distance_total == 0:
        # Points are the same, return first reference point coordinates
        return lon1, lat1
    
    # Project target vector onto reference direction vector
    dot_product = (dx_target * dx_pixel + dy_target * dy_pixel)
    t = dot_product / (pixel_distance_total**2)
    
    # Calculate longitude and latitude
    longitude = lon1 + t * dlon_geo
    latitude = lat1 + t * dlat_geo
    
    return longitude, latitude

def print_coordinate_calculation():
    """Calculate and print coordinates for the requested point."""
    # Reference points: (x, y, longitude, latitude)
    ref_points = [
        (1150, 819, 36.6232, 11.1788),
        (857, 1109, -3.3291, -39.1475)
    ]
    
    # Target point
    target_x, target_y = 1738, 1086
    
    # Calculate coordinates
    longitude, latitude = calculate_lat_lon(target_x, target_y, ref_points)
    
    print(f"\n=== 坐标计算结果 ===")
    print(f"像素坐标: ({target_x}, {target_y})")
    print(f"地理坐标: ({longitude:.4f}, {latitude:.4f})")
    print(f"经度: {longitude:.4f}°")
    print(f"纬度: {latitude:.4f}°")
    
    # Also print reference points for verification
    print(f"\n参考点验证:")
    for i, (x, y, lon, lat) in enumerate(ref_points, 1):
        calc_lon, calc_lat = calculate_lat_lon(x, y, ref_points)
        print(f"参考点 {i}: 像素({x}, {y}) -> 实际({lon}, {lat}) | 计算({calc_lon:.4f}, {calc_lat:.4f})")

class CoordinatePicker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.coordinates = []
        self.point_labels = ['左上角特征点', '左下角特征点', '右上角特征点', '右下角特征点']
        self.current_point_index = 0
        
        # 加载图像
        try:
            self.image = mpimg.imread(image_path)
            self.image_height, self.image_width = self.image.shape[:2]
            print(f"图像尺寸: {self.image_width} x {self.image_height}")
        except Exception as e:
            print(f"错误：无法加载图像 {image_path}")
            print(f"错误信息: {e}")
            return
            
        # 创建图形和轴
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title('世界地图坐标获取工具\n点击图像获取坐标 (建议按顺序点击: 左上→左下→右上→右下)', fontsize=14)
        
        # 连接鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # 添加控制按钮
        self.add_control_buttons()
        
        # 创建文本显示区域
        self.info_text = self.fig.text(0.02, 0.02, '', fontsize=10, 
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # 创建RGB显示区域
        self.rgb_text = self.fig.text(0.02, 0.92, '鼠标移动到图像上查看RGB值', fontsize=12, 
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        self.update_info_display()
        
    def on_mouse_move(self, event):
        """鼠标移动时显示当前像素的RGB值"""
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            
            # 确保坐标在图像范围内
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                # 获取RGB值
                if len(self.image.shape) == 3:  # 彩色图像
                    pixel = self.image[y, x]
                    
                    # 处理不同的图像格式
                    if self.image.shape[2] >= 3:  # RGB 或 RGBA
                        if self.image.dtype == np.float32 or self.image.dtype == np.float64:
                            # 如果是浮点数，转换为0-255范围
                            r, g, b = (pixel[:3] * 255).astype(int)
                        else:
                            r, g, b = pixel[:3].astype(int)
                        
                        # 如果是RGBA图像，也显示Alpha值
                        if self.image.shape[2] == 4:
                            if self.image.dtype == np.float32 or self.image.dtype == np.float64:
                                alpha = int(pixel[3] * 255)
                            else:
                                alpha = int(pixel[3])
                            rgb_info = f"像素坐标: ({x}, {y})\nRGBA值: R={r}, G={g}, B={b}, A={alpha}\n十六进制: #{r:02X}{g:02X}{b:02X}{alpha:02X}"
                        else:
                            rgb_info = f"像素坐标: ({x}, {y})\nRGB值: R={r}, G={g}, B={b}\n十六进制: #{r:02X}{g:02X}{b:02X}"
                    else:
                        # 处理其他格式
                        rgb_info = f"像素坐标: ({x}, {y})\n像素值: {pixel}"
                        
                else:  # 灰度图像
                    gray = self.image[y, x]
                    if self.image.dtype == np.float32 or self.image.dtype == np.float64:
                        gray = int(gray * 255)
                    rgb_info = f"像素坐标: ({x}, {y})\n灰度值: {gray}"
                
                self.rgb_text.set_text(rgb_info)
                self.fig.canvas.draw_idle()
            else:
                self.rgb_text.set_text("鼠标移动到图像上查看RGB值")
                self.fig.canvas.draw_idle()
        else:
            self.rgb_text.set_text("鼠标移动到图像上查看RGB值")
            self.fig.canvas.draw_idle()

    def add_control_buttons(self):
        # 清除按钮
        ax_clear = plt.axes([0.7, 0.01, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, '清除所有')
        self.btn_clear.on_clicked(self.clear_all)
        
        # 撤销按钮
        ax_undo = plt.axes([0.81, 0.01, 0.08, 0.04])
        self.btn_undo = Button(ax_undo, '撤销')
        self.btn_undo.on_clicked(self.undo_last)
        
        # 保存按钮
        ax_save = plt.axes([0.9, 0.01, 0.08, 0.04])
        self.btn_save = Button(ax_save, '保存')
        self.btn_save.on_clicked(self.save_coordinates)
        
    def on_click(self, event):
        if event.inaxes == self.ax and event.button == 1:  # 左键点击
            x, y = int(event.xdata), int(event.ydata)
            
            # 添加坐标到列表
            label = self.point_labels[self.current_point_index] if self.current_point_index < len(self.point_labels) else f'点{len(self.coordinates)+1}'
            self.coordinates.append({
                'label': label,
                'x': x, 
                'y': y,
                'index': len(self.coordinates)
            })
            
            # 在图上标记点
            color = ['red', 'green', 'blue', 'orange'][self.current_point_index % 4]
            self.ax.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
            self.ax.annotate(f'{len(self.coordinates)}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', color='white', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
            
            self.current_point_index = (self.current_point_index + 1) % len(self.point_labels)
            
            self.update_info_display()
            self.fig.canvas.draw()
            
            print(f"点击坐标: ({x}, {y}) - {label}")
            
    def clear_all(self, event):
        self.coordinates = []
        self.current_point_index = 0
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title('世界地图坐标获取工具\n点击图像获取坐标 (建议按顺序点击: 左上→左下→右上→右下)', fontsize=14)
        self.rgb_text.set_text('鼠标移动到图像上查看RGB值')
        self.update_info_display()
        self.fig.canvas.draw()
        print("已清除所有坐标")
        
    def undo_last(self, event):
        if self.coordinates:
            removed = self.coordinates.pop()
            self.current_point_index = (self.current_point_index - 1) % len(self.point_labels)
            
            # 重新绘制
            self.ax.clear()
            self.ax.imshow(self.image)
            self.ax.set_title('世界地图坐标获取工具\n点击图像获取坐标 (建议按顺序点击: 左上→左下→右上→右下)', fontsize=14)
            
            # 重新绘制剩余的点
            for i, coord in enumerate(self.coordinates):
                color = ['red', 'green', 'blue', 'orange'][i % 4]
                self.ax.plot(coord['x'], coord['y'], 'o', color=color, markersize=8, 
                           markeredgecolor='white', markeredgewidth=2)
                self.ax.annotate(f'{i+1}', (coord['x'], coord['y']), xytext=(5, 5), 
                               textcoords='offset points', color='white', fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color))
            
            self.update_info_display()
            self.fig.canvas.draw()
            print(f"已撤销: {removed['label']} ({removed['x']}, {removed['y']})")
            
    def update_info_display(self):
        if not self.coordinates:
            info = "点击说明:\n• 左键点击图像获取坐标\n• 建议按顺序获取四个角的特征点\n• 坐标将显示在这里"
        else:
            info = f"已获取 {len(self.coordinates)} 个坐标点:\n"
            for coord in self.coordinates:
                info += f"• {coord['label']}: ({coord['x']}, {coord['y']})\n"
            
            if len(self.coordinates) >= 4:
                info += "\n已获取四个角点，可以进行下一步处理！"
                
        self.info_text.set_text(info)
        
    def save_coordinates(self, event):
        if not self.coordinates:
            print("没有坐标可保存")
            return
            
        # 保存为JSON文件
        data = {
            'image_path': self.image_path,
            'image_size': {'width': self.image_width, 'height': self.image_height},
            'coordinates': self.coordinates,
            'total_points': len(self.coordinates)
        }
        
        output_file = 'world_map_coordinates.json'
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"坐标已保存到: {output_file}")
            
            # 同时打印到控制台
            print("\n=== 坐标总结 ===")
            print(f"图像尺寸: {self.image_width} x {self.image_height}")
            for coord in self.coordinates:
                print(f"{coord['label']}: ({coord['x']}, {coord['y']})")
                
        except Exception as e:
            print(f"保存失败: {e}")
    
    def show(self):
        plt.show()

def main():
    # 图像路径
    image_path = 'world_map.jpg'
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        return
    
    print("启动交互式坐标获取工具...")
    print("使用说明：")
    print("1. 鼠标移动到图像上可实时查看当前像素的RGB值")
    print("2. 左键点击图像上的特征点获取坐标")
    print("3. 建议按顺序点击：左上角 → 左下角 → 右上角 → 右下角")
    print("4. 点击'清除所有'按钮清除所有标记")
    print("5. 点击'撤销'按钮撤销最后一个点")
    print("6. 点击'保存'按钮将坐标保存到JSON文件")
    print("7. 关闭窗口结束程序")
    
    # 显示之前计算的坐标信息
    print("\n=== 之前计算的坐标信息 ===")
    print("像素坐标 (1738, 1086) 对应的地理坐标: (58.9219°, 39.2675°)")
    
    picker = CoordinatePicker(image_path)
    picker.show()

if __name__ == "__main__":
    main() 