# -*- coding: utf-8 -*-
"""
RGB到EC值转换器
根据特定的RGB规则计算EC值
"""

def rgb_to_ec(r, g, b):
    """
    根据RGB值计算EC值
    
    规则：
    1. 当B不为零时，EC为-1
    2. B为零时：
       - 当R从0-255变化、G恒为255时，EC值均匀从0-4变化
       - 当R恒为255、G从255-0变化时，EC值均匀从4-12变化
       - 当R从255-0变化、G恒为0时，EC从12-30均匀变化
    
    参数：
        r (int): 红色值 (0-255)
        g (int): 绿色值 (0-255) 
        b (int): 蓝色值 (0-255)
    
    返回：
        float: 计算得到的EC值
    """
    # 确保输入值在有效范围内
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    
    # 规则1：当B不为零时，EC为-1
    if b >=200 :
        return -1.0
    
    # B为零的情况
    if b <200:
        # 情况1：R从0-255变化、G恒为255时，EC值均匀从0-4变化
        if g >= 245:
            ec = r * 4.0 / 255.0
            return round(ec, 6)  # 保留6位小数精度
        
        # 情况2：R恒为255、G从255-0变化时，EC值均匀从4-12变化
        elif r>= 245:
            ec = 4.0 + (255 - g) * 8.0 / 255.0
            return round(ec, 6)
        
        # 情况3：R从255-0变化、G恒为0时，EC从12-30均匀变化
        elif g <= 10:
            ec = 12.0 + (255 - r) * 18.0 / 255.0
            return round(ec, 6)
        
        # 如果不符合上述任何情况，返回None或抛出异常
        elif r <=10:
            ec=30+g/3
            return round(ec, 6) 
        else :
            return -1.0

def ec_to_rgb_range1(ec):
    """
    反向计算：从EC值计算RGB值 (范围1: EC 0-4, G=255)
    """
    if not (0 <= ec <= 4):
        return None
    r = int(ec * 255.0 / 4.0)
    return (r, 255, 0)

def ec_to_rgb_range2(ec):
    """
    反向计算：从EC值计算RGB值 (范围2: EC 4-12, R=255)
    """
    if not (4 <= ec <= 12):
        return None
    g = int(255 - (ec - 4) * 255.0 / 8.0)
    return (255, g, 0)

def ec_to_rgb_range3(ec):
    """
    反向计算：从EC值计算RGB值 (范围3: EC 12-30, G=0)
    """
    if not (12 <= ec <= 30):
        return None
    r = int(255 - (ec - 12) * 255.0 / 18.0)
    return (r, 0, 0)

def ec_to_rgb(ec):
    """
    从EC值反向计算RGB值
    """
    if ec == -1:
        return "B值不为零的任意RGB组合"
    elif 0 <= ec <= 4:
        return ec_to_rgb_range1(ec)
    elif 4 < ec <= 12:
        return ec_to_rgb_range2(ec)
    elif 12 < ec <= 30:
        return ec_to_rgb_range3(ec)
    else:
        return None

def test_rgb_to_ec():
    """
    测试函数，验证RGB到EC转换的正确性
    """
    print("=== RGB到EC转换测试 ===")
    
    # 测试用例
    test_cases = [
        # (R, G, B), 期望的EC值范围或特定值
        (0, 255, 0, "EC应该为0"),      # R=0, G=255, B=0 -> EC=0
        (255, 255, 0, "EC应该为4"),    # R=255, G=255, B=0 -> EC=4
        (255, 0, 0, "EC应该为12"),     # R=255, G=0, B=0 -> EC=12
        (0, 0, 0, "EC应该为30"),       # R=0, G=0, B=0 -> EC=30
        (128, 255, 0, "EC应该约为2"),  # R=128, G=255, B=0 -> EC≈2
        (255, 128, 0, "EC应该约为8"),  # R=255, G=128, B=0 -> EC≈8
        (128, 0, 0, "EC应该约为21"),   # R=128, G=0, B=0 -> EC≈21
        (100, 100, 50, "EC应该为-1"),  # B≠0 -> EC=-1
        (100, 100, 1, "EC应该为-1"),   # B≠0 -> EC=-1
    ]
    
    for r, g, b, expected in test_cases:
        ec = rgb_to_ec(r, g, b)
        print(f"RGB({r}, {g}, {b}) -> EC = {ec} ({expected})")
    
    print("\n=== EC范围测试 ===")
    
    # 测试范围1: R 0->255, G=255, B=0
    print("范围1 (R变化, G=255, B=0):")
    for r in [0, 64, 128, 192, 255]:
        ec = rgb_to_ec(r, 255, 0)
        print(f"  RGB({r}, 255, 0) -> EC = {ec}")
    
    # 测试范围2: R=255, G 255->0, B=0  
    print("\n范围2 (R=255, G变化, B=0):")
    for g in [255, 192, 128, 64, 0]:
        ec = rgb_to_ec(255, g, 0)
        print(f"  RGB(255, {g}, 0) -> EC = {ec}")
    
    # 测试范围3: R 255->0, G=0, B=0
    print("\n范围3 (R变化, G=0, B=0):")
    for r in [255, 192, 128, 64, 0]:
        ec = rgb_to_ec(r, 0, 0)
        print(f"  RGB({r}, 0, 0) -> EC = {ec}")

def interactive_rgb_to_ec():
    """
    交互式RGB到EC转换工具
    """
    print("=== 交互式RGB到EC转换工具 ===")
    print("输入RGB值来计算对应的EC值")
    print("输入 'quit' 退出程序")
    
    while True:
        try:
            user_input = input("\n请输入RGB值 (格式: R G B): ").strip()
            
            if user_input.lower() == 'quit':
                print("程序退出")
                break
                
            # 解析输入
            rgb_values = user_input.split()
            if len(rgb_values) != 3:
                print("请输入3个数值 (R G B)")
                continue
                
            r, g, b = map(int, rgb_values)
            
            # 计算EC值
            ec = rgb_to_ec(r, g, b)
            
            if ec is None:
                print(f"RGB({r}, {g}, {b}) -> 无效的RGB组合，不符合任何EC计算规则")
            else:
                print(f"RGB({r}, {g}, {b}) -> EC = {ec}")
                
                # 如果可能，显示反向计算结果
                reverse_rgb = ec_to_rgb(ec)
                if reverse_rgb and reverse_rgb != "B值不为零的任意RGB组合":
                    print(f"  验证: EC({ec}) -> RGB{reverse_rgb}")
                
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n程序退出")
            break

if __name__ == "__main__":
    # 运行测试
    test_rgb_to_ec()
    
    # 运行交互式工具
    print("\n" + "="*50)
    interactive_rgb_to_ec() 