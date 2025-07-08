from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue
import time
from loguru import logger

import signal


class ImageVisualizer:
    def __init__(self, window_name="RGB Images"):
        self.window_name = window_name
        self.image_queue = Queue(maxsize=10)  # 限制队列大小避免内存溢出
        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        # signal.signal(signal.SIGTERM, self._signal_handler)


    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.stop_visualization()


    def start_visualization(self):
        """启动可视化进程"""
        self.process = mp.Process(target=self._visualization_worker)
        self.process.start()
        logger.info("Image visualization process started")


    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """上下文管理器出口"""
    #     self.stop_visualization()
    
    # def __del__(self):
    #     """析构函数"""
    #     try:
    #         self.stop_visualization()
    #     except:
    #         pass


    def stop_visualization(self):
        """停止可视化进程"""
        self.running = False
        
        if hasattr(self, 'process') and self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)  # 等待最多2秒
                
                # # 如果进程还在运行，强制杀死
                # if self.process.is_alive():
                #     self.process.kill()
                #     self.process.join()
            import sys  
            sys.exit(0)                
        # try:
        #     cv2.destroyAllWindows()
        # except:
        #     pass      
        
          
    def update_images(self, rgb_images):
        """更新图像数据"""
        if not self.image_queue.full():
            try:
                # 将图像列表转换为可序列化的格式
                processed_images = []
                for img in rgb_images:
                    if img is not None:
                        # 确保图像是uint8格式
                        if img.dtype != np.uint8:
                            if img.max() <= 1.0:
                                img = (img * 255).astype(np.uint8)
                            else:
                                img = img.astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        processed_images.append(img.copy())
                    else:
                        processed_images.append(None)
                
                self.image_queue.put(processed_images, block=False)
            except Exception as e:
                logger.warning(f"Failed to put images in queue: {e}")
    
    def _visualization_worker(self):
        """可视化工作进程"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        while self.running:
            try:
                # 非阻塞方式获取图像
                rgb_images = self.image_queue.get(timeout=0.1)
                
                if rgb_images and len(rgb_images) > 0:
                    # 处理多张图像显示
                    if len(rgb_images) == 1:
                        # 单张图像
                        img = rgb_images[0]
                        if img is not None:
                            cv2.imshow(self.window_name, img)
                    else:
                        # 多张图像拼接显示
                        valid_images = [img for img in rgb_images if img is not None]
                        if valid_images:
                            # 水平拼接图像
                            if len(valid_images) == 2:
                                combined_img = np.hstack(valid_images)
                            else:
                                # 多张图像网格显示
                                rows = int(np.ceil(np.sqrt(len(valid_images))))
                                cols = int(np.ceil(len(valid_images) / rows))
                                
                                # 创建网格
                                max_h = max(img.shape[0] for img in valid_images)
                                max_w = max(img.shape[1] for img in valid_images)
                                
                                grid_img = np.zeros((max_h * rows, max_w * cols, 3), dtype=np.uint8)
                                
                                for idx, img in enumerate(valid_images):
                                    row = idx // cols
                                    col = idx % cols
                                    h, w = img.shape[:2]
                                    grid_img[row*max_h:row*max_h+h, col*max_w:col*max_w+w] = img
                                
                                combined_img = grid_img
                            
                            cv2.imshow(self.window_name, combined_img)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    self.running = False
                    break
                    
            except Exception as e:
                # 队列超时或其他错误，继续循环
                continue
                
        cv2.destroyAllWindows()



def adjust_translation_along_quaternion(translation, quaternion, distance):
    """
    quaternion: [w, x, y, z]
    """
    rotation = R.from_quat(quaternion[[1, 2, 3, 0]])
    direction_vector = rotation.apply([0, 0, 1])
    reverse_direction = -direction_vector
    new_translation = translation + reverse_direction * distance
    return new_translation


def trans_quant_to_RT(translation, quaternion):
    """
    translation: [x, y, z]
    quaternion: [w, x, y, z]
    """
    RT = np.eye(4)
    RT[:3, 3] = translation
    RT[:3, :3] = R.from_quat(quaternion, scalar_first=True).as_matrix()
    return RT

def RT_to_tran_quaternion(RT):
    """
    return translation, quaternion [w, x, y, z]
    """
    return RT[:3, 3], R.from_matrix(RT[:3, :3]).as_quat(scalar_first=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
