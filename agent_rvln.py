import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
import torch
import cv2
import imageio
from habitat.utils.visualizations import maps
import random
import sys
from PIL import Image

# 添加项目根目录到 Python 路径
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_path)
sys.path.insert(0, project_root)

from transformers import InstructBlipProcessor
from models.rvln import RvlnMultiTask
from utils.utils import prepare_inputs_for_generate


class RVLN_Agent(Agent):
    def __init__(self, model_path, result_path, exp_save):
        
        print("Initialize RVLN Agent")
        
        self.result_path = result_path
        self.require_map = True if "video" in exp_save else False
        self.require_data = True if "data" in exp_save else False
        self.model_path = model_path
        
        # 创建必要的目录结构（与 NaVid 完全一致）
        if self.require_map or self.require_data:
            os.makedirs(self.result_path, exist_ok=True)
        
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        
        if self.require_map:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        
        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        
        # 加载模型
        print(f"Loading RVLN model from: {self.model_path}")
        try:
            self.processor = InstructBlipProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.processor.tokenizer
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = RvlnMultiTask.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            ).to(self.device)
            self.model.eval()
            
            print("RVLN Agent Initialization Complete")
            
        except Exception as e:
            print(f"[ERROR] Failed to load RVLN model: {e}")
            print("Please check:")
            print("  1. Model path exists and contains all required files")
            print("  2. config.json is valid JSON format")
            print("  3. Vision encoder config is properly set")
            raise
        
        # Prompt 模板（与路线预测任务匹配）
        self.prompt_template = "Given the navigation instruction: '{}', analyze the visual observations (RGB and depth images) to predict the best route. Output in format: {{'Route': N}} where N is from -1 to 8."
        
        # 历史观测队列（保存 numpy arrays）
        self.history_rgb_list = []
        self.history_depth_list = []
        self.max_history = 5  # 最多保存5帧历史
        
        # 可视化相关（与 NaVid 完全一致）
        self.rgb_list = []
        self.depth_list = []
        self.topdown_map_list = []
        
        # 状态管理
        self.count_id = 0
        self.pending_action_list = []
        self.episode_id = None
        
        self.reset()
    
    def reset(self):
        """重置 agent 状态（与 NaVid 完全一致的逻辑）"""
        
        # 保存上一个 episode 的视频
        if self.require_map:
            if len(self.topdown_map_list) != 0:
                output_video_path = os.path.join(self.result_path, "video", "{}.gif".format(self.episode_id))
                imageio.mimsave(output_video_path, self.topdown_map_list)
        
        # 重置所有状态
        self.history_rgb_list = []
        self.history_depth_list = []
        self.rgb_list = []
        self.depth_list = []
        self.topdown_map_list = []
        self.pending_action_list = []
        self.count_id += 1
    
    def process_observations(self, rgb, depth):
        """
        处理并添加新的观测到历史队列
        
        Args:
            rgb: RGB observation (numpy array)
            depth: Depth observation (numpy array)
        """
        # 转换为 numpy array
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb)
        if not isinstance(depth, np.ndarray):
            depth = np.array(depth)
        
        # RGB 处理：确保是 uint8 格式
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb = rgb.clip(0, 255).astype(np.uint8)
        
        # Depth 处理：归一化到 0-255（用于可视化）
        depth_vis = depth.copy()
        if depth_vis.dtype != np.uint8:
            depth_float = depth_vis.astype(np.float32)
            depth_float = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0)
            
            if depth_float.ndim == 3:
                if depth_float.shape[2] == 1:
                    depth_float = depth_float[..., 0]
                else:
                    depth_float = depth_float.mean(axis=2)
            
            d_min, d_max = float(depth_float.min()), float(depth_float.max())
            if d_max > d_min:
                depth_norm = (depth_float - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth_float)
            
            depth_vis = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        
        # 添加到历史队列
        self.history_rgb_list.append(rgb)
        self.history_depth_list.append(depth)  # 保存原始 depth，模型内部会处理
        
        # 保存用于可视化
        self.rgb_list.append(rgb)
        self.depth_list.append(depth_vis)
        
        # 保持历史长度限制
        if len(self.history_rgb_list) > self.max_history:
            self.history_rgb_list = self.history_rgb_list[-self.max_history:]
            self.history_depth_list = self.history_depth_list[-self.max_history:]
    
    def predict_route(self, instruction, max_new_tokens=100):
        """
        基于当前历史观测预测路线
        
        Args:
            instruction: 导航指令文本
            max_new_tokens: 生成的最大 token 数
            
        Returns:
            route_number: 预测的路线编号 (-1 to 8)
            output_text: 模型输出的完整文本
        """
        if len(self.history_rgb_list) == 0:
            raise RuntimeError("[RVLN Agent] No observations available.")
        
        # 准备输入（自动补齐到5帧）
        inputs = prepare_inputs_for_generate(
            self.history_rgb_list,
            self.history_depth_list,
            instruction,
            self.processor,
            self.device
        )
        
        # 生成预测
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=inputs["pixel_values"],
                depth_pixel_values=inputs["depth_pixel_values"],
                qformer_input_ids=inputs["qformer_input_ids"],
                qformer_attention_mask=inputs["qformer_attention_mask"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0
            )
        
        # 解码输出
        output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        # 解析路线编号
        route_number = self._parse_route_number(output_text)
        
        return route_number, output_text
    
    def _parse_route_number(self, output_text):
        """
        从模型输出文本中解析路线编号
        
        支持的格式:
        - {'Route': 3}
        - "Route": 3
        - Route: 3
        - 纯数字: 3
        """
        # 尝试匹配 Route: N 格式
        match = re.search(r"['\"]?Route['\"]?\s*[:=]\s*(-?\d+)", output_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # 尝试匹配纯数字
        match = re.search(r"(-?\d+)", output_text)
        if match:
            return int(match.group(1))
        
        # 默认返回 -1 (停止)
        return -1
    
    def route_to_actions(self, route_number):
        """
        将路线编号转换为动作序列
        
        路线编号映射:
        -1: stop
        0: turn left 30
        1-7: forward (路线越大，前进距离可能越远)
        8: turn right 30
        
        Args:
            route_number: 路线编号 (-1 to 8)
            
        Returns:
            action_list: 动作序列 [action_id, ...]
        """
        action_list = []
        
        if route_number == -1:
            # 停止
            action_list.append(0)
        elif route_number == 0:
            # 左转 30 度（执行1次左转动作）
            action_list.append(2)
        elif route_number == 8:
            # 右转 30 度（执行1次右转动作）
            action_list.append(3)
        elif 1 <= route_number <= 7:
            # 前进，路线编号越大可能前进越远
            # 这里简化为：路线 1-3 前进1次，4-5 前进2次，6-7 前进3次
            if route_number <= 3:
                forward_times = 1
            elif route_number <= 5:
                forward_times = 2
            else:
                forward_times = 3
            
            for _ in range(forward_times):
                action_list.append(1)
        else:
            # 未知路线，随机动作
            action_list.append(random.randint(1, 3))
        
        return action_list
    
    def addtext(self, image, instruction, navigation):
        """在图像上添加文本（与 NaVid 完全一致）"""
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.zeros((new_height, w, 3), np.uint8)
        new_image.fill(255)  
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize(instruction, font, 0.5, 2)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY + 0 * textsize[1]

        words = instruction.split(' ')
        x = 10
        line = ""

        for word in words:
            test_line = line + ' ' + word if line else word
            test_line_size, _ = cv2.getTextSize(test_line, font, 0.5, 2)

            if test_line_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, 0.5, (0, 0, 0), 2)

        y_line = y_line + 1 * textsize[1] + 10
        new_image = cv2.putText(new_image, navigation, (x, y_line), font, 0.5, (0, 0, 0), 2)

        return new_image
    
    def act(self, observations, info, episode_id):
        """
        主要的动作选择接口（与 NaVid 完全一致的接口）
        
        Args:
            observations: Habitat 观测字典，包含 rgb, depth, instruction 等
            info: 环境信息字典，包含 top_down_map_vlnce 等
            episode_id: 当前 episode 的 ID
            
        Returns:
            {"action": action_id}
        """
        self.episode_id = episode_id
        
        # 提取 RGB 和 Depth
        rgb = observations["rgb"]
        depth = observations.get("depth", observations.get("depth_sensor"))
        
        if depth is None:
            raise ValueError("[RVLN Agent] Missing depth in observations")
        
        # 处理观测
        self.process_observations(rgb, depth)
        
        # 生成可视化图像（与 NaVid 一致）
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
            # 拼接 RGB 和 top-down map
            output_im = np.concatenate((rgb, top_down_map), axis=1)
        
        # 如果有待执行的动作，优先执行（与 NaVid 完全一致）
        if len(self.pending_action_list) != 0:
            temp_action = self.pending_action_list.pop(0)
            
            if self.require_map:
                img = self.addtext(output_im, observations["instruction"]["text"], 
                                 f"Pending action: {temp_action}")
                self.topdown_map_list.append(img)
            
            return {"action": temp_action}
        
        # 使用 RVLN 模型预测路线
        instruction = observations["instruction"]["text"]
        route_number, output_text = self.predict_route(instruction)
        
        # 生成可视化文本
        navigation_text = f"Route: {route_number} | {output_text}"
        
        if self.require_map:
            img = self.addtext(output_im, instruction, navigation_text)
            self.topdown_map_list.append(img)
        
        # 将路线转换为动作序列
        action_list = self.route_to_actions(route_number)
        
        # 如果没有有效动作，使用随机动作
        if len(action_list) == 0:
            action_list.append(random.randint(1, 3))
        
        # 将动作加入待执行队列
        self.pending_action_list.extend(action_list)
        
        # 返回第一个动作
        return {"action": self.pending_action_list.pop(0)}