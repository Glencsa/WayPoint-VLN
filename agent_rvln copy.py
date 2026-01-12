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

# 添加项目根目录到 Python 路径
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_path)
sys.path.insert(0, project_root)

from transformers import InstructBlipProcessor
from models.rvln import RvlnMultiTask
from utils.utils import prepare_inputs_for_generate


class RVLN_Agent(Agent):
    def __init__(self, model_path, result_path, require_map=True):
        
        print("Initialize RVLN Agent")
        
        self.result_path = result_path
        self.require_map = require_map
        self.model_path = model_path
        
        # 创建必要的目录结构
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
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
            
        except Exception as e:
            print(f"[ERROR] Failed to load RVLN model: {e}")
            print("Please check:")
            print("  1. Model path exists and contains all required files")
            print("  2. config.json is valid JSON format")
            print("  3. Vision encoder config is properly set")
            raise
        
        print("RVLN Agent Initialization Complete")
        
        # Prompt 模板
        self.prompt_template = "You are a navigation agent. Given the visual observations (RGB and depth images) and the instruction: '{}', predict the best route to follow. Output the route number from -1 to 8, where -1 means stop."
        
        # 历史观测队列
        self.history_rgb_list = []
        self.history_depth_list = []
        self.max_history = 5  # 最多保存5帧历史
        
        # 可视化相关
        self.rgb_list = []
        self.depth_list = []
        self.topdown_map_list = []
        
        self.count_id = 0
        self.reset()
    
    def reset(self):
        """重置 agent 状态"""
        self.history_rgb_list = []
        self.history_depth_list = []
        self.rgb_list = []
        self.depth_list = []
        self.topdown_map_list = []
        self.count_id = 0
        print("[RVLN Agent] State reset")
    
    def process_observations(self, rgb, depth):
        """
        处理并添加新的观测到历史队列
        
        Args:
            rgb: RGB observation (numpy array or PIL Image)
            depth: Depth observation (numpy array or PIL Image)
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
        
        # Depth 处理：归一化到 0-255
        if depth.dtype != np.uint8:
            depth_float = depth.astype(np.float32)
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
            
            depth = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
        
        # 添加到历史队列
        self.history_rgb_list.append(rgb)
        self.history_depth_list.append(depth)
        
        # 保持历史长度限制
        if len(self.history_rgb_list) > self.max_history:
            self.history_rgb_list = self.history_rgb_list[-self.max_history:]
            self.history_depth_list = self.history_depth_list[-self.max_history:]
        
        # 保存用于可视化
        self.rgb_list.append(rgb)
        self.depth_list.append(depth)
        
        print(f"[RVLN Agent] Observation processed. History length: {len(self.history_rgb_list)}")
    
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
            raise RuntimeError("[RVLN Agent] No observations available. Call process_observations() first.")
        
        print(f"\n[RVLN Agent] Predicting route...")
        print(f"  Instruction: {instruction}")
        print(f"  History frames: {len(self.history_rgb_list)}")
        
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
        print(f"  Model output: {output_text}")
        
        # 解析路线编号
        route_number = self._parse_route_number(output_text)
        print(f"  Predicted route: {route_number}")
        
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
        print(f"[RVLN Agent] Warning: Could not parse route number from: {output_text}")
        return -1
    
    def act(self, observations):
        """
        Habitat Agent 接口：根据观测返回动作
        这个方法需要根据实际使用场景实现
        """
        # 提取 RGB 和 Depth
        rgb = observations.get("rgb", observations.get("rgb_sensor"))
        depth = observations.get("depth", observations.get("depth_sensor"))
        
        if rgb is None or depth is None:
            raise ValueError("[RVLN Agent] Missing RGB or Depth in observations")
        
        # 处理观测
        self.process_observations(rgb, depth)
        
        # 这里需要从 episode 中获取指令
        # 实际使用时需要传入指令或从环境中提取
        instruction = "Navigate to the target location"  # 占位符
        
        # 预测路线
        route_number, output_text = self.predict_route(instruction)
        
        # 将路线编号映射为 Habitat 动作
        # 这里需要根据实际的动作空间定义来映射
        action = self._map_route_to_action(route_number)
        
        return action
    
    def _map_route_to_action(self, route_number):
        """
        将路线编号映射为 Habitat 动作
        需要根据实际的动作空间定义来实现
        
        示例映射:
        -1: STOP (0)
        0: TURN_LEFT_30 (2)
        1-7: 不同的路线，这里简化为 MOVE_FORWARD (1)
        8: TURN_RIGHT_30 (3)
        """
        if route_number == -1:
            return 0  # STOP
        elif route_number == 0:
            return 2  # TURN_LEFT
        elif route_number == 8:
            return 3  # TURN_RIGHT
        else:
            return 1  # MOVE_FORWARD
    
    def save_trajectory(self, episode_id):
        """保存轨迹可视化"""
        if len(self.rgb_list) == 0:
            print("[RVLN Agent] No trajectory to save")
            return
        
        video_path = os.path.join(self.result_path, "video", f"episode_{episode_id}.mp4")
        
        try:
            writer = imageio.get_writer(video_path, fps=5)
            
            for i, (rgb, depth) in enumerate(zip(self.rgb_list, self.depth_list)):
                # 创建并排显示 RGB 和 Depth
                if depth.ndim == 2:
                    depth_vis = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
                else:
                    depth_vis = depth
                
                # 调整大小使其匹配
                if rgb.shape[:2] != depth_vis.shape[:2]:
                    depth_vis = cv2.resize(depth_vis, (rgb.shape[1], rgb.shape[0]))
                
                # 水平拼接
                frame = np.hstack([rgb, depth_vis])
                
                # 添加步数标注
                cv2.putText(frame, f"Step {i}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                writer.append_data(frame)
            
            writer.close()
            print(f"[RVLN Agent] Trajectory saved to: {video_path}")
            
        except Exception as e:
            print(f"[RVLN Agent] Failed to save trajectory: {e}")
    
    def save_log(self, episode_id, data):
        """保存日志数据"""
        log_path = os.path.join(self.result_path, "log", f"episode_{episode_id}.json")
        
        try:
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[RVLN Agent] Log saved to: {log_path}")
        except Exception as e:
            print(f"[RVLN Agent] Failed to save log: {e}")