#!/usr/bin/env python3

# 绝对路径配置（请按需修改为你的实际绝对路径）
BASE_DIR = "/home/yang/VLN/navid_ws/NaVid-VLN-CE"
EXP_CONFIG = f"{BASE_DIR}/VLN_CE/vlnce_baselines/config/r2r_baselines/cma.yaml"  # 示例，占位
RESULT_PATH = f"{BASE_DIR}/results"
EPISODE_ID = "6296"  # 优先使用 episode 过滤，若不想用则置为 None
SCENE_ID = None       # 若 EPISODE_ID 为 None 时按 scene 过滤
ACTION = 2           # 单步动作，默认 MOVE_FORWARD=1；可改为 0:STOP, 2:TURN_LEFT, 3:TURN_RIGHT

# RVLN 模型配置
CHECKPOINT_PATH = "output/rvln_merged_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

import os
import json
import numpy as np
import torch
from typing import Optional
from PIL import Image

from habitat.datasets import make_dataset
from habitat import Env
from VLN_CE.vlnce_baselines.config.default import get_config

# 导入 RVLN 相关模块
import sys
current_file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_file_dir)

from transformers import InstructBlipProcessor
from models.rvln import RvlnMultiTask
from utils.utils import prepare_inputs_for_generate

# 确保结果根目录存在（启动即创建）
os.makedirs(RESULT_PATH, exist_ok=True)
print(f"[INFO] Result root ensured: {os.path.abspath(RESULT_PATH)}")


class RVLNAgent:
    """RVLN 模型代理，负责加载模型并进行路线推断"""
    
    def __init__(self, checkpoint_path, device="cuda", dtype=torch.float16):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        
        # 历史观测队列（最多保存4帧历史 + 1帧当前 = 5帧）
        self.rgb_history = []
        self.depth_history = []
        
    def load_model(self):
        """加载 RVLN 模型和处理器"""
        print(f"[AGENT] Loading RVLN model from: {self.checkpoint_path}")
        
        # 加载 Processor
        self.processor = InstructBlipProcessor.from_pretrained(self.checkpoint_path)
        tokenizer = self.processor.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 加载模型
        self.model = RvlnMultiTask.from_pretrained(
            self.checkpoint_path,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        
        print(f"[AGENT] Model loaded successfully on {self.device}")
        
    def reset_history(self):
        """重置历史观测队列"""
        self.rgb_history = []
        self.depth_history = []
        print("[AGENT] History queue reset")
        
    def add_observation(self, rgb_image, depth_image):
        """
        添加新的观测到历史队列
        rgb_image: PIL Image 或 numpy array
        depth_image: PIL Image 或 numpy array
        """
        # 转换为 PIL Image
        if isinstance(rgb_image, np.ndarray):
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.clip(0, 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb_image[..., :3] if rgb_image.ndim == 3 else rgb_image)
            
        if isinstance(depth_image, np.ndarray):
            # 深度图归一化可视化
            d = depth_image.astype(np.float32)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            if d.ndim == 3 and d.shape[2] == 1:
                d = d[..., 0]
            elif d.ndim == 3:
                d = d.mean(axis=2)
            d_min, d_max = float(d.min()), float(d.max())
            if d_max > d_min:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
            d_vis = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
            depth_image = Image.fromarray(d_vis)
            
        self.rgb_history.append(rgb_image)
        self.depth_history.append(depth_image)
        
    def predict_route(self, instruction):
        """
        基于当前历史队列预测最佳路线
        返回: 预测的路线编号（-1 到 8）
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("[AGENT] Model not loaded. Call load_model() first.")
            
        if len(self.rgb_history) == 0:
            raise RuntimeError("[AGENT] No observations in history. Call add_observation() first.")
            
        print(f"[AGENT] Predicting route with {len(self.rgb_history)} observations...")
        
        # 准备输入（自动补齐到5帧）
        inputs = prepare_inputs_for_generate(
            self.rgb_history,
            self.depth_history,
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
                max_new_tokens=100,
                do_sample=False,
                repetition_penalty=1.0
            )
        
        # 解码输出
        output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"[AGENT] Model output: {output_text}")
        
        # 解析路线编号（从输出文本中提取）
        route_number = self._parse_route_number(output_text)
        print(f"[AGENT] Predicted route: {route_number}")
        
        return route_number, output_text
        
    def _parse_route_number(self, output_text):
        """从模型输出文本中解析路线编号"""
        import re
        
        # 尝试匹配 {'Route': N} 格式
        match = re.search(r"['\"]?Route['\"]?\s*:\s*(-?\d+)", output_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
        # 尝试匹配纯数字
        match = re.search(r"(-?\d+)", output_text)
        if match:
            return int(match.group(1))
            
        # 默认返回 -1（停止）
        print(f"[AGENT] Warning: Could not parse route number from: {output_text}")
        return -1


def _get_instruction_text(ep) -> Optional[str]:
    candidates = [
        getattr(ep, "instruction", None),
        getattr(ep, "goal", None),
        getattr(ep, "instruction_text", None),
    ]
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, dict):
            for key in ("text", "instruction", "command"):
                if key in c and isinstance(c[key], str) and c[key].strip():
                    return c[key]
    return None


def step(env, action):
    """执行一步动作并更新状态。
    参数:
      env: Habitat 环境实例
      action: 整数或字典动作；若为整数，将封装为 {"action": action}
    返回:
      obs: 新的观测字典
      info: 当前指标（env.get_metrics()）
      done: 是否结束（env.episode_over）
      episode_id: 当前 episode id
    """
    # 统一为 habitat 的动作字典格式
    act = action if isinstance(action, dict) else {"action": int(action)}
    obs = env.step(act)
    info = env.get_metrics()
    done = env.episode_over
    episode_id = env.current_episode.episode_id
    return obs, info, done, episode_id


def save_observation(env, obs, result_path, step_idx=0):
    """将当前观测保存为 npy 与可视化 PNG，并记录元信息。
    参数:
      env: Habitat 环境实例（用于获取 episode/scene 信息）
      obs: 当前观测字典（包含 rgb/depth 等）
      result_path: 输出根目录
      step_idx: 当前保存步骤索引（用于创建形如 scene_id_step0 的子目录）
    返回:
      out_meta: 保存的元信息字典
    """
    # 提取并标准化通道
    rgb = obs.get("rgb", None)
    if rgb is None:
        rgb = obs.get("rgb_sensor", None)
    depth = obs.get("depth", None)
    if depth is None:
        depth = obs.get("depth_sensor", None)
    if rgb is not None:
        rgb = np.asarray(rgb)
    if depth is not None:
        depth = np.asarray(depth)
    

    # 以 results/<scene_name>/{image,depth}/step_{n}.(png,npy) 组织
    raw_scene_id = getattr(env.current_episode, "scene_id", "unknown_scene")
    scene_id_str = str(raw_scene_id)
    scene_basename = os.path.basename(scene_id_str)
    scene_name = os.path.splitext(scene_basename)[0] if ("." in scene_basename) else scene_basename

    scene_root = os.path.join(result_path, scene_name)
    image_dir = os.path.join(scene_root, "image")
    depth_dir = os.path.join(scene_root, "depth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"[INFO] Scene root: {os.path.abspath(scene_root)}")

    # 文件路径（按 step_n 命名）
    rgb_npy_path = os.path.join(image_dir, f"step_{int(step_idx)}.npy")
    rgb_png_path = os.path.join(image_dir, f"step_{int(step_idx)}.png")
    depth_npy_path = os.path.join(depth_dir, f"step_{int(step_idx)}.npy")
    depth_png_path = os.path.join(depth_dir, f"step_{int(step_idx)}.png")

    # 同步一个 meta（覆盖写，记录最新 step 信息；如需所有步的 meta，可改为 step_n.json）
    meta_path = os.path.join(scene_root, "meta.json")

    out_meta = {
        "episode_id": env.current_episode.episode_id,
        "scene_id": scene_id_str,
        "step_idx": int(step_idx),
        "instruction": _get_instruction_text(env.current_episode),
        "rgb_shape": None,
        "depth_shape": None,
        "paths": {
            "rgb_npy": rgb_npy_path,
            "rgb_png": rgb_png_path,
            "depth_npy": depth_npy_path,
            "depth_png": depth_png_path,
        },
    }

    # 保存 NPY
    if rgb is not None:
        try:
            np.save(rgb_npy_path, rgb)
            out_meta["rgb_shape"] = list(rgb.shape)
        except Exception as e:
            out_meta["rgb_error"] = str(e)
    if depth is not None:
        try:
            np.save(depth_npy_path, depth)
            out_meta["depth_shape"] = list(depth.shape)
        except Exception as e:
            out_meta["depth_error"] = str(e)

    # 保存 PNG
    try:
        from PIL import Image
        if rgb is not None:
            arr = np.asarray(rgb)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            if arr.ndim == 3:
                Image.fromarray(arr[..., :3]).save(rgb_png_path)
            elif arr.ndim == 2:
                Image.fromarray(arr).save(rgb_png_path)
        if depth is not None:
            d = np.asarray(depth).astype(np.float32)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            if d.ndim == 3:
                if d.shape[2] == 1:
                    d = d[..., 0]
                else:
                    d = d.mean(axis=2)
            d_min, d_max = float(d.min()), float(d.max())
            if d_max > d_min:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
            d_vis = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(d_vis).save(depth_png_path)
    except Exception as e:
        out_meta["png_error"] = str(e)

    with open(meta_path, "w") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    print(
        "[INFO] Saved step {} to:\n  - {}\n  - {}\n  - {}\n  - {}".format(
            int(step_idx), rgb_npy_path, rgb_png_path, depth_npy_path, depth_png_path
        )
    )
    return out_meta


class InteractiveSession:
    """支持增量式操作的交互会话：初始化、保存当前观测、按动作步进并保存。"""
    def __init__(self, base_dir, exp_config, result_path, episode_id=None, scene_id=None, use_agent=False):
        self.base_dir = base_dir
        self.exp_config = exp_config
        self.result_path = result_path
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.env = None
        self.obs = None
        self.step_idx = 0
        
        # RVLN Agent 相关
        self.use_agent = use_agent
        self.agent = None
        if self.use_agent:
            self.agent = RVLNAgent(CHECKPOINT_PATH, DEVICE, DTYPE)

    def init(self):
        # 切到 VLN_CE 目录，保证相对路径可用
        vlnce_dir = os.path.join(self.base_dir, "VLN_CE")
        os.chdir(vlnce_dir)
        config = get_config(self.exp_config, opts=None)
        dataset = make_dataset(
            id_dataset=config.TASK_CONFIG.DATASET.TYPE,
            config=config.TASK_CONFIG.DATASET,
        )
        dataset.episodes.sort(key=lambda ep: ep.episode_id)
        filtered = dataset.episodes
        if self.episode_id:
            filtered = [ep for ep in filtered if str(ep.episode_id) == str(self.episode_id)]
        elif self.scene_id:
            filtered = [ep for ep in filtered if str(getattr(ep, "scene_id", "")) == str(self.scene_id)]
        if not filtered:
            raise ValueError("No episode found for given filters (episode_id/scene_id).")
        dataset.episodes = [filtered[0]]
        np.random.seed(42)
        self.env = Env(config.TASK_CONFIG, dataset)
        self.obs = self.env.reset()
        self.step_idx = 0
        
        # 初始化时也确保结果根目录存在
        os.makedirs(self.result_path, exist_ok=True)
        print(f"[INFO] Result root ensured in init: {os.path.abspath(self.result_path)}")
        
        # 如果使用 agent，加载模型并重置历史
        if self.use_agent and self.agent is not None:
            self.agent.load_model()
            self.agent.reset_history()
            
            # 添加初始观测到 agent
            rgb = self.obs.get("rgb", self.obs.get("rgb_sensor", None))
            depth = self.obs.get("depth", self.obs.get("depth_sensor", None))
            if rgb is not None and depth is not None:
                self.agent.add_observation(rgb, depth)
                print("[INFO] Initial observation added to agent")
        
        return self.obs

    def save_current(self):
        if self.env is None or self.obs is None:
            raise RuntimeError("Session not initialized. Call init() first.")
        meta = save_observation(self.env, self.obs, self.result_path, step_idx=self.step_idx)
        print(f"[INFO] Initial observation saved for step {self.step_idx}.")
        return meta

    def predict_next_action(self):
        """使用 agent 预测下一步动作"""
        if not self.use_agent or self.agent is None:
            raise RuntimeError("Agent not enabled. Set use_agent=True when creating session.")
        
        if self.env is None:
            raise RuntimeError("Session not initialized. Call init() first.")
        
        # 获取当前 episode 的指令
        instruction = _get_instruction_text(self.env.current_episode)
        if instruction is None:
            instruction = "Navigate to the target location."
        
        print(f"\n[INFO] Instruction: {instruction}")
        
        # 使用 agent 预测路线
        route_number, output_text = self.agent.predict_route(instruction)
        
        return {
            "route_number": route_number,
            "output_text": output_text,
            "instruction": instruction
        }

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Session not initialized. Call init() first.")
        self.obs, info, done, episode_id = step(self.env, action)
        self.step_idx += 1
        
        # 如果使用 agent，添加新观测到历史队列
        if self.use_agent and self.agent is not None:
            rgb = self.obs.get("rgb", self.obs.get("rgb_sensor", None))
            depth = self.obs.get("depth", self.obs.get("depth_sensor", None))
            if rgb is not None and depth is not None:
                self.agent.add_observation(rgb, depth)
                print(f"[INFO] Step {self.step_idx} observation added to agent (history length: {len(self.agent.rgb_history)})")
        
        meta = save_observation(self.env, self.obs, self.result_path, step_idx=self.step_idx)
        print(f"[INFO] Step {self.step_idx} saved.")
        return {
            "meta": meta,
            "info": info,
            "done": done,
            "episode_id": episode_id,
        }

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


def run_single_scene_observe(exp_config: str, result_path: str, episode_id: Optional[str], scene_id: Optional[str]) -> None:
    # 增量式：仅初始化并保存当前观测，不自动执行动作；外部可多次调用 session.step(action)
    session = InteractiveSession(BASE_DIR, exp_config, result_path, episode_id, scene_id)
    obs = session.init()
    out_meta_init = session.save_current()
    print(json.dumps({"init": out_meta_init}, ensure_ascii=False, indent=2))

    session.step(ACTION)
    out_meta_step = session.save_current()
    print(json.dumps({"step": out_meta_step}, ensure_ascii=False, indent=2))


def run_with_agent_demo(exp_config: str, result_path: str, episode_id: Optional[str], scene_id: Optional[str], max_steps: int = 10) -> None:
    """
    使用 RVLN Agent 进行自动导航的演示
    
    参数:
        exp_config: 实验配置文件路径
        result_path: 结果保存路径
        episode_id: episode ID（可选）
        scene_id: scene ID（可选）
        max_steps: 最大步数
    """
    # 创建带 agent 的交互会话
    session = InteractiveSession(BASE_DIR, exp_config, result_path, episode_id, scene_id, use_agent=True)
    
    # 初始化环境和模型
    print("=" * 60)
    print("初始化环境和 RVLN Agent...")
    print("=" * 60)
    obs = session.init()
    out_meta_init = session.save_current()
    print(json.dumps({"init": out_meta_init}, ensure_ascii=False, indent=2))
    
    # 开始导航循环
    for step_idx in range(max_steps):
        print("\n" + "=" * 60)
        print(f"Step {step_idx + 1}/{max_steps}")
        print("=" * 60)
        
        # 使用 agent 预测下一步动作
        prediction = session.predict_next_action()
        print(f"\n[PREDICTION]")
        print(f"  Route Number: {prediction['route_number']}")
        print(f"  Model Output: {prediction['output_text']}")
        print(f"  Instruction: {prediction['instruction']}")
        
        # 将路线编号映射为 Habitat 动作
        # -1: STOP, 0: TURN_LEFT_30, 1-7: 路线编号, 8: TURN_RIGHT_30
        # 这里需要根据实际情况映射，暂时使用简单映射
        route_num = prediction['route_number']
        if route_num == -1:
            action = 0  # STOP
            print(f"  -> Action: STOP")
        elif route_num == 0:
            action = 2  # TURN_LEFT
            print(f"  -> Action: TURN_LEFT")
        elif route_num == 8:
            action = 3  # TURN_RIGHT
            print(f"  -> Action: TURN_RIGHT")
        else:
            action = 1  # MOVE_FORWARD (默认)
            print(f"  -> Action: MOVE_FORWARD (route {route_num})")
        
        # 执行动作
        result = session.step(action)
        
        print(f"\n[RESULT]")
        print(f"  Done: {result['done']}")
        print(f"  Episode ID: {result['episode_id']}")
        
        # 如果任务完成，退出循环
        if result['done']:
            print("\n" + "=" * 60)
            print("Episode 完成!")
            print("=" * 60)
            break
    
    # 关闭环境
    session.close()
    print("\n环境已关闭")


if __name__ == "__main__":
    # 原始单步执行模式（不使用 agent）
    # run_single_scene_observe(EXP_CONFIG, RESULT_PATH, EPISODE_ID, SCENE_ID)
    
    # 使用 RVLN Agent 自动导航模式
    run_with_agent_demo(EXP_CONFIG, RESULT_PATH, EPISODE_ID, SCENE_ID, max_steps=20)
