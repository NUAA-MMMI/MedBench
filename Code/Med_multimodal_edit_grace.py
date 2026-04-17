import json
import os
import random
from statistics import mean
from typing import Dict, List, Tuple
import numpy as np
import torch
import types
from sentence_transformers import SentenceTransformer

from easyeditor import (BaseEditor, CaptionDataset, FTHyperParams,
                        GraceHyperParams, GraceHyperParams,
                        MENDMultimodalHparams, MENDMultimodalTrainingHparams,
                        MultimodalEditor, MultimodalTrainer,
                        SERACMultimodalHparams, SERACMultimodalTrainingHparams,
                        VQADataset, encode_ike_facts_multimodal)

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_result(metrics):
    """打印评估结果，处理可能为 None 的值"""
    rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics])
    rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics])

    # 处理可能有 rephrase_image_acc 的情况
    rephrase_image_acc_values = []
    for m in metrics:
        if 'rephrase_image_acc' in m['post'] and m['post']['rephrase_image_acc'] is not None:
            rephrase_image_acc_values.append(
                m['post']['rephrase_image_acc'].item() if hasattr(m['post']['rephrase_image_acc'], 'item')
                else m['post']['rephrase_image_acc']
            )

    # 处理可能为 None 的 locality_acc
    locality_acc_values = []
    locality_image_acc_values = []

    for m in metrics:
        # 只收集非 None 的值
        if 'locality_acc' in m['post'] and m['post']['locality_acc'] is not None:
            locality_acc_values.append(
                m['post']['locality_acc'].item() if hasattr(m['post']['locality_acc'], 'item')
                else m['post']['locality_acc']
            )

        if 'multimodal_locality_acc' in m['post'] and m['post']['multimodal_locality_acc'] is not None:
            locality_image_acc_values.append(
                m['post']['multimodal_locality_acc'].item() if hasattr(m['post']['multimodal_locality_acc'], 'item')
                else m['post']['multimodal_locality_acc']
            )

    # 打印结果
    print(f'rewrite_acc: {rewrite_acc:.4f}')
    print(f'rephrase_acc: {rephrase_acc:.4f}')

    if rephrase_image_acc_values:
        rephrase_image_acc = mean(rephrase_image_acc_values)
        print(
            f'rephrase_image_acc: {rephrase_image_acc:.4f} (基于 {len(rephrase_image_acc_values)}/{len(metrics)} 个有效样本)')
    else:
        print(f'rephrase_image_acc: N/A (没有有效样本)')

    if locality_acc_values:
        locality_acc = mean(locality_acc_values)
        print(f'locality_acc: {locality_acc:.4f} (基于 {len(locality_acc_values)}/{len(metrics)} 个有效样本)')
    else:
        print(f'locality_acc: N/A (所有样本都无效)')

    if locality_image_acc_values:
        locality_image_acc = mean(locality_image_acc_values)
        print(
            f'multimodal_locality_acc: {locality_image_acc:.4f} (基于 {len(locality_image_acc_values)}/{len(metrics)} 个有效样本)')
    else:
        print(f'multimodal_locality_acc: N/A (所有样本都无效)')

    # 打印统计信息
    print(f'\n=== 样本统计 ===')
    print(f'总样本数: {len(metrics)}')
    print(f'有效 rephrase_image 样本数: {len(rephrase_image_acc_values)}')
    print(f'有效 locality 样本数: {len(locality_acc_values)}')
    print(f'有效 multimodal locality 样本数: {len(locality_image_acc_values)}')
    print(f'无效 locality 样本数: {len(metrics) - len(locality_acc_values)}')
    print(f'无效 multimodal locality 样本数: {len(metrics) - len(locality_image_acc_values)}')


def prepare_for_multimodal_editor(json_data: List[Dict],
                                  main_image_base: str,
                                  loc_image_base: str,
                                  max_items: int = None,
                                  start_idx: int = 0) -> Tuple:  # 添加 start_idx 参数
    """
    将JSON数据转换为MultimodalEditor.edit()所需的输入格式。
    此版本经过修改，以确保为 llavaov_processors 提供正确的图像数据格式。

    参数:
        json_data: JSON数据列表
        main_image_base: 主图像基础路径
        loc_image_base: 局部性图像基础路径
        max_items: 最大处理条目数（可选）
        start_idx: 开始处理的索引位置（默认为0）

    返回:
        (prompts, targets, images, file_types,
         rephrase_prompts, rephrase_images, locality_inputs)
    """
    prompts = []
    targets = []
    images = []
    file_types = []
    rephrase_prompts = []
    rephrase_images = []

    locality_inputs = {
        "text": {"prompt": [], "ground_truth": []},
        "vision": {"image": [], "prompt": [], "ground_truth": []}
    }

    # 确保 start_idx 在有效范围内
    if start_idx < 0:
        start_idx = 0
    elif start_idx >= len(json_data):
        print(f"警告：start_idx ({start_idx}) 超出数据范围 ({len(json_data)})，返回空结果")
        return prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs

    # 从指定位置开始截取数据
    if max_items is not None:
        end_idx = min(start_idx + max_items, len(json_data))
        json_data = json_data[start_idx:end_idx]
        print(f"处理数据范围：[{start_idx}:{end_idx}]，共 {len(json_data)} 条")
    else:
        json_data = json_data[start_idx:]
        print(f"处理数据范围：[{start_idx}:{len(json_data) + start_idx}]，共 {len(json_data)} 条")

    for item in json_data:
        # 处理主问题
        options_str = "\n\noptions:\n" + "\n".join(
            [f"{key}: {value}" for key, value in item["options"].items()]
        )
        full_question = item["question"] + options_str
        prompts.append(full_question)
        targets.append(item["label_ans"])

        # 处理主图像和重述图像 (支持单/多图像)
        item_image_files = item.get("images")  # 使用 .get() 以确保安全
        if item_image_files:  # 如果 "images" 存在且不为空列表
            item_image_paths = [os.path.join(main_image_base, img) for img in item_image_files]

            if len(item_image_paths) == 1:
                # 对于单个图像，处理器需要一个字符串路径。
                images.append(item_image_paths[0])
                rephrase_images.append(item_image_paths[0])
                file_types.append("image")
            else:
                # 对于多个图像，处理器需要一个字符串路径列表。
                images.append(item_image_paths)
                rephrase_images.append(item_image_paths)
                file_types.append("multi-image")
        else:
            # 对于没有图像的条目，附加空占位符。
            # 注意：下游的 `multimodal_editor` 可能无法在未经修改的情况下正常处理此情况，
            # 因为它可能期望每个条目都有一个图像。
            images.append([])
            rephrase_images.append([])
            file_types.append("")

        # 处理重述问题
        rephrased_question = item["rephase_question"] + options_str
        rephrase_prompts.append(rephrased_question)

        # 处理文本局部性输入
        locality_inputs["text"]["prompt"].append(item["t_loc_q"])
        locality_inputs["text"]["ground_truth"].append(item["t_loc_ans"])

        # 处理视觉局部性输入（单张图像）
        loc_image = os.path.join(loc_image_base, item["m_loc_img"])
        locality_inputs["vision"]["image"].append(loc_image)
        locality_inputs["vision"]["prompt"].append(item["m_loc_q"])
        locality_inputs["vision"]["ground_truth"].append(item["m_loc_ans"])

    return prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs


def test_GRACE_LLaVA_OneVision_VQA():
    """测试 LLaVA OneVision 模型的 GRACE 编辑"""
    # 加载数据
    with open('../dataset/filtered_data.json', 'r') as f:
        datalist = json.load(f)
    MAX_ITEMS = 1355  # 限制数据量为1000条
    # 准备数据
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        # max_items=1355  # 如果需要限制数据量，取消注释        # understanding   --wst
        start_idx=1355  # 从头开始处理数据                 # reasoning       --wst
    )

    print(f"处理 {len(prompts)} 条数据")

    # 加载 GRACE 超参数
    params = 'hparams/GRACE/llavaov-7b.yaml'
    hparams = GraceHyperParams.from_hparams(params)

    # 创建编辑器
    editor = MultimodalEditor.from_hparams(hparams)
    print("Model structure:")
    for name, module in editor.model.named_children():
        print(f"  - {name}")

    # 添加调试信息
    print("\n=== 数据检查 ===")
    print(f"prompts 数量: {len(prompts)}")
    print(f"targets 数量: {len(targets)}")
    print(f"images 数量: {len(images)}")
    print(f"locality_inputs text prompts 数量: {len(locality_inputs['text']['prompt'])}")
    print(f"locality_inputs text ground_truth 数量: {len(locality_inputs['text']['ground_truth'])}")
    print(f"locality_inputs vision prompts 数量: {len(locality_inputs['vision']['prompt'])}")
    print(f"locality_inputs vision ground_truth 数量: {len(locality_inputs['vision']['ground_truth'])}")
    print(f"locality_inputs vision images 数量: {len(locality_inputs['vision']['image'])}")

    # 执行编辑
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        file_type=file_types,
        image=images,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_images,  # GRACE 可能需要这个
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        use_data_augmentation=False,  # GRACE 通常不需要数据增强
        write_file="./temp_res/GRACE_LLaVA.json",
    )
    print(params)
    print_result(metrics)


def test_GRACE_qwen2_VQA():
    """测试 Qwen2-VL 模型的 GRACE 编辑"""
    # 加载数据
    with open('../dataset/filtered_data.json', 'r') as f:
        datalist = json.load(f)
    MAX_ITEMS = 1355  # 如果需要限制数据量，取消注释
    # 准备数据
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        # max_items=MAX_ITEMS,  # 如果需要限制数据量，取消注释
        start_idx=1355  # 从头开始处理数据
    )

    print(f"处理 {len(prompts)} 条数据")
    params = 'hparams/GRACE/qwen2vl-7b.yaml'
    # 加载 GRACE 超参数
    hparams = GraceHyperParams.from_hparams(params)

    # 创建编辑器
    editor = MultimodalEditor.from_hparams(hparams)
    print("Model structure:")
    for name, module in editor.model.named_children():
        print(f"  - {name}")

    # 执行编辑
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        file_type=file_types,
        image=images,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_images,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        use_data_augmentation=False,
        write_file="./temp_res/GRACE_qwen2.json",
    )
    print(params)
    print_result(metrics)

def test_GRACE_huatuo():
    """测试 Huatuo 模型的 GRACE 编辑"""
    # 加载数据
    with open('../dataset/filtered_data.json', 'r') as f:
        datalist = json.load(f)
    MAX_ITEMS = 1355  # 如果需要限制数据量，取消注释
    # 准备数据
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        max_items=MAX_ITEMS,  # 如果需要限制数据量，取消注释
        # start_idx=1355  # 从头开始处理数据
    )

    print(f"处理 {len(prompts)} 条数据")
    params = 'hparams/GRACE/huatuo.yaml'
    # 加载 GRACE 超参数
    hparams = GraceHyperParams.from_hparams(params)

    # 创建编辑器
    editor = MultimodalEditor.from_hparams(hparams)
    print("Model structure:")
    for name, module in editor.model.named_children():
        print(f"  - {name}")

    # 执行编辑
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        file_type=file_types,
        image=images,
        rephrase_prompts=rephrase_prompts,
        rephrase_image=rephrase_images,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        use_data_augmentation=False,
        write_file="./temp_res/GRACE_huatuo.json",
    )
    print(params)
    print_result(metrics)


if __name__ == "__main__":
    set_seed(42)  # 设置随机种子以确保结果可重现
    # 测试 LLaVA OneVision
    # test_GRACE_LLaVA_OneVision_VQA()

    # 测试 Qwen2-VL
    test_GRACE_qwen2_VQA()
    # 测试 Huatuo
    # test_GRACE_huatuo()