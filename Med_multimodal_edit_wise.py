import json
import os
import random
from statistics import mean
from rich.console import Console
from typing import Dict, List, Tuple
import numpy as np
import torch
import types
from sentence_transformers import SentenceTransformer

from easyeditor import (BaseEditor, CaptionDataset, FTHyperParams, GraceHyperParams,
                        IKEMultimodalHyperParams, MENDMultimodalHparams,
                        MENDMultimodalTrainingHparams, MultimodalEditor,
                        MultimodalTrainer, SERACMultimodalHparams,
                        SERACMultimodalTrainingHparams, VQADataset,
                        WISEMultimodalHyperParams, encode_ike_facts_multimodal)

def set_seed(seed=42):
    """设置随机种子以确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def print_result(metrics):
#     """打印评估结果，处理可能为 None 的值"""
#     # 修复：去掉.item()调用，因为值已经是float类型
#     rewrite_acc = mean([m['post']['rewrite_acc'] for m in metrics])
#     rephrase_acc = mean([m['post']['rephrase_acc'] for m in metrics])

#     # 处理可能有 rephrase_image_acc 的情况
#     rephrase_image_acc_values = []
#     for m in metrics:
#         if 'rephrase_image_acc' in m['post'] and m['post']['rephrase_image_acc'] is not None:
#             rephrase_image_acc_values.append(m['post']['rephrase_image_acc'])

#     # 处理可能为 None 的 locality_acc
#     locality_acc_values = []
#     locality_image_acc_values = []

#     for m in metrics:
#         # 只收集非 None 的值
#         if 'locality_acc' in m['post'] and m['post']['locality_acc'] is not None:
#             locality_acc_values.append(m['post']['locality_acc'])

#         if 'multimodal_locality_acc' in m['post'] and m['post']['multimodal_locality_acc'] is not None:
#             locality_image_acc_values.append(m['post']['multimodal_locality_acc'])

#     # 打印结果
#     print(f'rewrite_acc: {rewrite_acc:.4f}')
#     print(f'rephrase_acc: {rephrase_acc:.4f}')

#     if rephrase_image_acc_values:
#         rephrase_image_acc = mean(rephrase_image_acc_values)
#         print(
#             f'rephrase_image_acc: {rephrase_image_acc:.4f} (基于 {len(rephrase_image_acc_values)}/{len(metrics)} 个有效样本)')
#     else:
#         print(f'rephrase_image_acc: N/A (没有有效样本)')

#     if locality_acc_values:
#         locality_acc = mean(locality_acc_values)
#         print(f'locality_acc: {locality_acc:.4f} (基于 {len(locality_acc_values)}/{len(metrics)} 个有效样本)')
#     else:
#         print(f'locality_acc: N/A (所有样本都无效)')

#     if locality_image_acc_values:
#         locality_image_acc = mean(locality_image_acc_values)
#         print(
#             f'multimodal_locality_acc: {locality_image_acc:.4f} (基于 {len(locality_image_acc_values)}/{len(metrics)} 个有效样本)')
#     else:
#         print(f'multimodal_locality_acc: N/A (所有样本都无效)')

#     # 打印统计信息
#     print(f'\n=== 样本统计 ===')
#     print(f'总样本数: {len(metrics)}')
#     print(f'有效 rephrase_image 样本数: {len(rephrase_image_acc_values)}')
#     print(f'有效 locality 样本数: {len(locality_acc_values)}')
#     print(f'有效 multimodal locality 样本数: {len(locality_image_acc_values)}')
#     print(f'无效 locality 样本数: {len(metrics) - len(locality_acc_values)}')
#     print(f'无效 multimodal locality 样本数: {len(metrics) - len(locality_image_acc_values)}')

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


    

def test_WISE_LLaVA_OneVision_VQA():
    with open('../dataset/filtered_data.json', 'r') as f:
        datalist = json.load(f)

    # 仅处理前100条数据
    MAX_ITEMS = 1355

    # 传递给 prepare_for_multimodal_editor 的 max_items 参数
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        max_items=MAX_ITEMS  # 添加 max_items 参数
        # start_idx=1355  # 添加 start_idx 参数，默认为0

    )

    print(f"处理 {len(prompts)} 条数据")

    # 添加调试信息
    print("=== 调试信息 ===")
    print(f"prompts 数量: {len(prompts)}")
    print(f"targets 数量: {len(targets)}")
    print(f"images 数量: {len(images)}")
    print(f"locality_inputs text prompts 数量: {len(locality_inputs['text']['prompt'])}")
    print(f"locality_inputs text ground_truth 数量: {len(locality_inputs['text']['ground_truth'])}")
    print(f"locality_inputs vision prompts 数量: {len(locality_inputs['vision']['prompt'])}")
    print(f"locality_inputs vision ground_truth 数量: {len(locality_inputs['vision']['ground_truth'])}")
    print(f"locality_inputs vision images 数量: {len(locality_inputs['vision']['image'])}")

    # 检查是否有空的 locality 数据
    empty_text_prompts = sum(1 for p in locality_inputs['text']['prompt'] if not p or p.strip() == '')
    empty_text_gt = sum(1 for gt in locality_inputs['text']['ground_truth'] if not gt or gt.strip() == '')
    empty_vision_prompts = sum(1 for p in locality_inputs['vision']['prompt'] if not p or p.strip() == '')
    empty_vision_gt = sum(1 for gt in locality_inputs['vision']['ground_truth'] if not gt or gt.strip() == '')

    print(f"空的 text prompts: {empty_text_prompts}")
    print(f"空的 text ground_truth: {empty_text_gt}")
    print(f"空的 vision prompts: {empty_vision_prompts}")
    print(f"空的 vision ground_truth: {empty_vision_gt}")

    # 检查前几个样本
    # print("\n=== 前3个样本检查 ===")
    # for i in range(min(3, len(prompts))):
    #     print(f"\n样本 {i}:")
    #     print(f"  prompt: {prompts[i][:3]}...")
    #     print(f"  target: {targets[i]}")
    #     print(f"  text locality prompt: {locality_inputs['text']['prompt'][i]}")
    #     print(f"  text locality gt: {locality_inputs['text']['ground_truth'][i]}")
    #     print(f"  vision locality prompt: {locality_inputs['vision']['prompt'][i]}")
    #     print(f"  vision locality gt: {locality_inputs['vision']['ground_truth'][i]}")
    # print("================\n")
    params = 'hparams/WISE/llavaov-7b.yaml'
    hparams = WISEMultimodalHyperParams.from_hparams(params)
    editor = MultimodalEditor.from_hparams(hparams)
    # torch.autograd.set_detect_anomaly(True)
    # 继续原有的编辑调用...
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        file_type=file_types,
        image=images,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        use_data_augmentation=False,  # 禁用数据增强以简化调试
        # write_file="./temp_res/WISE_llava_reasoning.json",
        write_file="./temp_res/WISE_llava_understanding.json",  # 修改为正确的输出文件路径
    )
    print(params)
    print("编辑完成，评估结果：")
    print_result(metrics)

def test_WISE_qwen2_VQA():
    with open('../dataset/filtered_data.json', 'r') as f:
        datalist = json.load(f)

    # 仅处理前100条数据
    MAX_ITEMS = 1355

    # 传递给 prepare_for_multimodal_editor 的 max_items 参数
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        max_items=MAX_ITEMS  # 添加 max_items 参数                 understanding
        # start_idx=1355  # 添加 start_idx 参数，默认为0            reasoning
    )

    print(f"处理 {len(prompts)} 条数据")

    # 添加调试信息
    print("=== 调试信息 ===")
    print(f"prompts 数量: {len(prompts)}")
    print(f"targets 数量: {len(targets)}")
    print(f"images 数量: {len(images)}")
    print(f"locality_inputs text prompts 数量: {len(locality_inputs['text']['prompt'])}")
    print(f"locality_inputs text ground_truth 数量: {len(locality_inputs['text']['ground_truth'])}")
    print(f"locality_inputs vision prompts 数量: {len(locality_inputs['vision']['prompt'])}")
    print(f"locality_inputs vision ground_truth 数量: {len(locality_inputs['vision']['ground_truth'])}")
    print(f"locality_inputs vision images 数量: {len(locality_inputs['vision']['image'])}")

    # 检查是否有空的 locality 数据
    empty_text_prompts = sum(1 for p in locality_inputs['text']['prompt'] if not p or p.strip() == '')
    empty_text_gt = sum(1 for gt in locality_inputs['text']['ground_truth'] if not gt or gt.strip() == '')
    empty_vision_prompts = sum(1 for p in locality_inputs['vision']['prompt'] if not p or p.strip() == '')
    empty_vision_gt = sum(1 for gt in locality_inputs['vision']['ground_truth'] if not gt or gt.strip() == '')

    print(f"空的 text prompts: {empty_text_prompts}")
    print(f"空的 text ground_truth: {empty_text_gt}")
    print(f"空的 vision prompts: {empty_vision_prompts}")
    print(f"空的 vision ground_truth: {empty_vision_gt}")

    # 检查前几个样本
    # print("\n=== 前3个样本检查 ===")
    # for i in range(min(3, len(prompts))):
    #     print(f"\n样本 {i}:")
    #     print(f"  prompt: {prompts[i][:3]}...")
    #     print(f"  target: {targets[i]}")
    #     print(f"  text locality prompt: {locality_inputs['text']['prompt'][i]}")
    #     print(f"  text locality gt: {locality_inputs['text']['ground_truth'][i]}")
    #     print(f"  vision locality prompt: {locality_inputs['vision']['prompt'][i]}")
    #     print(f"  vision locality gt: {locality_inputs['vision']['ground_truth'][i]}")
    # print("================\n")
    params = 'hparams/WISE/qwen2-vl-7b.yaml'
    hparams = WISEMultimodalHyperParams.from_hparams(params)
    editor = MultimodalEditor.from_hparams(hparams)
    # print(prompts)
    # print(targets)
    # print(file_types)
    # print(locality_inputs)
    # print(rephrase_prompts)
    # print("Model structure:")
    # for name, module in editor.model.named_children():
    #     print(f"  - {name}")
    # # 使用更小的批处理大小（如果支持）
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        targets=targets,
        file_type=file_types,
        image=images,
        rephrase_prompts=rephrase_prompts,
        # rephrase_image=rephrase_images,  # Pass the prepared rephrase images
        locality_inputs=locality_inputs,
        # loc_prompts=loc_prompts,
        # train_ds=train_ds,
        sequential_edit=False,
        keep_original_weight=True,
        eval_metric='token em',
        # file_type=file_type,
        # use_data_augmentation=True,
        # data_augmentation_prompts=data_augmentation_prompt,
        write_file="./temp_res/WISE_qwen2.json",
    )
    print(params)
    print("编辑完成，评估结果：")
    print(metrics)
    print_result(metrics)


def test_WISE_Med_qwen2vl_VQA():
    print("=== 开始测试 WISE Med Qwen2VL ===")
    
    # 检查数据文件
    # # data_file = '../dataset/combined.json'
    data_file = '../dataset/filtered_data.json'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        datalist = json.load(f)

    MAX_ITEMS = 1355  # 设置要处理的最大条目数
    print(f"加载了 {len(datalist)} 条数据，将处理前 {MAX_ITEMS} 条")

    # 准备数据
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        max_items=MAX_ITEMS
    )

    print(f"处理 {len(prompts)} 条数据")

    # 检查配置文件
    config_file = 'hparams/WISE/med-qwen2-vl-2b.yaml'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    print(f"✅ 配置文件存在: {config_file}")
    
    # 加载超参数
    try:
        hparams = WISEMultimodalHyperParams.from_hparams(config_file)
        print(f"✅ 超参数加载成功")
        print(f"模型名称: {hparams.model_name}")
        print(f"算法名称: {hparams.alg_name}")
        print(f"设备: {hparams.device}")
        
        # 检查模型路径
        if hasattr(hparams, 'model_name') and isinstance(hparams.model_name, str):
            if hparams.model_name.startswith('./') or hparams.model_name.startswith('/'):
                # 本地路径
                if not os.path.exists(hparams.model_name):
                    print(f"❌ 模型路径不存在: {hparams.model_name}")
                    print("建议修改配置文件中的 model_name 为:")
                    print("  - Qwen/Qwen2-VL-2B-Instruct")
                    print("  - 或确保本地路径正确")
                    return
                else:
                    print(f"✅ 模型路径存在: {hparams.model_name}")
            else:
                print(f"使用 Hugging Face 模型: {hparams.model_name}")
        
    except Exception as e:
        print(f"❌ 超参数加载失败: {e}")
        return

    # 创建编辑器
    try:
        print("开始创建 MultimodalEditor...")
        editor = MultimodalEditor.from_hparams(hparams)
        print("✅ MultimodalEditor 创建成功")
        
        # 检查编辑器状态
        if hasattr(editor, 'model') and editor.model is not None:
            print(f"✅ 模型加载成功: {type(editor.model)}")
        else:
            print("❌ 模型未正确加载")
            return
            
    except Exception as e:
        print(f"❌ MultimodalEditor 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 执行编辑
    try:
        print("开始执行编辑...")
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            targets=targets,
            file_type=file_types,
            image=images,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            sequential_edit=False,
            keep_original_weight=True,
            eval_metric='token em',
            write_file="/mnt/40t/medEdit_nuaa/EasyEdit/temp_res/WISE_Med_qwen2.json",
        )
        print("✅ 编辑执行成功")
        print_result(metrics)
        
    except Exception as e:
        print(f"❌ 编辑执行失败: {e}")
        import traceback
        traceback.print_exc()

def test_WISE_mistral():
    print("=== 开始测试 WISE mistral ===")

    # 检查数据文件
    # # data_file = '../dataset/combined.json'
    data_file = '../dataset/filtered_data.json'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        datalist = json.load(f)

    MAX_ITEMS = 1355  # 设置要处理的最大条目数
    print(f"加载了 {len(datalist)} 条数据，将处理前 {MAX_ITEMS} 条")

    # 准备数据
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        max_items=MAX_ITEMS,
        # start_idx=1355  # 添加 start_idx 参数，默认为0
    )

    print(f"处理 {len(prompts)} 条数据")

    # 检查配置文件
    config_file = 'hparams/WISE/mistral-7b.yaml'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    print(f"✅ 配置文件存在: {config_file}")
    
    # 加载超参数
    try:
        hparams = WISEMultimodalHyperParams.from_hparams(config_file)
        print(f"✅ 超参数加载成功")
        print(f"模型名称: {hparams.model_name}")
        print(f"算法名称: {hparams.alg_name}")
        print(f"设备: {hparams.device}")
        
        # 检查模型路径
        if hasattr(hparams, 'model_name') and isinstance(hparams.model_name, str):
            if hparams.model_name.startswith('./') or hparams.model_name.startswith('/'):
                # 本地路径
                if not os.path.exists(hparams.model_name):
                    print(f"❌ 模型路径不存在: {hparams.model_name}")
                    print("建议修改配置文件中的 model_name 为:")
                    print("  - Qwen/Qwen2-VL-2B-Instruct")
                    print("  - 或确保本地路径正确")
                    return
                else:
                    print(f"✅ 模型路径存在: {hparams.model_name}")
            else:
                print(f"使用 Hugging Face 模型: {hparams.model_name}")
        
    except Exception as e:
        print(f"❌ 超参数加载失败: {e}")
        return

    # 创建编辑器
    try:
        print("开始创建 MultimodalEditor...")
        editor = MultimodalEditor.from_hparams(hparams)
        print("✅ MultimodalEditor 创建成功")
        
        # 检查编辑器状态
        if hasattr(editor, 'model') and editor.model is not None:
            print(f"✅ 模型加载成功: {type(editor.model)}")
        else:
            print("❌ 模型未正确加载")
            return
            
    except Exception as e:
        print(f"❌ MultimodalEditor 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 执行编辑
    try:
        print("开始执行编辑...")
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            targets=targets,
            file_type=file_types,
            image=images,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            sequential_edit=False,
            keep_original_weight=True,
            eval_metric='token em',
            write_file="./temp_res/WISE_Med_mistral.json",
        )
        print("✅ 编辑执行成功")
        print_result(metrics)
        
    except Exception as e:
        print(f"❌ 编辑执行失败: {e}")
        import traceback
        traceback.print_exc()

def test_WISE_huatuo():
    print("=== 开始测试 WISE Huatuo ===")

    # 检查数据文件
    # # data_file = '../dataset/combined.json'
    data_file = '../dataset/filtered_data.json'
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    with open(data_file, 'r') as f:
        datalist = json.load(f)

    MAX_ITEMS = 1355  # 设置要处理的最大条目数
    print(f"加载了 {len(datalist)} 条数据，将处理前 {MAX_ITEMS} 条")

    # 准备数据
    prompts, targets, images, file_types, rephrase_prompts, rephrase_images, locality_inputs = prepare_for_multimodal_editor(
        datalist,
        "/root/autodl-tmp/Med-Project02/dataset/images2/",
        "/root/autodl-tmp/Med-Project02/dataset",
        max_items=MAX_ITEMS,
        # start_idx=1355  # 添加 start_idx 参数，默认为0
    )

    print(f"处理 {len(prompts)} 条数据")

    # 检查配置文件
    config_file = 'hparams/WISE/huatuo.yaml'
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return
    
    print(f"✅ 配置文件存在: {config_file}")
    
    # 加载超参数
    try:
        hparams = WISEMultimodalHyperParams.from_hparams(config_file)
        print(f"✅ 超参数加载成功")
        print(f"模型名称: {hparams.model_name}")
        print(f"算法名称: {hparams.alg_name}")
        print(f"设备: {hparams.device}")
        
        # 检查模型路径
        if hasattr(hparams, 'model_name') and isinstance(hparams.model_name, str):
            if hparams.model_name.startswith('./') or hparams.model_name.startswith('/'):
                # 本地路径
                if not os.path.exists(hparams.model_name):
                    print(f"❌ 模型路径不存在: {hparams.model_name}")
                    print("建议修改配置文件中的 model_name 为:")
                    print("  - Qwen/Qwen2-VL-2B-Instruct")
                    print("  - 或确保本地路径正确")
                    return
                else:
                    print(f"✅ 模型路径存在: {hparams.model_name}")
            else:
                print(f"使用 Hugging Face 模型: {hparams.model_name}")
        
    except Exception as e:
        print(f"❌ 超参数加载失败: {e}")
        return

    # 创建编辑器
    try:
        print("开始创建 MultimodalEditor...")
        editor = MultimodalEditor.from_hparams(hparams)
        print("✅ MultimodalEditor 创建成功")
        
        # 检查编辑器状态
        if hasattr(editor, 'model') and editor.model is not None:
            print(f"✅ 模型加载成功: {type(editor.model)}")
        else:
            print("❌ 模型未正确加载")
            return
            
    except Exception as e:
        print(f"❌ MultimodalEditor 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 执行编辑
    try:
        print("开始执行编辑...")

        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            targets=targets,
            file_type=file_types,
            image=images,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            sequential_edit=False,
            keep_original_weight=True,
            eval_metric='token em',
            write_file="./temp_res/WISE_huatuo.json",
        )
        print("✅ 编辑执行成功")
        print_result(metrics)
        
    except Exception as e:
        print(f"❌ 编辑执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    set_seed(42)  # 设置随机种子以确保结果可重现
    # test_WISE_LLaVA_OneVision_VQA()
    test_WISE_qwen2_VQA()
    # test_WISE_Med_qwen2vl_VQA()
    # test_WISE_huatuo()
    # test_WISE_mistral()