import json
import os
from typing import List, Dict, Tuple
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import re


class LLaVAMedErrorFilter:
    """使用 LLaVA-Med 模型筛选出答错的数据"""

    def __init__(self, model_name: str = "microsoft/llava-med-v1.5-mistral-7b", device: str = "cuda"):
        """
        初始化模型

        Args:
            model_name: 模型名称
            device: 运行设备
        """
        self.device = device
        self.model_name = model_name

        # 加载模型和处理器
        print(f"加载模型: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def prepare_question(self, item: Dict) -> str:
        """
        准备问题文本，格式化为单选题

        Args:
            item: 数据项

        Returns:
            格式化的问题文本
        """
        question = item["question"]
        options_str = "\n\nOptions:\n" + "\n".join(
            [f"{key}: {value}" for key, value in item["options"].items()]
        )

        # 构建提示词，要求模型只返回选项字母
        prompt = f"""Please answer the following medical question by selecting the correct option. Only respond with the letter (A, B, C, or D) of the correct answer.

Question: {question}{options_str}

Answer:"""

        return prompt

    def get_model_answer(self, prompt: str, image_paths: List[str]) -> str:
        """
        获取模型的答案

        Args:
            prompt: 问题文本
            image_paths: 图像路径列表

        Returns:
            模型预测的答案
        """
        # 加载图像
        images = []
        for img_path in image_paths:
            if os.path.exists(img_path):
                image = Image.open(img_path).convert("RGB")
                images.append(image)

        # 如果没有图像，使用纯文本模式
        if not images:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            # 有图像时使用多模态输入
            inputs = self.processor(
                text=prompt,
                images=images,
                return_tensors="pt"
            ).to(self.device)

        # 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False
            )

        # 解码答案
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)

        # 提取答案中的选项字母
        answer_text = answer.split("Answer:")[-1].strip()

        # 使用正则表达式提取第一个出现的选项字母
        match = re.search(r'[A-D]', answer_text.upper())
        if match:
            return match.group()

        # 如果没有找到明确的选项，尝试其他模式
        if "A" in answer_text.upper():
            return "A"
        elif "B" in answer_text.upper():
            return "B"
        elif "C" in answer_text.upper():
            return "C"
        elif "D" in answer_text.upper():
            return "D"

        return "N/A"  # 无法识别答案

    def check_answer(self, predicted: str, correct: str) -> bool:
        """
        检查答案是否正确

        Args:
            predicted: 预测答案
            correct: 正确答案

        Returns:
            是否正确
        """
        return predicted.upper().strip() == correct.upper().strip()

    def filter_incorrect_answers(
            self,
            data_path: str,
            image_base_path: str,
            output_path: str = "incorrect_answers.json",
            max_items: int = None,
            save_interval: int = 100
    ) -> Tuple[List[Dict], float]:
        """
        筛选出答错的数据

        Args:
            data_path: JSON数据文件路径
            image_base_path: 图像基础路径
            output_path: 输出文件路径
            max_items: 最大处理数量
            save_interval: 保存间隔

        Returns:
            错误数据列表和准确率
        """
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if max_items:
            data = data[:max_items]

        incorrect_items = []
        correct_count = 0
        total_count = 0

        # 处理每个数据项
        for idx, item in enumerate(tqdm(data, desc="筛选错误答案")):
            try:
                # 准备问题
                prompt = self.prepare_question(item)

                # 准备图像路径
                image_paths = []
                if "images" in item and item["images"]:
                    for img_name in item["images"]:
                        img_path = os.path.join(image_base_path, img_name)
                        image_paths.append(img_path)

                # 获取模型答案
                predicted_answer = self.get_model_answer(prompt, image_paths)
                correct_answer = item["label_ans"]

                # 检查答案
                is_correct = self.check_answer(predicted_answer, correct_answer)

                # 记录结果
                total_count += 1
                if is_correct:
                    correct_count += 1
                else:
                    # 保存错误的数据项
                    error_item = item.copy()
                    error_item["predicted_answer"] = predicted_answer
                    error_item["error_index"] = idx
                    incorrect_items.append(error_item)

                # 定期保存结果
                if (idx + 1) % save_interval == 0:
                    self.save_results(incorrect_items, output_path)
                    print(f"\n已处理 {idx + 1} 项，当前准确率: {correct_count / total_count:.2%}")
                    print(f"错误数量: {len(incorrect_items)}")

            except Exception as e:
                print(f"\n处理第 {idx} 项时出错: {str(e)}")
                continue

        # 保存最终结果
        self.save_results(incorrect_items, output_path)

        # 计算准确率
        accuracy = correct_count / total_count if total_count > 0 else 0

        # 打印统计信息
        print(f"\n=== 筛选完成 ===")
        print(f"总数据量: {total_count}")
        print(f"正确数量: {correct_count}")
        print(f"错误数量: {len(incorrect_items)}")
        print(f"准确率: {accuracy:.2%}")

        # 保存统计信息
        stats = {
            "total_count": total_count,
            "correct_count": correct_count,
            "incorrect_count": len(incorrect_items),
            "accuracy": accuracy,
            "model_name": self.model_name
        }

        stats_path = output_path.replace(".json", "_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return incorrect_items, accuracy

    def save_results(self, incorrect_items: List[Dict], output_path: str):
        """保存结果到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(incorrect_items, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

    def analyze_errors(self, incorrect_items: List[Dict]) -> Dict:
        """
        分析错误类型

        Args:
            incorrect_items: 错误数据列表

        Returns:
            错误分析结果
        """
        error_analysis = {
            "total_errors": len(incorrect_items),
            "error_distribution": {},
            "predicted_distribution": {},
            "correct_distribution": {},
            "no_answer_count": 0
        }

        for item in incorrect_items:
            predicted = item.get("predicted_answer", "N/A")
            correct = item["label_ans"]

            # 统计预测答案分布
            error_analysis["predicted_distribution"][predicted] = \
                error_analysis["predicted_distribution"].get(predicted, 0) + 1

            # 统计正确答案分布
            error_analysis["correct_distribution"][correct] = \
                error_analysis["correct_distribution"].get(correct, 0) + 1

            # 统计错误模式
            error_key = f"{correct}->{predicted}"
            error_analysis["error_distribution"][error_key] = \
                error_analysis["error_distribution"].get(error_key, 0) + 1

            # 统计无法识别的答案
            if predicted == "N/A":
                error_analysis["no_answer_count"] += 1

        return error_analysis


def main():
    """主函数示例"""
    # 配置参数
    data_path = "../dataset/combined.json"
    image_base_path = "/mnt/40t/medEdit_nuaa/dataset/images2/"
    output_path = "llava_med_incorrect_answers.json"

    # 创建筛选器
    filter = LLaVAMedErrorFilter(
        model_name="microsoft/llava-med-v1.5-mistral-7b",
        device="cuda"
    )

    # 执行筛选
    incorrect_items, accuracy = filter.filter_incorrect_answers(
        data_path=data_path,
        image_base_path=image_base_path,
        output_path=output_path,
        max_items=100,  # 可以先测试少量数据
        save_interval=20
    )

    # 分析错误
    if incorrect_items:
        error_analysis = filter.analyze_errors(incorrect_items)

        # 保存错误分析
        analysis_path = output_path.replace(".json", "_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)

        print(f"\n错误分析已保存到: {analysis_path}")
        print(f"错误模式分布: {error_analysis['error_distribution']}")


if __name__ == "__main__":
    main()