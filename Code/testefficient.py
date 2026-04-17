import json
import os
import random
import time
import torch
import gc
import psutil
import GPUtil
import sys
from statistics import mean
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging level to reduce redundant output
# logging.getLogger('easyeditor').setLevel(logging.WARNING)
# logging.getLogger('transformers').setLevel(logging.WARNING)

from easyeditor import (
    MultimodalEditor, 
    WISEMultimodalHyperParams,
    GraceHyperParams,
    IKEMultimodalHyperParams,
    LoRAMultimodalHyperParams
)

# Create Rich console for formatted output
console = Console()

class SuppressOutput:
    """Context manager to suppress stdout output"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_memory_thorough(reset_cuda=False):
    """Thoroughly clean memory and CUDA cache"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        if reset_cuda:
            try:
                torch.cuda.init()
            except:
                pass
    
    gc.collect()
    
    if hasattr(torch, '_C'):
        if hasattr(torch._C, '_cuda_emptyCache'):
            torch._C._cuda_emptyCache()

def get_gpu_memory_info(device_id=0):
    """Get GPU memory information"""
    GPUs = GPUtil.getGPUs()
    if device_id < len(GPUs):
        gpu = GPUs[device_id]
        return {
            'used': gpu.memoryUsed,
            'total': gpu.memoryTotal,
            'free': gpu.memoryFree,
            'utilization': gpu.memoryUtil * 100
        }
    return None

def get_cpu_memory_info():
    """Get CPU memory information"""
    memory = psutil.virtual_memory()
    return {
        'used': memory.used / (1024**3),
        'total': memory.total / (1024**3),
        'available': memory.available / (1024**3),
        'percent': memory.percent
    }

def measure_memory_and_time(func, *args, device_id=0, **kwargs):
    """Measure memory and time consumption of a function"""
    gc.collect()
    torch.cuda.empty_cache()
    
    gpu_info_before = get_gpu_memory_info(device_id)
    cpu_info_before = get_cpu_memory_info()
    
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    gpu_info_after = get_gpu_memory_info(device_id)
    cpu_info_after = get_cpu_memory_info()
    
    gpu_memory_delta = 0
    if gpu_info_before and gpu_info_after:
        gpu_memory_delta = gpu_info_after['used'] - gpu_info_before['used']
    
    cpu_memory_delta = cpu_info_after['used'] - cpu_info_before['used']
    
    return {
        'result': result,
        'execution_time': execution_time,
        'gpu_memory_delta': gpu_memory_delta,
        'cpu_memory_delta': cpu_memory_delta,
        'gpu_memory_peak': gpu_info_after['used'] if gpu_info_after else 0,
        'cpu_memory_peak': cpu_info_after['used']
    }

def prepare_test_data(json_path: str, 
                     main_image_base: str,
                     loc_image_base: str,
                     num_samples_per_category: int = 5,
                     random_seed: int = 42) -> List[Dict]:
    """Prepare test data for both understanding and reasoning categories"""
    with open(json_path, 'r') as f:
        datalist = json.load(f)
    
    # Split data into understanding (first 1355) and reasoning (remaining)
    understanding_data = datalist[:1355]
    reasoning_data = datalist[1355:]
    
    # Set random seed
    if random_seed is not None:
        random.seed(random_seed)
    
    # Sample from each category
    understanding_indices = random.sample(range(len(understanding_data)), 
                                       min(num_samples_per_category, len(understanding_data)))
    reasoning_indices = random.sample(range(len(reasoning_data)), 
                                     min(num_samples_per_category, len(reasoning_data)))
    
    # Adjust reasoning indices to account for offset
    reasoning_indices = [i + 1355 for i in reasoning_indices]
    
    # Prepare data for each category
    categories = {
        'understanding': {'data': understanding_data, 'indices': understanding_indices},
        'reasoning': {'data': reasoning_data, 'indices': reasoning_indices}
    }
    
    test_data_list = []
    
    for category, cat_info in categories.items():
        selected_data = [cat_info['data'][i] if category == 'understanding' else 
                        cat_info['data'][i - 1355] for i in cat_info['indices']]
        
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
        
        for item in selected_data:
            options_str = "\n\noptions:\n" + "\n".join(
                [f"{key}: {value}" for key, value in item["options"].items()]
            )
            full_question = item["question"] + options_str
            prompts.append(full_question)
            targets.append(item["label_ans"])
            
            item_image_files = item.get("images", [])
            if item_image_files:
                item_image_paths = [os.path.join(main_image_base, img) for img in item_image_files]
                if len(item_image_paths) == 1:
                    images.append(item_image_paths[0])
                    rephrase_images.append(item_image_paths[0])
                    file_types.append("image")
                else:
                    images.append(item_image_paths)
                    rephrase_images.append(item_image_paths)
                    file_types.append("multi-image")
            else:
                images.append([])
                rephrase_images.append([])
                file_types.append("")
            
            rephrased_question = item["rephase_question"] + options_str
            rephrase_prompts.append(rephrased_question)
            
            loc_image = os.path.join(loc_image_base, item["m_loc_img"])
            locality_inputs["text"]["prompt"].append(item["t_loc_q"])
            locality_inputs["text"]["ground_truth"].append(item["t_loc_ans"])
            locality_inputs["vision"]["image"].append(loc_image)
            locality_inputs["vision"]["prompt"].append(item["m_loc_q"])
            locality_inputs["vision"]["ground_truth"].append(item["m_loc_ans"])
        
        test_data_list.append({
            'category': category,
            'prompts': prompts,
            'targets': targets,
            'images': images,
            'file_types': file_types,
            'rephrase_prompts': rephrase_prompts,
            'rephrase_images': rephrase_images,
            'locality_inputs': locality_inputs,
            'selected_indices': cat_info['indices']
        })
    
    return test_data_list

def test_single_method(method_name: str,
                      hparams_path: str,
                      test_data: Dict,
                      device_id: int = 0,
                      output_dir: str = "./performance_results",
                      verbose: bool = False) -> Dict:
    """Test performance of a single method"""
    console.print(f"\n[bold blue]Testing method: {method_name} (Category: {test_data['category']})[/bold blue]")
    
    hparams_class_map = {
        'WISE': WISEMultimodalHyperParams,
        'GRACE': GraceHyperParams,
        'IKE': IKEMultimodalHyperParams,
        'LORA': LoRAMultimodalHyperParams
    }
    
    alg_name = method_name.split('_')[0].upper()
    if alg_name not in hparams_class_map:
        console.print(f"[red]Unknown method: {method_name} (Algorithm: {alg_name})[/red]")
        console.print(f"[yellow]Supported methods: {list(hparams_class_map.keys())}[/yellow]")
        return None
    
    try:
        HyperParamsClass = hparams_class_map[alg_name]
        hparams = HyperParamsClass.from_hparams(hparams_path)
        hparams.device = device_id
        
        def create_editor():
            return MultimodalEditor.from_hparams(hparams)
        
        console.print("[yellow]Creating editor...[/yellow]")
        editor_metrics = measure_memory_and_time(create_editor, device_id=device_id)
        editor = editor_metrics['result']
        
        def run_edit():
            if not verbose and alg_name in ['WISE', 'LORA']:
                with SuppressOutput():
                    metrics, edited_model, weights = editor.edit(
                        prompts=test_data['prompts'],
                        targets=test_data['targets'],
                        file_type=test_data['file_types'],
                        image=test_data['images'],
                        rephrase_prompts=test_data['rephrase_prompts'],
                        locality_inputs=test_data['locality_inputs'],
                        sequential_edit=False,
                        keep_original_weight=True,
                        eval_metric='token em',
                        use_data_augmentation=False,
                        write_file=None
                    )
            else:
                metrics, edited_model, weights = editor.edit(
                    prompts=test_data['prompts'],
                    targets=test_data['targets'],
                    file_type=test_data['file_types'],
                    image=test_data['images'],
                    rephrase_prompts=test_data['rephrase_prompts'],
                    locality_inputs=test_data['locality_inputs'],
                    sequential_edit=False,
                    keep_original_weight=True,
                    eval_metric='token em',
                    use_data_augmentation=False,
                    write_file=None
                )
            return metrics, edited_model, weights
        
        console.print("[yellow]Executing edit...[/yellow]")
        edit_metrics = measure_memory_and_time(run_edit, device_id=device_id)
        metrics, edited_model, weights = edit_metrics['result']
        
        if metrics:
            rewrite_acc = mean([m['post']['rewrite_acc'].item() for m in metrics if 'rewrite_acc' in m['post']])
            rephrase_acc = mean([m['post']['rephrase_acc'].item() for m in metrics if 'rephrase_acc' in m['post']])
        else:
            rewrite_acc = 0.0
            rephrase_acc = 0.0
        
        results = {
            'method_name': method_name,
            'category': test_data['category'],
            'num_samples': len(test_data['prompts']),
            'editor_creation_time': editor_metrics['execution_time'],
            'editor_gpu_memory': editor_metrics['gpu_memory_delta'],
            'edit_execution_time': edit_metrics['execution_time'],
            'edit_gpu_memory': edit_metrics['gpu_memory_delta'],
            'total_time': editor_metrics['execution_time'] + edit_metrics['execution_time'],
            'peak_gpu_memory': max(editor_metrics['gpu_memory_peak'], edit_metrics['gpu_memory_peak']),
            'rewrite_accuracy': rewrite_acc,
            'rephrase_accuracy': rephrase_acc,
            'avg_time_per_sample': edit_metrics['execution_time'] / len(test_data['prompts']),
            'timestamp': datetime.now().isoformat()
        }
        
        console.print("[dim]Performing deep cleanup...[/dim]")
        
        if 'metrics' in locals():
            del metrics
        
        if 'edited_model' in locals():
            if hasattr(edited_model, 'cpu'):
                try:
                    edited_model.cpu()
                except:
                    pass
            del edited_model
        
        if hasattr(editor, 'model'):
            if hasattr(editor.model, '_forward_hooks'):
                editor.model._forward_hooks.clear()
            if hasattr(editor.model, '_backward_hooks'):
                editor.model._backward_hooks.clear()
            
            if alg_name in ['WISE', 'GRACE'] and hasattr(editor, 'reset_layer'):
                try:
                    editor.reset_layer()
                except:
                    pass
            
            try:
                editor.model.cpu()
            except:
                pass
            del editor.model
        
        del editor
        
        if 'weights' in locals() and weights is not None:
            if hasattr(weights, '__call__'):
                try:
                    weights()
                except:
                    pass
            del weights
        
        clean_memory_thorough()
        
        return results
        
    except Exception as e:
        console.print(f"[red]Error testing {method_name} (Category: {test_data['category']}): {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        return None

def benchmark_all_methods(config: Dict) -> List[Dict]:
    """Benchmark all methods for both categories"""
    set_seed(config.get('random_seed', 42))
    
    console.print("[bold green]Preparing test data...[/bold green]")
    test_data_list = prepare_test_data(
        json_path=config['data_path'],
        main_image_base=config['main_image_base'],
        loc_image_base=config['loc_image_base'],
        num_samples_per_category=config['num_samples'],
        random_seed=config.get('random_seed', 42)
    )
    
    all_results = []
    verbose = config.get('verbose', False)
    
    for test_data in test_data_list:
        console.print(f"\n[bold cyan]Testing category: {test_data['category']}[/bold cyan]")
        console.print(f"[green]Selected {len(test_data['prompts'])} samples[/green]")
        console.print(f"[green]Sample indices: {test_data['selected_indices']}[/green]")
        
        for i, method_config in enumerate(config['methods']):
            console.print(f"\n[cyan]Progress: {i+1}/{len(config['methods'])}[/cyan]")
            
            result = test_single_method(
                method_name=method_config['name'],
                hparams_path=method_config['hparams_path'],
                test_data=test_data,
                device_id=config.get('device_id', 0),
                output_dir=config.get('output_dir', './performance_results'),
                verbose=verbose
            )
            
            if result:
                all_results.append(result)
            
            console.print("[dim]Cleaning up resources...[/dim]")
            clean_memory_thorough()
            set_seed(config.get('random_seed', 42))
            time.sleep(5)
    
    return all_results

def display_results(results: List[Dict]):
    """Display results in a table, separated by category"""
    for category in ['understanding', 'reasoning']:
        table = Table(title=f"Multimodal Editing Performance Comparison ({category.capitalize()})")
        
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Samples", style="magenta")
        table.add_column("Total Time(s)", style="green")
        table.add_column("Avg Time/Sample(s)", style="green")
        table.add_column("Peak GPU Memory(MB)", style="yellow")
        table.add_column("Rewrite Accuracy", style="blue")
        table.add_column("Rephrase Accuracy", style="blue")
        
        category_results = [r for r in results if r['category'] == category]
        
        for r in category_results:
            table.add_row(
                r['method_name'],
                str(r['num_samples']),
                f"{r['total_time']:.2f}",
                f"{r['avg_time_per_sample']:.2f}",
                f"{r['peak_gpu_memory']:.2f}",
                f"{r['rewrite_accuracy']:.4f}",
                f"{r['rephrase_accuracy']:.4f}"
            )
        
        console.print(table)

def save_results(results: List[Dict], output_path: str):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    console.print(f"[green]Results saved to: {output_path}[/green]")

def main():
    """Main function: Run performance benchmark"""
    config = {
        'data_path': '../dataset/filtered_data.json',
        'main_image_base': '/root/autodl-tmp/Med-Project02/dataset/images2',
        'loc_image_base': '/root/autodl-tmp/Med-Project02/dataset',
        'num_samples': 5,  # Samples per category
        'random_seed': 42,
        'device_id': 0,
        'output_dir': './performance_results',
        'methods': [
            {
                'name': 'IKE_llavaov',
                'hparams_path': 'hparams/IKE/llavaov_7b.yaml'
            },
            {
                'name': 'IKE_qwen2vl',
                'hparams_path': 'hparams/IKE/qwen2vl_7b.yaml'
            },
            {
                'name': 'IKE_huatuo',
                'hparams_path': 'hparams/IKE/huatuo.yaml'
            },
            {
                'name': 'LoRA_llavaov',
                'hparams_path': 'hparams/LoRA/llavaov-7b.yaml'
            },
            {
                'name': 'LoRA_qwen2vl',
                'hparams_path': 'hparams/LoRA/qwen2vl-7b.yaml'
            },
            {
                'name': 'LoRA_huatuo',
                'hparams_path': 'hparams/LoRA/huatuo.yaml'
            },
            {
                'name': 'WISE_llavaov',
                'hparams_path': 'hparams/WISE/llavaov-7b.yaml'
            },
            {
                'name': 'WISE_qwen2vl',
                'hparams_path': 'hparams/WISE/qwen2-vl-7b.yaml'
            },
            {
                'name': 'WISE_huatuo',
                'hparams_path': 'hparams/WISE/huatuo.yaml'
            },
            {
                'name': 'GRACE_llavaov',
                'hparams_path': 'hparams/GRACE/llavaov-7b.yaml'
            },
            {
                'name': 'GRACE_qwen2vl',
                'hparams_path': 'hparams/GRACE/qwen2vl-7b.yaml'
            },
            {
                'name': 'GRACE_huatuo',
                'hparams_path': 'hparams/GRACE/huatuo.yaml'
            }
        ]
    }
    
    console.print("[bold magenta]Starting multimodal editing performance benchmark[/bold magenta]")
    results = benchmark_all_methods(config)
    
    if results:
        display_results(results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            config['output_dir'], 
            f'benchmark_results_{timestamp}.json'
        )
        save_results(results, output_path)
    else:
        console.print("[red]No tests completed successfully[/red]")

if __name__ == "__main__":
    main()