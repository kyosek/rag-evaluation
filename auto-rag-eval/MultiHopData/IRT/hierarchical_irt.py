from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
import json
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class ExamResult:
    """Class to store exam results for a model configuration"""
    llm_name: str
    retriever_name: Optional[str] = None
    responses: Optional[List[bool]] = None
    num_hops: Optional[List[int]] = None
    exam_type: Optional[str] = None
    
    @classmethod
    def combine_exam_takers(cls, results: List['ExamResult']) -> 'ExamResult':
        """Combine results from different exam takers (LLMs) that took the same exam by concatenating their responses"""
        if not results:
            raise ValueError("Cannot combine empty list of results")
            
        # Verify all results have compatible type
        first = results[0]
        if not all(r.retriever_name == first.retriever_name for r in results):
            raise ValueError("All results must have the same retriever type")
            
        # Simply concatenate responses and hop numbers
        combined_responses = []
        combined_hops = []
        for result in results:
            combined_responses.extend(result.responses)
            if result.num_hops:
                combined_hops.extend(result.num_hops)
            
        llm_names = "_".join(r.llm_name for r in results)
        
        return cls(
            llm_name=f"combined_{llm_names}",
            retriever_name=first.retriever_name,
            responses=combined_responses,
            num_hops=combined_hops,
            exam_type=first.exam_type
        )
    
    @classmethod
    def from_json_file(cls, filepath: str, llm_name: str, retriever_name: Optional[str] = None, 
                      exam_type: Optional[str] = None) -> 'ExamResult':
        """Create an ExamResult instance from a JSON file containing exam responses"""
        try:
            with open(filepath, 'r') as f:
                # First try to load as a single JSON object
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                except json.JSONDecodeError:
                    # If that fails, try reading line by line
                    f.seek(0)
                    data = []
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Warning: Skipping invalid JSON line in {filepath}")
            
            if not data:
                raise ValueError(f"No valid JSON data found in {filepath}")
            
            responses = []
            num_hops = []
            
            for item in data:
                # Extract response data
                if 'is_correct' in item:
                    responses.append(item['is_correct'])
                else:
                    # Try to determine correctness by comparing answers
                    is_correct = (
                        'model_answer' in item and 
                        'correct_answer' in item and 
                        item['model_answer'].strip().upper() == item['correct_answer'].strip().upper()
                    )
                    responses.append(is_correct)
                
                # Extract hop data
                if 'number_of_hops' in item:
                    num_hops.append(item['number_of_hops'])
                elif 'num_hops' in item:
                    num_hops.append(item['num_hops'])
                else:
                    num_hops.append(0)  # Default to 0 hops if not specified
            
            return cls(
                llm_name=llm_name,
                retriever_name=retriever_name,
                responses=responses,
                num_hops=num_hops,
                exam_type=exam_type
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        except Exception as e:
            raise ValueError(f"Error processing {filepath}: {str(e)}")
    
    @classmethod
    def combine_exam_results(cls, result1: 'ExamResult', result2: 'ExamResult') -> 'ExamResult':
        """Combine two exam results into a single result"""
        if result1.llm_name != result2.llm_name or result1.retriever_name != result2.retriever_name:
            raise ValueError("Cannot combine results from different models or retrievers")
        
        return cls(
            llm_name=result1.llm_name,
            retriever_name=result1.retriever_name,
            responses=result1.responses + result2.responses,
            num_hops=result1.num_hops + result2.num_hops,
            exam_type="combined"
        )
    
    @classmethod
    def from_json_files(cls, filepaths: Dict[str, Dict[str, Union[str, Dict[str, str]]]],
                       combine_exams: bool = False,
                       combine_exam_takers: bool = False) -> List['ExamResult']:
        """Create ExamResult instances from filepaths, with option to combine exam takers"""
        results = []
        exam_taker_groups = defaultdict(list)  # Group by retriever and exam type
        
        for llm_name, retriever_paths in filepaths.items():
            for retriever_name, filepath_info in retriever_paths.items():
                try:
                    ret_name = None if retriever_name == 'closed_book' else retriever_name
                    
                    if combine_exams and isinstance(filepath_info, dict):
                        # Load and combine single/multi hop exam results
                        single_result = cls.from_json_file(
                            filepath_info['single'], llm_name, ret_name, exam_type="single")
                        multi_result = cls.from_json_file(
                            filepath_info['multi'], llm_name, ret_name, exam_type="multi")
                        combined_result = cls.combine_exam_results(single_result, multi_result)
                        
                        if combine_exam_takers:
                            # Group by retriever type for later combination
                            key = (ret_name, combined_result.exam_type)
                            exam_taker_groups[key].append(combined_result)
                        else:
                            results.append(combined_result)
                    else:
                        # Load single exam result
                        filepath = filepath_info if isinstance(filepath_info, str) else filepath_info['default']
                        result = cls.from_json_file(filepath, llm_name, ret_name)
                        
                        if combine_exam_takers:
                            key = (ret_name, result.exam_type)
                            exam_taker_groups[key].append(result)
                        else:
                            results.append(result)
                        
                except Exception as e:
                    print(f"Warning: Failed to load results for {llm_name}/{retriever_name}: {e}")
                    continue
        
        if combine_exam_takers:
            # Combine results from different exam takers for each group
            for (ret_name, exam_type), group_results in exam_taker_groups.items():
                if len(group_results) > 1:  # Only combine if we have multiple results
                    try:
                        combined = cls.combine_exam_takers(group_results)
                        results.append(combined)
                    except Exception as e:
                        print(f"Warning: Failed to combine exam takers for {ret_name}/{exam_type}: {e}")
        
        if not results:
            raise ValueError("No valid exam results could be loaded")
        
        return results
    
class MultihopIRTModel:
    """Hierarchical IRT model for multihop question evaluation"""
    
    def __init__(self, exam_results: Union[ExamResult, List[ExamResult]], num_questions: int):
        # Convert single ExamResult to list if necessary
        if isinstance(exam_results, ExamResult):
            exam_results = [exam_results]
            
        self.exam_results = exam_results
        self.num_questions = num_questions
        
        # Map unique LLMs and retrievers to indices
        self.llm_map = {
            llm: idx for idx, llm in enumerate(
                set(result.llm_name for result in exam_results)
            )
        }
        self.retriever_map = {
            ret: idx for idx, ret in enumerate(
                set(result.retriever_name for result in exam_results if result.retriever_name)
            )
        }
        
        # Convert responses to numpy array
        self.response_matrix = np.array([
            result.responses for result in exam_results
        ])
        
        # Store number of hops
        self.num_hops = np.array([
            result.num_hops for result in exam_results
        ])
        
        # Model dimensions
        self.num_llms = len(self.llm_map)
        self.num_retrievers = len(self.retriever_map)
        self.num_models = len(exam_results)
        self.model_colors = {}
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(exam_results)))
        for i, result in enumerate(exam_results):
            key = f"{result.llm_name}_{result.retriever_name}"
            self.model_colors[key] = colors[i]
        
    def compute_theta(self, theta_params: np.array) -> np.array:
        """Compute ability parameters for each model"""
        llm_params = theta_params[:self.num_llms]
        retriever_params = theta_params[self.num_llms:self.num_llms + self.num_retrievers]
        
        abilities = []
        for result in self.exam_results:
            ability = llm_params[self.llm_map[result.llm_name]]
            if result.retriever_name:
                ability += retriever_params[self.retriever_map[result.retriever_name]]
            abilities.append(ability)
            
        return np.array(abilities)
    
    def irt_3pl(self, theta: np.array, a: float, b: float) -> float:
        """3PL IRT model with fixed guessing parameter c=0.25"""
        c = 0.25  # Fixed for 4-choice questions
        return c + ((1 - c) / (1 + np.exp(-a * (theta - b))))
    
    def neg_log_likelihood(self, params: np.array) -> float:
        """Compute negative log likelihood for optimization"""
        # Unpack parameters
        a = params[:self.num_questions]  # discrimination
        b = params[self.num_questions:2*self.num_questions]  # difficulty
        theta = self.compute_theta(params[2*self.num_questions:])
        
        # Compute likelihood
        likelihood = 0
        for i in range(self.num_questions):
            p = self.irt_3pl(theta=theta, a=a[i], b=b[i])
            likelihood += np.sum(
                self.response_matrix[:,i] * np.log(p) + 
                (1 - self.response_matrix[:,i]) * np.log(1 - p)
            )
        
        # Add hop-based regularization
        hop_penalty = 0.1 * np.sum(np.abs(a * self.num_hops[:,np.newaxis] - b))
        
        return -(likelihood - hop_penalty)
    
    def fit(self) -> Dict[str, np.array]:
        """Fit the IRT model using L-BFGS-B optimization"""
        # Initial parameter guesses
        initial_params = np.concatenate([
            np.ones(self.num_questions),  # a (discrimination)
            np.zeros(self.num_questions),  # b (difficulty)
            np.zeros(self.num_llms + self.num_retrievers)  # theta components
        ])
        
        # Parameter bounds
        bounds = (
            [(0.5, 1.5) for _ in range(self.num_questions)] +  # a bounds
            [(0.01, 1.0) for _ in range(self.num_questions)] +  # b bounds
            [(-3.0, 3.0) for _ in range(self.num_llms + self.num_retrievers)]  # theta bounds
        )
        
        # Optimize
        result = minimize(
            self.neg_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Extract parameters
        params = {
            'discrimination': result.x[:self.num_questions],
            'difficulty': result.x[self.num_questions:2*self.num_questions],
            'theta_params': result.x[2*self.num_questions:],
            'theta': self.compute_theta(result.x[2*self.num_questions:])
        }
        
        return params
    
    def compute_information(self, params: Dict[str, np.array], theta: np.array) -> np.array:
        """Compute Fisher information for questions across ability levels"""
        information = np.zeros((self.num_questions, len(theta)))
        c = 0.25
        
        for i in range(self.num_questions):
            p = self.irt_3pl(theta=theta, a=params['discrimination'][i], b=params['difficulty'][i])
            information[i] = (params['discrimination'][i]**2 * (p - c)**2 * (1 - p)) / ((1 - c)**2 * p)
            
        return information
    
    def save_analysis(self, params: Dict[str, np.array], output_path: str):
        """Save analysis results to a JSON file"""
        # Get unique hop counts
        unique_hops = sorted(list(set(hop for result in self.exam_results for hop in result.num_hops)))
        
        # Calculate average difficulty and discrimination by hop count
        hop_stats = {}
        for hop_count in unique_hops:
            # Get indices for questions with this hop count
            indices = [i for i, hops in enumerate(self.exam_results[0].num_hops) if hops == hop_count]
            
            hop_stats[str(hop_count)] = {
                "avg_difficulty": float(np.mean(params['difficulty'][indices])),
                "avg_discrimination": float(np.mean(params['discrimination'][indices])),
                "num_questions": len(indices)
            }
        
        # Prepare model ability results
        model_abilities = {}
        for result in self.exam_results:
            model_key = f"{result.llm_name}"
            if result.retriever_name:
                model_key += f"_{result.retriever_name}"
            
            idx = self.exam_results.index(result)
            model_abilities[model_key] = float(params['theta'][idx])
        
        # Prepare component abilities
        component_abilities = {
            "llms": {
                llm: float(params['theta_params'][idx])
                for llm, idx in self.llm_map.items()
            },
            "retrievers": {
                ret: float(params['theta_params'][self.num_llms + idx])
                for ret, idx in self.retriever_map.items()
            } if self.retriever_map else {}
        }
        
        # Overall exam statistics
        overall_stats = {
            "avg_difficulty": float(np.mean(params['difficulty'])),
            "avg_discrimination": float(np.mean(params['discrimination'])),
            "total_questions": self.num_questions
        }
        
        analysis_results = {
            "overall_stats": overall_stats,
            "hop_analysis": hop_stats,
            "model_abilities": model_abilities,
            "component_abilities": component_abilities,
        }
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
    
    def plot_results(self, params: Dict[str, np.array], save_path: Optional[str] = None, title_prefix: str = ""):
        """Enhanced plot_results with new visualization functions"""
        theta_range = np.linspace(-3, 3, 100)
        
        plt.style.use('seaborn-v0_8-poster')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Item characteristic curves
        for i in range(self.num_questions):
            p = self.irt_3pl(theta=theta_range, a=params['discrimination'][i], b=params['difficulty'][i])
            ax1.plot(theta_range, p, alpha=0.3, color='gray')
        
        # Use new plot_model_abilities for enhanced visualization
        self.plot_model_abilities(ax1, theta_range, params, self.model_colors)
        ax1.set_title(f'{title_prefix}Item Characteristic Curves\nProbability of Correct Response vs. Ability')
        ax1.set_xlabel('Model Ability (θ)')
        ax1.set_ylabel('Probability of Correct Response')
        
        # Information curves
        info = self.compute_information(params, theta_range)
        for i in range(self.num_questions):
            ax2.plot(theta_range, info[i], alpha=0.3, color='gray')
        
        # Use plot_model_abilities again for information curves
        self.plot_model_abilities(ax2, theta_range, params, self.model_colors)
        ax2.set_title(f'{title_prefix}Item Information Curves\nMeasurement Precision Across Ability Levels')
        ax2.set_xlabel('Model Ability (θ)')
        ax2.set_ylabel('Information (Measurement Precision)')
        
        # Average information by hop count
        self.plot_hop_information(ax3, info, theta_range)
        
        # Use new plot_component_abilities for enhanced visualization
        self.plot_component_abilities(ax4, params)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_model_abilities(self, ax, theta_range, params, model_colors):
        """Plot model abilities with distinct colors"""
        for i, (result, theta) in enumerate(zip(self.exam_results, params['theta'])):
            label = f"{result.llm_name}"
            if result.retriever_name:
                label += f" ({result.retriever_name})"
            
            key = f"{result.llm_name}_{result.retriever_name}"
            ax.scatter(theta, 0, marker='x', color=model_colors[key], 
                      s=100, label=label)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    def plot_component_abilities(self, ax, params):
        """Plot component abilities with stacked bars"""
        llm_abilities = params['theta_params'][:self.num_llms]
        ret_abilities = params['theta_params'][self.num_llms:]
        
        x = np.arange(self.num_llms)
        width = 0.35
        
        # Create mapping of LLM names to their results
        llm_results = {}
        for result in self.exam_results:
            if result.llm_name not in llm_results:
                llm_results[result.llm_name] = []
            llm_results[result.llm_name].append(result)
        
        # Plot LLM abilities
        llm_bars = ax.bar(x, llm_abilities, width, label='LLM Base Ability',
                        color='skyblue')
        
        # Plot retriever abilities as stacked bars
        if self.num_retrievers > 0:
            for llm_idx, (llm_name, results) in enumerate(llm_results.items()):
                for result in results:
                    if result.retriever_name:
                        ret_idx = self.retriever_map[result.retriever_name]
                        ax.bar(x[llm_idx], ret_abilities[ret_idx], width,
                            bottom=llm_abilities[llm_idx],
                            label=f'Retriever: {result.retriever_name}',
                            color='lightgreen', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels([llm for llm in self.llm_map.keys()], rotation=45)
        ax.set_title('Model Component Abilities')
        ax.set_xlabel('Language Models')
        ax.set_ylabel('Ability Parameter')
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    def plot_hop_information(self, ax, info, theta_range):
        """Plot average information by hop count"""
        unique_hops = np.unique(self.exam_results[0].num_hops)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_hops)))
        
        for hop_num, color in zip(unique_hops, colors):
            hop_indices = [i for i, h in enumerate(self.exam_results[0].num_hops) 
                         if h == hop_num]
            if hop_indices:
                mean_info = info[hop_indices].mean(axis=0)
                ax.plot(theta_range, mean_info, label=f'{hop_num} hops',
                       color=color, linewidth=2)
        
        ax.set_title('Average Information by Hop Count')
        ax.set_xlabel('Model Ability (θ)')
        ax.set_ylabel('Average Information')
        ax.legend(title='Number of Hops')
    
    def compare_hop_difficulties(self, other_model: 'MultihopIRTModel', 
                               params: Dict[str, np.array], 
                               other_params: Dict[str, np.array],
                               save_path: Optional[str] = None):
        """Compare difficulties between two exam sets (e.g., multihop vs single-hop)"""
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Calculate statistics for both models
        def get_hop_stats(model, params):
            unique_hops = np.unique(model.exam_results[0].num_hops)
            stats = {}
            for hops in unique_hops:
                indices = [i for i, h in enumerate(model.exam_results[0].num_hops) if h == hops]
                stats[hops] = {
                    'difficulty': np.mean(params['difficulty'][indices]),
                    'std': np.std(params['difficulty'][indices]),
                    'count': len(indices)
                }
            return stats
        
        stats1 = get_hop_stats(self, params)
        stats2 = get_hop_stats(other_model, other_params)
        
        # Plot difficulties
        x = np.arange(max(len(stats1), len(stats2)))
        width = 0.35
        
        plt.bar(x - width/2, [stats1[h]['difficulty'] for h in stats1], 
                width, label='Exam Set 1', color='skyblue')
        plt.bar(x + width/2, [stats2[h]['difficulty'] for h in stats2], 
                width, label='Exam Set 2', color='lightgreen')
        
        plt.xlabel('Number of Hops')
        plt.ylabel('Average Difficulty')
        plt.title('Comparison of Question Difficulties by Hop Count')
        plt.xticks(x, list(stats1.keys()))
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return statistical comparison
        return stats1, stats2


def run_analysis(filepaths: Dict[str, Dict[str, Union[str, Dict[str, str]]]], 
                combine_exams: bool = False,
                combine_exam_takers: bool = False,
                output_dir: str = "analysis_results"):
    """Run IRT analysis with option to combine exam results"""
    # Load and process exam results
    exam_results = ExamResult.from_json_files(filepaths, combine_exams=combine_exams, combine_exam_takers=combine_exam_takers)
    
    # Initialize and fit the model
    model = MultihopIRTModel(exam_results, num_questions=len(exam_results[0].responses))
    params = model.fit()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save analysis results
    analysis_type = "combined" if combine_exams else "separate"
    model.save_analysis(params, output_dir / f"{analysis_type}_analysis.json")
    
    # Generate plots
    model.plot_results(params, 
                      save_path=output_dir / f"{analysis_type}_analysis_plots.png",
                      title_prefix=f"{analysis_type.capitalize()} Exam Analysis: ")
    
    return model, params


def compare_exam_sets(model1: MultihopIRTModel, model2: MultihopIRTModel,
                     save_path: Optional[str] = None):
    """Compare two exam sets and visualize results"""
    # Fit models separately
    params1 = model1.fit()
    params2 = model2.fit()
    
    # Calculate statistics for each exam set
    def get_exam_stats(model, params):
        difficulties = params['difficulty']
        discriminations = params['discrimination']
        
        # Calculate statistics by hop count
        hop_stats = {}
        unique_hops = np.unique(model.exam_results[0].num_hops)
        
        for hop_num in unique_hops:
            hop_indices = [i for i, h in enumerate(model.exam_results[0].num_hops) 
                         if h == hop_num]
            
            if hop_indices:
                hop_stats[int(hop_num)] = {
                    'mean_difficulty': float(np.mean(difficulties[hop_indices])),
                    'std_difficulty': float(np.std(difficulties[hop_indices])),
                    'mean_discrimination': float(np.mean(discriminations[hop_indices])),
                    'num_questions': len(hop_indices)
                }
        
        return {
            'overall_difficulty': float(np.mean(difficulties)),
            'overall_discrimination': float(np.mean(discriminations)),
            'num_questions': len(difficulties),
            'hop_stats': hop_stats
        }
    
    stats1 = get_exam_stats(model1, params1)
    stats2 = get_exam_stats(model2, params2)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Overall difficulty comparison
    plt.subplot(2, 1, 1)
    labels = ['Exam Set 1', 'Exam Set 2']
    difficulties = [stats1['overall_difficulty'], stats2['overall_difficulty']]
    
    plt.bar(labels, difficulties, color=['skyblue', 'lightgreen'])
    plt.title('Overall Difficulty Comparison')
    plt.ylabel('Mean Difficulty Parameter')
    
    # Add value labels on bars
    for i, v in enumerate(difficulties):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 2: Difficulty by hop count
    plt.subplot(2, 1, 2)
    
    # Get all unique hop counts
    all_hops = sorted(set(list(stats1['hop_stats'].keys()) + 
                         list(stats2['hop_stats'].keys())))
    
    x = np.arange(len(all_hops))
    width = 0.35
    
    # Plot bars for each exam set
    difficulties1 = [stats1['hop_stats'].get(h, {'mean_difficulty': 0})['mean_difficulty'] 
                    for h in all_hops]
    difficulties2 = [stats2['hop_stats'].get(h, {'mean_difficulty': 0})['mean_difficulty'] 
                    for h in all_hops]
    
    plt.bar(x - width/2, difficulties1, width, label='Exam Set 1', color='skyblue')
    plt.bar(x + width/2, difficulties2, width, label='Exam Set 2', color='lightgreen')
    
    plt.xlabel('Number of Hops')
    plt.ylabel('Mean Difficulty Parameter')
    plt.title('Difficulty Comparison by Hop Count')
    plt.xticks(x, all_hops)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'exam_set1': stats1,
        'exam_set2': stats2,
        'comparison_summary': {
            'difficulty_difference': stats1['overall_difficulty'] - stats2['overall_difficulty'],
            'discrimination_difference': stats1['overall_discrimination'] - stats2['overall_discrimination'],
            'question_count_difference': stats1['num_questions'] - stats2['num_questions']
        }
    }


if __name__ == "__main__":
    # Configuration
    task_domains = ["gov_report", "hotpotqa", "multifieldqa_en", "SecFilings", "wiki"]
    exam_generators = ["llama_3_2_3b", "gemma2_9b", "ministral_8b"]
    
    # Analysis configurations
    analysis_modes = [
        {"combine_exams": True, "combine_exam_takers": False, "output_suffix": "combined_takers"}
        # {"combine_exams": False, "combine_exam_takers": False, "output_suffix": "combined_takers"}
        # {"combine_exams": True, "combine_exam_takers": True, "output_suffix": "combined_takers"} - Don't use
    ]
    
    for task_domain in task_domains:
        print(f"\nProcessing task domain: {task_domain}")
        
        for exam_generator in exam_generators:
            print(f"Processing exam generator: {exam_generator}")
            
            # Define filepaths structure
            filepaths = {
                'llama3-8b': {
                    'closed_book': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_closed_{exam_generator}_single_hop_exam_processed.json.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_closed_exam_new_{exam_generator}_processed_v3.json.json'
                    },
                    'DenseV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'DenseV4': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_{exam_generator}_single_hop_exam_processed_v4.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_exam_new_{exam_generator}_processed_v4.json_5_results.json'
                    },
                    'SparseV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Sparse_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Sparse_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'HybridV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Hybrid_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Hybrid_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'RerankV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Rerank_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Rerank_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'DenseV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'SparseV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Sparse_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Sparse_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'HybridV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Hybrid_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Hybrid_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'RerankV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Rerank_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Rerank_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'DenseV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Dense_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'SparseV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Sparse_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Sparse_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'HybridV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Hybrid_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Hybrid_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'RerankV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Rerank_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_Rerank_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'open_book': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_open_{exam_generator}_single_hop_exam_processed.json.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/llama_3_1_8b_open_exam_new_{exam_generator}_processed_v3.json.json'
                    },
                },
                'mistral-8b': {
                    'closed_book': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_closed_{exam_generator}_single_hop_exam_processed.json.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_closed_exam_new_{exam_generator}_processed_v3.json.json'
                    },
                    'DenseV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'DenseV4': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_{exam_generator}_single_hop_exam_processed_v4.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_exam_new_{exam_generator}_processed_v4.json_5_results.json'
                    },
                    'SparseV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Sparse_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Sparse_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'HybridV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Hybrid_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Hybrid_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'RerankV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Rerank_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Rerank_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'DenseV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'SparseV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Sparse_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Sparse_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'HybridV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Hybrid_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Hybrid_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'RerankV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Rerank_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Rerank_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'DenseV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Dense_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'SparseV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Sparse_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Sparse_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'HybridV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Hybrid_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Hybrid_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'RerankV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Rerank_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_Rerank_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'open_book': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_open_{exam_generator}_single_hop_exam_processed.json.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/ministral_8b_open_exam_new_{exam_generator}_processed_v3.json.json'
                    },
                },
                'gemma2_9b': {
                    'closed_book': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_closed_{exam_generator}_single_hop_exam_processed.json.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_closed_exam_new_{exam_generator}_processed_v3.json.json'
                    },
                    'DenseV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'DenseV4': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_{exam_generator}_single_hop_exam_processed_v4.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_exam_new_{exam_generator}_processed_v4.json_5_results.json'
                    },
                    'SparseV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Sparse_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Sparse_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'HybridV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Hybrid_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Hybrid_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'RerankV3': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Rerank_{exam_generator}_single_hop_exam_processed.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Rerank_exam_new_{exam_generator}_processed_v3.json_5_results.json'
                    },
                    'DenseV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'SparseV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Sparse_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Sparse_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'HybridV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Hybrid_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Hybrid_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'RerankV5-5': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Rerank_{exam_generator}_single_hop_exam_processed_v5.json_5_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Rerank_exam_new_{exam_generator}_processed_v5.json_5_results.json'
                    },
                    'DenseV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Dense_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'SparseV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Sparse_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Sparse_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'HybridV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Hybrid_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Hybrid_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'RerankV5-10': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Rerank_{exam_generator}_single_hop_exam_processed_v5.json_10_results.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_Rerank_exam_new_{exam_generator}_processed_v5.json_10_results.json'
                    },
                    'open_book': {
                        'single': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_open_{exam_generator}_single_hop_exam_processed.json.json',
                        'multi': f'auto-rag-eval/MultiHopData/{task_domain}/exam_results/gemma2_9b_open_exam_new_{exam_generator}_processed_v3.json.json'
                    },
                }
            }
            
            # Run analysis for each configuration
            for analysis_config in analysis_modes:
                output_dir = f"analysis_results/{task_domain}/{exam_generator}/{analysis_config['output_suffix']}"
                print(f"\nRunning analysis with configuration:")
                print(f"- Combine exams: {analysis_config['combine_exams']}")
                print(f"- Combine exam takers: {analysis_config['combine_exam_takers']}")
                print(f"- Output directory: {output_dir}")
                
                try:
                    model, params = run_analysis(
                        filepaths=filepaths,
                        combine_exams=analysis_config['combine_exams'],
                        combine_exam_takers=analysis_config['combine_exam_takers'],
                        output_dir=output_dir
                    )
                    print(f"Analysis completed successfully")
                except Exception as e:
                    print(f"Error during analysis: {str(e)}")
                    continue
