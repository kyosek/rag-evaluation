import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import json
from scipy.stats import norm
import matplotlib.pyplot as plt

@dataclass
class ExamResult:
    """Class to store exam results for a model configuration"""
    llm_name: str
    retriever_name: Optional[str] = None
    responses: Optional[List[bool]] = None
    num_hops: Optional[List[int]] = None
    
    @classmethod
    def from_json_file(cls, filepath: str, llm_name: str, retriever_name: Optional[str] = None) -> 'ExamResult':
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
    
    # def save_analysis(self, params: Dict[str, np.array], output_path: str):
    #     """Save analysis results to a JSON file"""
    #     # Get unique hop counts
    #     unique_hops = sorted(list(set(hop for result in self.exam_results for hop in result.num_hops)))
        
    #     # Calculate average difficulty and discrimination by hop count
    #     hop_stats = {}
    #     for hop_count in unique_hops:
    #         # Get indices for questions with this hop count
    #         indices = [i for i, hops in enumerate(self.exam_results[0].num_hops) if hops == hop_count]
            
    #         hop_stats[str(hop_count)] = {
    #             "avg_difficulty": float(np.mean(params['difficulty'][indices])),
    #             "avg_discrimination": float(np.mean(params['discrimination'][indices])),
    #             "num_questions": len(indices)
    #         }
        
    #     # Prepare model ability results
    #     model_abilities = {}
    #     for result in self.exam_results:
    #         model_key = f"{result.llm_name}"
    #         if result.retriever_name:
    #             model_key += f"_{result.retriever_name}"
            
    #         idx = self.exam_results.index(result)
    #         model_abilities[model_key] = float(params['theta'][idx])
        
    #     # Prepare component abilities
    #     component_abilities = {
    #         "llms": {
    #             llm: float(params['theta_params'][idx])
    #             for llm, idx in self.llm_map.items()
    #         },
    #         "retrievers": {
    #             ret: float(params['theta_params'][self.num_llms + idx])
    #             for ret, idx in self.retriever_map.items()
    #         } if self.retriever_map else {}
    #     }
        
    #     # Overall exam statistics
    #     overall_stats = {
    #         "avg_difficulty": float(np.mean(params['difficulty'])),
    #         "avg_discrimination": float(np.mean(params['discrimination'])),
    #         "total_questions": self.num_questions
    #     }
        
    #     analysis_results = {
    #         "overall_stats": overall_stats,
    #         "hop_analysis": hop_stats,
    #         "model_abilities": model_abilities,
    #         "component_abilities": component_abilities,
    #     }
        
    #     with open(output_path, 'w') as f:
    #         json.dump(analysis_results, f, indent=2)
            
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
                    num_hops.append(2)  # Default to 2 hops if not specified
            
            return cls(
                llm_name=llm_name,
                retriever_name=retriever_name,
                responses=responses,
                num_hops=num_hops
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {filepath}")
        except Exception as e:
            raise ValueError(f"Error processing {filepath}: {str(e)}")
    
    @classmethod
    def from_json_files(cls, filepaths: Dict[str, Dict[str, str]]) -> List['ExamResult']:
        """Create multiple ExamResult instances from a dictionary mapping model configurations to file paths"""
        results = []
        
        for llm_name, retriever_paths in filepaths.items():
            for retriever_name, filepath in retriever_paths.items():
                try:
                    # Use None for closed_book to indicate no retriever
                    ret_name = None if retriever_name == 'closed_book' else retriever_name
                    result = cls.from_json_file(filepath, llm_name, ret_name)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Failed to load {filepath} for {llm_name}/{retriever_name}: {e}")
                    continue
        
        if not results:
            raise ValueError("No valid exam results could be loaded from the provided files")
        
        # Verify all results have the same number of questions
        num_questions = len(results[0].responses)
        if not all(len(result.responses) == num_questions for result in results):
            raise ValueError("All exam results must have the same number of questions")
        
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
    
    def plot_results(self, params: Dict[str, np.array], save_path: Optional[str] = None):
        """Plot IRT analysis results"""
        theta_range = np.linspace(-3, 3, 100)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Item characteristic curves
        for i in range(self.num_questions):
            p = self.irt_3pl(
                theta=theta_range, 
                a=params['discrimination'][i], 
                b=params['difficulty'][i]
            )
            ax1.plot(theta_range, p, alpha=0.3)
            
        # Mark model abilities
        for i, theta in enumerate(params['theta']):
            ax1.scatter(theta, 0, marker='x', color='red')
            
        ax1.set_title('Item Characteristic Curves')
        ax1.set_xlabel('Ability (θ)')
        ax1.set_ylabel('P(correct)')
        
        # Information curves
        info = self.compute_information(params, theta_range)
        for i in range(self.num_questions):
            ax2.plot(theta_range, info[i], alpha=0.3)
            
        # Mark model abilities
        for theta in params['theta']:
            ax2.scatter(theta, 0, marker='x', color='red')
            
        ax2.set_title('Item Information Curves')
        ax2.set_xlabel('Ability (θ)')
        ax2.set_ylabel('Information')
        
        # Average information by number of hops
        unique_hops = np.unique(self.exam_results[0].num_hops)
        for hops in unique_hops:
            # Get indices for questions with this hop count
            hop_indices = [i for i, h in enumerate(self.exam_results[0].num_hops) if h == hops]
            if hop_indices:
                mean_info = info[hop_indices].mean(axis=0)
                ax3.plot(theta_range, mean_info, label=f'{hops} hops')
            
        ax3.set_title('Average Information by Hop Count')
        ax3.set_xlabel('Ability (θ)')
        ax3.set_ylabel('Information')
        ax3.legend()
        
        # Ability components
        llm_abilities = params['theta_params'][:self.num_llms]
        ret_abilities = params['theta_params'][self.num_llms:]
        
        ax4.bar(range(self.num_llms), llm_abilities, label='LLM')
        if self.num_retrievers > 0:
            ax4.bar(
                range(self.num_retrievers), 
                ret_abilities,
                label='Retriever',
                alpha=0.5
            )
            
        ax4.set_title('Model Component Abilities')
        ax4.set_xlabel('Model Index')
        ax4.set_ylabel('Ability Parameter')
        ax4.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    # Load a single exam result
    # single_result = ExamResult.from_json_file(
    # filepath='auto-rag-eval/MultiHopData/gov_report/exam_results/llama_3_1_8b_closed_exam_new_llama_3_2_3b_processed_v2.json.json',
    # llm_name='llama3-8b',
    # retriever_name='closed_book'
# )

# Load multiple exam results
    filepaths = {
        'llama3-8b': {
            'closed_book': 'auto-rag-eval/MultiHopData/gov_report/exam_results/llama_3_1_8b_closed_exam_new_llama_3_2_3b_processed_v2.json.json',
            'open_book': 'auto-rag-eval/MultiHopData/gov_report/exam_results/llama_3_1_8b_open_exam_new_llama_3_2_3b_processed_v2.json.json'
        },
        'mistral-8b': {
            'closed_book': 'auto-rag-eval/MultiHopData/gov_report/exam_results/ministral-8b_closed_exam_new_llama_3_2_3b_processed_v2.json.json',
            'open_book': 'auto-rag-eval/MultiHopData/gov_report/exam_results/ministral-8b_open_exam_new_llama_3_2_3b_processed_v2.json.json'
        },
        'gemma2-27b': {
            'closed_book': 'auto-rag-eval/MultiHopData/gov_report/exam_results/gemma2-27b_closed_exam_new_llama_3_2_3b_processed_v2.json.json',
            'open_book': 'auto-rag-eval/MultiHopData/gov_report/exam_results/gemma2-27b_open_exam_new_llama_3_2_3b_processed_v2.json.json'
        }
    }

    exam_results = ExamResult.from_json_files(filepaths)

    # Initialize and fit the model with loaded results
    model = MultihopIRTModel(exam_results, num_questions=len(exam_results[0].responses))
    # model = MultihopIRTModel(single_result, num_questions=len(single_result.responses))
    params = model.fit()
    
    # Plot results
    model.plot_results(params, save_path="irt_analysis.png")
