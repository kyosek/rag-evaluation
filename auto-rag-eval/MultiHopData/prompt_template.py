from typing import List, Dict

from LLMServer.llama.llama_instant import ModelType


class PromptTemplate:
    @staticmethod
    def get_question_generation_prompt_template(model_type: ModelType, chunks: List[Dict[str, str]], task_domain, documentation) -> str:
        """Return the appropriate prompt template based on model type."""
        num_chunks = len(chunks)
        prompts = {
            ModelType.LLAMA_3_2_3B: f"""
            <<SYS>>
            You are an expert exam question generator specialising in creating challenging multihop multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.
            
            Core requirements:
            1. Question MUST require synthesising information from at least {num_chunks} different chunks
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing all chunks
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question
            
            Question Design Principles:
            1. Incorporate subtle dependencies between chunks
            2. Require careful analysis of conditional statements
            3. Include scenarios where surface-level reading might lead to wrong conclusions
            4. Design distractors that would be chosen if key information from certain chunks is missed
            
            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}

            Generate a question following this format:
            Question: [Complex question requiring multi-hop reasoning]
            A) [Option incorporating some but not all key information]
            B) [Option based on common misinterpretation]
            C) [Option that would be correct if one crucial detail is missed]
            D) [Correct option requiring synthesis of all chunks]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            """,
            ModelType.MINISTRAL_8B: f"""
            <s>[INST]
            You are an expert exam question generator specialising in creating challenging multihop multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.
            
            Core requirements:
            1. Question MUST require synthesising information from at least {num_chunks} different chunks
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing all chunks
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question
            
            Question Design Principles:
            1. Incorporate subtle dependencies between chunks
            2. Require careful analysis of conditional statements
            3. Include scenarios where surface-level reading might lead to wrong conclusions
            4. Design distractors that would be chosen if key information from certain chunks is missed
            
            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            Domain: {task_domain}
            Documentation: {documentation}

            Generate a question following this format:
            Question: [Complex question requiring multi-hop reasoning]
            A) [Option incorporating some but not all key information]
            B) [Option based on common misinterpretation]
            C) [Option that would be correct if one crucial detail is missed]
            D) [Correct option requiring synthesis of all chunks]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            [/INST]</s>
            """,
            ModelType.LLAMA_3_1_8B: f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert exam question generator specialising in creating challenging multihop multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.

            **Core requirements:**
            1. Question MUST require synthesizing information from at least {num_chunks} different chunks
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing all chunks
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question

            **Question Design Principles:**
            1. Incorporate subtle dependencies between chunks
            2. Require careful analysis of conditional statements
            3. Include scenarios where surface-level reading might lead to wrong conclusions
            4. Design distractors that would be chosen if key information from certain chunks is missed

            **Format Requirements:**
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            **Domain:** {task_domain}
            **Documentation:** {documentation}

            **Generate a question following this format:**
            Question: [Complex question requiring multi-hop reasoning]
            A) [Option incorporating some but not all key information]
            B) [Option based on common misinterpretation]
            C) [Option that would be correct if one crucial detail is missed]
            D) [Correct option requiring synthesis of all chunks]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            Reasoning Steps: [Step-by-step breakdown of how to arrive at the correct answer]
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generate a question.
            <|eot_id|>
            """,
            ModelType.GEMMA2_9BB: f"""
            <start_of_turn>user
            You are an expert exam question generator specialising in creating challenging multihop multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.

            Core requirements:
            1. Question MUST require synthesising information from at least {num_chunks} different chunks
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing all chunks
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question

            Question Design Principles:
            1. Incorporate subtle dependencies between chunks
            2. Require careful analysis of conditional statements
            3. Include scenarios where surface-level reading might lead to wrong conclusions
            4. Design distractors that would be chosen if key information from certain chunks is missed

            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            Domain: {task_domain}
            Documentation: {documentation}

            Generate a question following this format:
            Question: [Complex question requiring multi-hop reasoning]
            A) [Option incorporating some but not all key information]
            B) [Option based on common misinterpretation]
            C) [Option that would be correct if one crucial detail is missed]
            D) [Correct option requiring synthesis of all chunks]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]<end_of_turn>
            <start_of_turn>model
            """,
    }
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])
    
    @staticmethod
    def get_single_hop_question_generation_prompt_template(model_type: ModelType, chunks: List[Dict[str, str]], task_domain, documentation) -> str:
        prompts = {
            ModelType.LLAMA_3_2_3B: f"""
            <<SYS>>
            You are an expert exam question generator specialising in creating challenging multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.

            **Core requirements:**
            1. Question MUST require synthesizing information from the given chunk
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing the given chunk
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question

            **Question Design Principles:**
            1. Require careful analysis of conditional statements
            2. Include scenarios where surface-level reading might lead to wrong conclusions
            3. Design distractors that would be chosen if key information from certain part of information is missed
            
            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}

            Generate a question following this format:
            Question: [Complex question requiring deep reasoning]
            A) [Option incorporating some but not all key information]
            B) [Option based on common misinterpretation]
            C) [Option that would be correct if one crucial detail is missed]
            D) [Correct option requiring synthesis of all chunks]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            """,
            ModelType.LLAMA_3_1_8B: f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert exam question generator specialising in creating challenging multiple-choice questions (1 correct answer and 3 distractors) that require complex reasoning across multiple pieces of information.

            **Core requirements:**
            1. Question MUST require synthesizing information from the given chunk
            2. Distractors must be highly plausible and based on common misconceptions or partial understanding
            3. The correct answer should not be obvious without carefully analysing the given chunk
            4. Each distractor should represent a different type of reasoning error
            5. As students do not have an access to the chunk information, do not mention chunks in the question

            **Question Design Principles:**
            1. Require careful analysis of conditional statements
            2. Include scenarios where surface-level reading might lead to wrong conclusions
            3. Design distractors that would be chosen if key information from certain part of information is missed

            **Format Requirements:**
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            **Domain:** {task_domain}
            **Documentation:** {documentation}

            **Generate a question following this format:**
            Generate a question following this format:
            Question: [Complex question requiring deep reasoning]
            A) [Option incorporating some but not all key information]
            B) [Option based on common misinterpretation]
            C) [Option that would be correct if one crucial detail is missed]
            D) [Correct option requiring synthesis of all chunks]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generate a question.
            <|eot_id|>
            """,
    }
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])

# return f"""
    # <<SYS>>
    # You are an expert exam question generator specializing in creating challenging multiple-choice questions that require complex reasoning across multiple pieces of information.
    
    # Required reasoning type: {reasoning_type}
    # Identified relationships between chunks: {relationships}
    
    # Core requirements:
    # 1. Question MUST require synthesizing information from at least {len(chunks)} different chunks
    # 2. Distractors must be highly plausible and based on common misconceptions or partial understanding
    # 3. The correct answer should not be obvious without carefully analyzing all chunks
    # 4. Each distractor should represent a different type of reasoning error
    
    # Question Design Principles:
    # 1. Incorporate subtle dependencies between chunks
    # 2. Require careful analysis of conditional statements
    # 3. Include scenarios where surface-level reading might lead to wrong conclusions
    # 4. Design distractors that would be chosen if key information from certain chunks is missed
    
    # Format Requirements:
    # - Question text should be clear but complex
    # - Each option must start with A), B), C), or D)
    # <</SYS>>

    # Domain: {task_domain}
    # Documentation: {documentation}

    # Generate a question following this format:
    # Question: [Complex question requiring multi-hop reasoning]
    # A) [Option incorporating some but not all key information]
    # B) [Option based on common misinterpretation]
    # C) [Option that would be correct if one crucial detail is missed]
    # D) [Correct option requiring synthesis of all chunks]
    # Correct Answer: [Letter one of "A", "B", "C" or "D"]
    # Reasoning Steps: [Step-by-step breakdown of how to arrive at the correct answer]
    # """