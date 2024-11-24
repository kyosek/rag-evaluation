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
            You are an expert exam question generator specializing in creating challenging multi-hop multiple-choice questions that require complex reasoning across multiple pieces of information.
            Your questions should test students' ability to synthesize information rather than rely on surface-level understanding.
            
            Core Requirements:
            - Multi-Hop Integration
            - Question MUST require synthesising information from all the chunks
            - The synthesis should involve logical relationships, temporal dependencies, or combining disparate insights
            - No single chunk should be sufficient to answer the question

            Question Design
            - Questions must be challenging but fair
            - Avoid obvious wording that makes the answer apparent without analysis
            - Include subtle dependencies and contradictions across chunks
            - Questions should be solvable using only the provided information
            - Do not reference chunks in the question text, as students won't have access to this information

            Distractor Design
            Create three highly plausible distractors, each representing a specific type of reasoning error:
            - Partial Understanding Distractor: Uses some but not all key information from chunks
            - Misinterpretation Distractor: Based on common misunderstanding or overgeneralization
            - Critical Detail Distractor: Would be correct if one crucial dependency between chunks is missed

            Answer Requirements
            - Correct answer must require careful synthesis of ALL chunks
            - Each distractor should exploit different common reasoning pitfalls
            - Surface-level reading should make distractors appear plausible

            Format Requirements:
            Question: Clear but complex question text that demands multi-chunk synthesis
            Options:
            - Each option must start with A), B), C), or D)
            - One correct answer integrating all chunk information
            - Three distractors following the design principles above
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            
            You do not need to generate anything else other
            
            Format example:
            Question: [Question]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}
            """,
            ModelType.MINISTRAL_8B: f"""
            <s>[INST]
            You are an expert exam question generator specializing in creating challenging multi-hop multiple-choice questions that require complex reasoning across multiple pieces of information.
            Your questions should test students' ability to synthesize information rather than rely on surface-level understanding.
            
            Core Requirements:
            - Multi-Hop Integration
            - Question MUST require synthesising information from all the chunks
            - The synthesis should involve logical relationships, temporal dependencies, or combining disparate insights
            - No single chunk should be sufficient to answer the question

            Question Design
            - Questions must be challenging but fair
            - Avoid obvious wording that makes the answer apparent without analysis
            - Include subtle dependencies and contradictions across chunks
            - Questions should be solvable using only the provided information
            - Do not reference chunks in the question text, as students won't have access to this information

            Distractor Design
            Create three highly plausible distractors, each representing a specific type of reasoning error:
            - Partial Understanding Distractor: Uses some but not all key information from chunks
            - Misinterpretation Distractor: Based on common misunderstanding or overgeneralization
            - Critical Detail Distractor: Would be correct if one crucial dependency between chunks is missed

            Answer Requirements
            - Correct answer must require careful synthesis of ALL chunks
            - Each distractor should exploit different common reasoning pitfalls
            - Surface-level reading should make distractors appear plausible

            Format Requirements:
            Question: Clear but complex question text that demands multi-chunk synthesis
            Options:
            - Each option must start with A), B), C), or D)
            - One correct answer integrating all chunk information
            - Three distractors following the design principles above
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            
            You do not need to generate anything else other
            
            Format example:
            Question: [Question]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}
            [/INST]</s>
            """,
            ModelType.LLAMA_3_1_8B: f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert exam question generator specializing in creating challenging multi-hop multiple-choice questions that require complex reasoning across multiple pieces of information.
            Your questions should test students' ability to synthesize information rather than rely on surface-level understanding.
            
            Core Requirements:
            - Multi-Hop Integration
            - Question MUST require synthesising information from all the chunks
            - The synthesis should involve logical relationships, temporal dependencies, or combining disparate insights
            - No single chunk should be sufficient to answer the question

            Question Design
            - Questions must be challenging but fair
            - Avoid obvious wording that makes the answer apparent without analysis
            - Include subtle dependencies and contradictions across chunks
            - Questions should be solvable using only the provided information
            - Do not reference chunks in the question text, as students won't have access to this information

            Distractor Design
            Create three highly plausible distractors, each representing a specific type of reasoning error:
            - Partial Understanding Distractor: Uses some but not all key information from chunks
            - Misinterpretation Distractor: Based on common misunderstanding or overgeneralization
            - Critical Detail Distractor: Would be correct if one crucial dependency between chunks is missed

            Answer Requirements
            - Correct answer must require careful synthesis of ALL chunks
            - Each distractor should exploit different common reasoning pitfalls
            - Surface-level reading should make distractors appear plausible

            Format Requirements:
            Question: Clear but complex question text that demands multi-chunk synthesis
            Options:
            - Each option must start with A), B), C), or D)
            - One correct answer integrating all chunk information
            - Three distractors following the design principles above
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            
            You do not need to generate anything else other
            
            Format example:
            Question: [Question]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}

            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generate a question.
            <|eot_id|>
            """,
            ModelType.GEMMA2_9B: f"""
            <start_of_turn>user
            You are an expert exam question generator specializing in creating challenging multi-hop multiple-choice questions that require complex reasoning across multiple pieces of information.
            Your questions should test students' ability to synthesize information rather than rely on surface-level understanding.
            
            Core Requirements:
            - Multi-Hop Integration
            - Question MUST require synthesising information from all the chunks
            - The synthesis should involve logical relationships, temporal dependencies, or combining disparate insights
            - No single chunk should be sufficient to answer the question

            Question Design
            - Questions must be challenging but fair
            - Avoid obvious wording that makes the answer apparent without analysis
            - Include subtle dependencies and contradictions across chunks
            - Questions should be solvable using only the provided information
            - Do not reference chunks in the question text, as students won't have access to this information

            Distractor Design
            Create three highly plausible distractors, each representing a specific type of reasoning error:
            - Partial Understanding Distractor: Uses some but not all key information from chunks
            - Misinterpretation Distractor: Based on common misunderstanding or overgeneralization
            - Critical Detail Distractor: Would be correct if one crucial dependency between chunks is missed

            Answer Requirements
            - Correct answer must require careful synthesis of ALL chunks
            - Each distractor should exploit different common reasoning pitfalls
            - Surface-level reading should make distractors appear plausible

            Format Requirements:
            Question: Clear but complex question text that demands multi-chunk synthesis
            Options:
            - Each option must start with A), B), C), or D)
            - One correct answer integrating all chunk information
            - Three distractors following the design principles above
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            
            You do not need to generate anything else other
            
            Format example:
            Question: [Question]
            A) [Option A]
            B) [Option B]
            C) [Option C]
            D) [Option D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}
            <end_of_turn>
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
            3. Design distractors that would be chosen if key information from certain chunks is missed
                3.1. One distractor should incorporate some but not all key information
                3.2. One distractor should be based on common misinterpretation
                3.3. One distractor would be correct if one crucial detail is missed
            4. Correct option requires synthesis of all key information
            
            Format Requirements:
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)
            <</SYS>>

            Domain: {task_domain}
            Documentation: {documentation}

            Generate a question following this format:
            Question: <Question>
            A) [Choice A]
            B) [Choice B]
            C) [Choice C]
            D) [Choice D]
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
            3. Design distractors that would be chosen if key information from certain chunks is missed
                3.1. One distractor should incorporate some but not all key information
                3.2. One distractor should be based on common misinterpretation
                3.3. One distractor would be correct if one crucial detail is missed
            4. Correct option requires synthesis of all key information

            **Format Requirements:**
            - Question text should be clear but complex
            - Each option must start with A), B), C), or D)

            **Domain:** {task_domain}
            **Documentation:** {documentation}

            **Generate a question following this format:**
            Question: <Question>
            A) [Choice A]
            B) [Choice B]
            C) [Choice C]
            D) [Choice D]
            Correct Answer: [Letter one of "A", "B", "C" or "D"]
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Generate a question.
            <|eot_id|>
            """,
    }
        return prompts.get(model_type, prompts[ModelType.LLAMA_3_2_3B])
    
    @staticmethod
    def get_verification_prompt(question_data: dict, chunks: List[Dict[str, str]]) -> str:
        """Format the verification prompt with the question data and chunks."""
        # Extract options from choices
        options = [choice.split(') ')[1] for choice in question_data['choices']]
        
        # Format chunk text
        chunk_text = '\n'.join([f"Chunk{i+1}: {chunk['text']}" 
                               for i, chunk in enumerate(chunks)])
        
        return PromptTemplate.verification_prompt_template(question_data, options, chunk_text)

    @staticmethod
    def verification_prompt_template(question_data: dict, options: List[str], chunk_text: List[str]):
        question=question_data['question'],
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
        correct_answer=question_data['correct_answer'],
        chunk_text=chunk_text
            
        prompt = f"""<s>[INST]  
            You are an expert exam question verifier.     
            Analyse the following question-options-answer triplet.          
            
            Task:
            1. Examine if question and choices make sense compared to the given documents
            2. Examine if the "correct_answer" actually correct
            3. Examine if all the chunks are necessary to answer the question correctly
            4. Give a feedback to improve the exam quality
            
            Output instruction:
            The output must follow the output format with the following entities and do not need to add anything else:
            - required_chunks: List of chunk numbers needed to answer, e.g., [1, 3] means chunks 1 and 3 are needed
            - synthesis_feedback: Explanation of how the information needs to be combined
            - quality_feedback: Feedback to improve the exam quality and difficulty
            - confidence: 1-5 scale of confidence in this assessment
            
            Output format:     
            {{
                "required_chunks": [int],
                "synthesis_feedback": string,
                "quality_feedback": string,
                "confidence": int
            }}
            [/INST]
            
            Question: {question}
            Options:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            Correct Answer: {correct_answer}
            Documents: {chunk_text}
            </s>
            """
        return prompt

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