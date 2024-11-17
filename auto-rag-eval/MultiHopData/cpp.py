from LLMServer.llama.llama_instant import ModelFactory, ModelType


    
def main():
    prompt = """Answer this question:
    
    Question: How do the payment methods for NAS treatment services in hospitals and non-hospital settings differ, and what factors influence the coverage of these services in non-hospital settings?\n\nA) Hospitals receive a single bundled payment for NAS treatment services, while non-hospital settings receive separate payments for each service. The coverage of services in non-hospital settings depends on state-specific mechanisms and the infant's foster care status.\n\nB) Non-hospital settings always receive lower reimbursements than hospitals for NAS treatment services, and the coverage of services in non-hospital settings is determined by the facility's compliance with Medicaid standards.\n\nC) Hospitals receive a single bundled payment for NAS treatment services, while non-hospital settings receive separate payments for each service. The coverage of services in non-hospital settings is not influenced by the infant's foster care status or the facility's compliance with Medicaid standards.\n\nD) Non-hospital settings receive a single bundled payment for NAS treatment services, while hospitals receive separate payments for each service. The coverage of services in non-hospital settings depends on the facility's compliance with Medicaid standards, state-specific mechanisms, and the infant's foster care status.
    
    """
    # Create Mistral model
    mistral_model = ModelFactory.create_model(ModelType.MISTRAL_7B)
    response = mistral_model.invoke(prompt)
    print(f"Mistral response: {response}\n")

    mixtral_model = ModelFactory.create_model(ModelType.MIXTRAL_8_7B)
    
    print(mixtral_model.invoke(prompt))
    
    # mixtral22_model = ModelFactory.create_model(ModelType.MIXTRAL_8_22B)
    
    # print(mixtral22_model.invoke("What are the advantages of Apple Silicon?"))


if __name__ == "__main__":
    main()