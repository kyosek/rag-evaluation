from LLMServer.llama.llama_instant import ModelFactory, ModelType


    
def main():
    prompt = """Hi, how are you doing?
    """
    # Create Mistral model
    # mistral_model = ModelFactory.create_model(ModelType.MISTRAL_7B)
    # response = mistral_model.invoke("What are the advantages of Apple Silicon?")
    # print(f"Mistral response: {response}\n")

    # mixtral_model = ModelFactory.create_model(ModelType.MIXTRAL_8_7B)
    
    # print(mixtral_model.invoke(prompt))
    
    mixtral22_model = ModelFactory.create_model(ModelType.MIXTRAL_8_22B)
    
    print(mixtral22_model.invoke("What are the advantages of Apple Silicon?"))


if __name__ == "__main__":
    main()