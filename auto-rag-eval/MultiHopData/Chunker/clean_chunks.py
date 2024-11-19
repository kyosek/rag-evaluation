import json
from collections import defaultdict


def deduplicate_json(input_file, output_file):
    """
    Detect, count and remove duplicate content entries from JSON file.
    Keeps the first occurrence of each duplicate.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    
    Returns:
        tuple: (total_duplicates, duplicate_stats)
    """
    # Read JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Track duplicates
    content_seen = {}  # Store first occurrence of each content
    duplicate_count = 0
    duplicate_stats = defaultdict(int)
    
    # Process each document
    for doc in data:
        # Check document level content
        doc_content = doc['content']
        if doc_content in content_seen:
            duplicate_count += 1
            duplicate_stats['document_level'] += 1
            doc['content'] = content_seen[doc_content]
        else:
            content_seen[doc_content] = doc_content
        
        # Process chunks
        if 'chunks' in doc:
            seen_chunks = {}
            filtered_chunks = []
            
            for chunk in doc['chunks']:
                chunk_content = chunk['content']
                if chunk_content in seen_chunks:
                    duplicate_count += 1
                    duplicate_stats['chunk_level'] += 1
                    continue
                else:
                    seen_chunks[chunk_content] = True
                    filtered_chunks.append(chunk)
            
            # Update chunks with deduplicated list
            doc['chunks'] = filtered_chunks
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return duplicate_count, dict(duplicate_stats)


def main(input_file, output_file):
    
    try:
        total_dupes, stats = deduplicate_json(input_file, output_file)
        
        print(f"Deduplication complete!")
        print(f"Total duplicates removed: {total_dupes}")
        print("\nDuplicate statistics:")
        for level, count in stats.items():
            print(f"- {level}: {count} duplicates")
        print(f"\nCleaned data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    input_file = "auto-rag-eval/MultiHopData/wiki/chunks/docs_chunk_semantic.json"
    output_file = "auto-rag-eval/MultiHopData/wiki/chunks/docs_chunk_semantic_cleaned.json"
    
    main(input_file, output_file)
