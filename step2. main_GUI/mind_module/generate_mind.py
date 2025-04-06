import re
import os
from graphviz import Digraph
from openai import OpenAI # Assuming your OpenAI setup code is here

# ==============================================================================
# 1. Function to Generate Mind Map Text Structure (using LLM)
# ==============================================================================

# Use the generate_response function you provided
# (Assuming 'client' is initialized as in your example)
# Setting up the OpenAI client (as provided in the question)
try:
    # It's generally better practice to load keys from environment variables
    # or a config file rather than hardcoding.
    # For this example, we'll use the provided code.
    client = OpenAI(
        base_url="https://www.dmxapi.com/v1",
        api_key="sk-JWU9jpEGAv2kHo3YBNB1TPIuhVjT4Nnf60GT55n7iz5GY3g7" # Replace with your actual key if needed
    )
    # Simple test call to check connectivity (optional)
    # client.models.list()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None # Ensure client is None if initialization fails


def generate_response(prompt, model="gpt-4o-mini", max_tokens=500):
    """
    Uses the OpenAI API to generate a response.
    (Modified slightly to handle potential client initialization errors)

    Args:
        prompt (str): The input prompt text.
        model (str): The model name to use.
        max_tokens (int): The maximum number of tokens for the response.

    Returns:
        str: The generated response text, or an error message.
    """
    if not client:
        return "Error: OpenAI client not initialized."
    try:
        # Using the Chat Completions endpoint
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"An error occurred during API call: {str(e)}"


def generate_mindmap_text_structure(topic, model="gpt-4o-mini", max_tokens=400):
    """
    Asks the LLM to generate a hierarchical text structure for a mind map
    based on a given topic, using hyphens for levels.

    Args:
        topic (str): The central topic for the mind map.
        model (str): The model name to use for generation.
        max_tokens (int): Max tokens for the LLM response.

    Returns:
        str: A string representing the mind map structure with hyphens,
             or an error message.
    """
    prompt = f"""
Generate a hierarchical mind map structure for the topic: "{topic}".
Use hyphens (-) to indicate levels. The main topic should be at the first level.
Sub-topics should use more hyphens (e.g., '--', '---').
Keep the text for each node concise.

Example format:
   - MainTopic    
         - SubTopicA
               - SubTopicB
         - SubTopicC
               - SubTopicD
                     - SubTopicE
                     - SubTopicF
                           - SubTopicG
                                 - SubTopicH
                           - SubTopicI
         - SubTopicK
               - SubTopicL
               - SubTopicM
               - SubTopicN
                     - SubTopicO
               - SubTopicP

Please provide the structure for the topic: "{topic}"
    """
    response = generate_response(prompt, model=model, max_tokens=max_tokens)
    # Basic check if the response seems like the requested format
    if response and '-' in response.split('\n')[0]:
        return response
    elif response.startswith("An error occurred") or response.startswith("Error:"):
         return response # Return the error message
    else:
        # Attempt to salvage if the format is slightly off but contains hyphens
        lines = response.split('\n')
        structured_lines = [line for line in lines if line.strip().startswith('-')]
        if structured_lines:
            print("Warning: LLM response might not be perfectly formatted, attempting to use lines starting with '-'.")
            return "\n".join(structured_lines)
        else:
            return f"Error: LLM did not return the expected hyphenated structure.\nResponse:\n{response}"


# ==============================================================================
# 2. Function to Parse Text Structure and Generate PNG Mind Map
# ==============================================================================

def visualize_mindmap_from_text(text_structure, output_filename="mindmap", output_format="png", engine="dot"):
    """
    Parses a hyphen-based text structure and generates a mind map image using Graphviz.

    Args:
        text_structure (str): The mind map structure using hyphens for levels.
        output_filename (str): The base name for the output file (without extension).
        output_format (str): The output image format (e.g., 'png', 'svg', 'pdf').
        engine (str): The Graphviz layout engine ('dot', 'neato', 'fdp', etc.).
                      'dot' is usually good for hierarchies.

    Returns:
        str: The full path to the generated image file, or an error message.
    """
    if not text_structure or text_structure.startswith("Error:") or not text_structure.strip():
        return f"Error: Invalid or empty text structure provided.\nInput was: {text_structure}"

    try:
        # Initialize Graphviz Digraph (directed graph)
        # Using 'strict=False' allows multiple edges between same nodes if needed,
        # though unlikely for typical mind maps generated this way.
        dot = Digraph(comment='Mind Map', format=output_format, engine=engine, graph_attr={'rankdir': 'LR'}, strict=False) # LR = Left to Right layout

        # --- Styling (Customize as desired) ---
        dot.attr('node', shape='box', style='filled', fillcolor='lightblue',
               fontname='Microsoft YaHei', # Use a font that supports Chinese if needed, e.g., 'SimHei', 'Microsoft YaHei' on Windows, 'Arial Unicode MS' or others on Mac/Linux
               fontsize='10')
        dot.attr('edge', color='gray', arrowhead='vee', penwidth='1.0')
        dot.attr(ranksep='0.5', nodesep='0.3') # Adjust spacing

        # --- Parsing Logic ---
        lines = text_structure.strip().split('\n')
        node_counter = 0
        level_to_last_node_id = {} # Dictionary to track the parent node at each level {level: node_id}
        parent_stack = [] # Alternative using a stack: [(level, node_id)]

        for line in lines:
            line = line.rstrip() # Remove trailing whitespace
            if not line.strip():
                continue # Skip empty lines

            # Determine level and text
            level = 0
            while level < len(line) and line[level] == '-':
                level += 1

            if level == 0: # Skip lines not starting with '-'
                print(f"Skipping line (no hyphen prefix): {line}")
                continue

            node_text = line[level:].strip()
            if not node_text: # Skip if only hyphens
                continue

            node_id = f"node_{node_counter}"
            node_counter += 1

            # Add the node
            dot.node(node_id, node_text)

            # Find parent and add edge
            # Pop nodes from stack with level >= current level
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            # The top of the stack is the parent
            if parent_stack:
                parent_id = parent_stack[-1][1]
                dot.edge(parent_id, node_id)
            # Else: This is a top-level node (or the first node)

            # Push current node onto the stack
            parent_stack.append((level, node_id))


        # --- Render the graph ---
        # The render function saves the file and returns the filename.
        # It might throw an exception if Graphviz executable is not found.
        output_path = dot.render(output_filename, view=False, cleanup=True) # cleanup=True removes source file
        return output_path

    except ImportError:
        return "Error: The 'graphviz' Python library is not installed. Please install it using 'pip install graphviz'."
    except FileNotFoundError: # Often indicates Graphviz software itself is not installed or not in PATH
         return ("Error: Graphviz executable not found. "
                 "Ensure Graphviz is installed and its 'bin' directory is in the system PATH.")
    except Exception as e:
        return f"An error occurred during mind map generation: {str(e)}"

# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == "__main__":
    # --- Example 1: Generate structure from LLM and visualize ---
    print("\n--- Example 1: Generating structure via LLM ---")
    topic = "What’s the  shoes?"
    print(f"Topic: {topic}")

    mindmap_structure_text = generate_mindmap_text_structure(topic)

    print("\nGenerated Text Structure:")
    print(mindmap_structure_text)

    if mindmap_structure_text and not mindmap_structure_text.startswith("Error:"):
        image_path = visualize_mindmap_from_text(mindmap_structure_text, output_filename="quantum_computing_mindmap")
        print(f"\nMind map image generated: {image_path}")
    else:
        print("\nSkipping visualization due to errors in structure generation.")

    # --- Example 2: Visualize a predefined text structure ---
    print("\n--- Example 2: Visualizing predefined structure ---")
    predefined_structure = """
    - Project Management
    -- Initiation
    --- Define Goals
    --- Identify Stakeholders
    -- Planning
    --- Create WBS (Work Breakdown Structure)
    --- Develop Schedule
    --- Resource Allocation
    -- Execution
    --- Task Management
    --- Team Collaboration
    -- Monitoring & Control
    --- Track Progress
    --- Manage Risks
    -- Closure
    --- Final Report
    --- Lessons Learned
    """
    print("\nPredefined Text Structure:")
    print(predefined_structure)

    image_path_2 = visualize_mindmap_from_text(predefined_structure, output_filename="project_management_mindmap")
    print(f"\nMind map image generated: {image_path_2}")

    # --- Example 3: Handling potential error ---
    print("\n--- Example 3: Handling potential error ---")
    invalid_structure = "This is not a valid structure"
    error_result = visualize_mindmap_from_text(invalid_structure)
    print(f"\nResult for invalid structure: {error_result}")
