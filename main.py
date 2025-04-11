import importlib.util
import os
from dotenv import load_dotenv
from langgraph.types import Command

# Load environment variables from .env file
load_dotenv()


def list_available_scripts(directory="."):
    """List all Python scripts in the specified directory.

    Returns a dictionary with key as the number before underscore in filename
    and value as the filename.
    """
    scripts = {}
    for file in os.listdir(directory):
        if file.endswith(".py") and file != "main.py" and file[0].isdigit():
            # Extract the number part (before underscore) from the script name
            filename = os.path.splitext(file)[0]  # Remove .py extension
            parts = filename.split("_", 1)
            if len(parts) > 1 and parts[0].isdigit():
                key = int(parts[0])
                scripts[key] = file
            else:
                # For files without the number_ format, use the filename as string key
                scripts[filename] = file
    return scripts


def import_graph(script_path):
    """Import and run a Python script from the given path."""
    try:
        # Get absolute path if relative path provided
        if not os.path.isabs(script_path):
            script_path = os.path.join(os.path.dirname(__file__), script_path)

        # Get module name from filename
        module_name = os.path.splitext(os.path.basename(script_path))[0]

        # Import module dynamically
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            raise ImportError(f"Could not load spec for module {module_name}")

        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"Could not load module {module_name}")

        spec.loader.exec_module(module)

        # Access the graph_builder
        if hasattr(module, "graph"):
            graph = module.graph
            print(f"Successfully loaded graph from {module_name}")
            return graph

        print(f"No graph_builder found in {module_name}")
        return None

    except Exception as e:
        print(f"Error running script: {e}")
        return False


# ===== EXECUTION FUNCTIONS =====
def stream_graph_updates(graph, user_input: str):
    """Process user input through the graph and stream the results."""

    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(
        {
            "messages": [{
                "role": "user",
                "content": user_input
            }],
            "mood": "neutral"
        },
        config,
        stream_mode="values",
    )

    for event in events:
        event["messages"][-1].pretty_print()
        # Check if the message has tool calls and needs human assistance
        if (hasattr(event["messages"][-1], "tool_calls")
                and event["messages"][-1].tool_calls
                and event["messages"][-1].tool_calls[0].get("name")
                == "human_assistance"):
            print("Human assistance needed!")
            human_response = input("Human: ")
            if human_response.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            human_command = Command(resume={"data": human_response})
            events = graph.stream(human_command, config, stream_mode="values")
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        try:
            _snapshot = graph.get_state(config)
            pass
        except Exception:
            pass
    

def main():
    print("Script Runner Utility")
    print("=====================")

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # List available scripts
    available_scripts = list_available_scripts(current_dir)

    if not available_scripts:
        print(f"No Python scripts found in {current_dir}")
        return

    print("\nAvailable scripts:")
    # Sort the keys to display scripts in numerical order
    sorted_keys = sorted(available_scripts.keys())
    for key in sorted_keys:
        script = available_scripts[key]
        script_name = os.path.splitext(script)[0]  # Remove .py extension
        print(f"{key}. {script_name}")

    # Get user input
    choice = input("\nEnter script number, full path, or 'q' to quit: ")
    if choice.lower() == "q":
        exit()

    script_key = int(choice)
    if script_key in available_scripts:
        script_path = available_scripts[script_key]
        graph = import_graph(script_path)

        if graph is None:
            print(f"No graph found in {script_path}")
            exit()

        # ===== VISUALIZATION (OPTIONAL) =====
        try:
            os.makedirs("./graphs", exist_ok=True)
            with open(f"./graphs/{script_path}.png", "wb") as f:
                f.write(graph.get_graph(xray=True).draw_mermaid_png())
        except Exception:
            # This requires some extra dependencies and is optional
            pass

        while True:
            user_input = input(
                "\n\n\n\n================================== User Input ==================================\n\nUser: "
            )
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                exit()

            stream_graph_updates(graph, user_input)

    else:
        print(f"No script found with number {script_key}")


if __name__ == "__main__":
    main()
