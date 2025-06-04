import ast
import sys

def categorize_code(code):
    tree = ast.parse(code)
    categories = {
        'data_loading': [],
        'model_definition': [],
        'loss_optimizer': [],
        'training_loop': [],
        'evaluation': [],
        'checkpointing': []
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = getattr(node.func, 'attr', getattr(node.func, 'id', ''))
            try:
                code_snippet = ast.unparse(node)
            except Exception:
                continue

            if func in ['DataLoader', 'TensorDataset', 'transforms']:
                categories['data_loading'].append(code_snippet)
            elif func in ['Linear', 'Conv2d', 'Module']:
                categories['model_definition'].append(code_snippet)
            elif func in ['CrossEntropyLoss', 'Adam', 'SGD']:
                categories['loss_optimizer'].append(code_snippet)
            elif func in ['backward', 'step', 'zero_grad']:
                categories['training_loop'].append(code_snippet)
            elif func in ['eval', 'no_grad']:
                categories['evaluation'].append(code_snippet)
            elif func in ['save', 'load', 'state_dict']:
                categories['checkpointing'].append(code_snippet)

    return categories

def main():
    if len(sys.argv) != 2:
        print("Usage: python categorize_pytorch_code.py <code_file.py>")
        sys.exit(1)

    filepath = sys.argv[1]
    try:
        with open(filepath, 'r') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    categories = categorize_code(code)

    for step, snippets in categories.items():
        print(f"\n=== {step.replace('_', ' ').title()} ===")
        for line in snippets:
            print(f"- {line}")

if __name__ == "__main__":
    main()

