import pickle
import os

def load_copydays_dataset(pkl_path):
    """Load the Copydays dataset ground truth file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print("Dataset structure:")
    print("Type:", type(data))
    print("\nContent summary:")
    if isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        if len(data) > 0:
            print("First element type:", type(data[0]))
            print("First element preview:", str(data[0])[:100])
    elif isinstance(data, dict):
        print("Keys:", list(data.keys()))
        for key in data:
            print(f"\nKey: {key}")
            print("Value type:", type(data[key]))
            print("Value preview:", str(data[key])[:100])
    else:
        print("Data preview:", str(data)[:100])

    return data

if __name__ == "__main__":
    pkl_path = "copydays/gnd_copydays.pkl"
    if not os.path.exists(pkl_path):
        print(f"Error: File {pkl_path} not found!")
    else:
        data = load_copydays_dataset(pkl_path)
