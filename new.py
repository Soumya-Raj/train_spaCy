from test.model_evaluation import ModelEval
import time
import argparse

def main(args):
    with open("validation_converted.txt") as f:
        EXAMPLES = f.read()
        #print(EXAMPLES)
    start_time = time.time()
    dict_arg = {"model_path": args.model_path,
                "validation_set": EXAMPLES, "entity_config_key": args.entity_key}
    results = ModelEval(**dict_arg).model_evaluate()
    print(results)
    print(
        f"validation executed in {time.time() - start_time} seconds"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required=True, help="Folder path of trained model")
    parser.add_argument('--validation_set', type = str, required=True, help="Folder path of validation set file")
    parser.add_argument('--entity_key', type = str, required=True, help="Folder path of trained model")
    
    args = parser.parse_args()

    main(args)
