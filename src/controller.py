from data.read_csv_folds import ReadCSV
from utility.doc_jsonl_to_json import ConvertToJson
from utility.json_to_traintuple import JsonToSpacy
from model.train_spacy import TrainSpacy
from model.model_evaluation import ModelEval
import time
import argparse


def main(args):
    if args.service == "generate_data":
        start_time = time.time()
        dict_arg = {
            "model_name": args.model_name,
            "output_fname": args.output_fname,
            "start_index": args.start_index,
            "end_index": args.end_index,
        }
        read_csv = ReadCSV(**dict_arg)
        df_dict = read_csv.read_csv_as_df()
        # read_csv.randomize_list(df_dict)
        d_json = read_csv.auto_annotate_data(df_dict)
        read_csv.json_to_jsonl(d_json)
        print(
            f"Auto annotated training data using {args.model_name} in {time.time() - start_time} seconds"
        )

    elif args.service == "jsonl_to_json":
        start_time = time.time()
        dict_arg = {"input_fname": args.input_fname, "output_fname": args.output_fname}
        ConvertToJson(**dict_arg).jsonl_to_json()
        print(
            f"Converted doccano json to spacy required json in {time.time() - start_time} seconds"
        )

    elif args.service == "json_to_train":
        start_time = time.time()
        dict_arg = {"input_fname": args.input_fname}
        convert_json = JsonToSpacy(**dict_arg)
        convert_json.to_spacyformat()
        print(
            f"Converted json to spacy training tuple format in {time.time() - start_time} seconds"
        )

    elif args.service == "train_model":
        start_time = time.time()
        dict_arg = {
            "model": None,
            "new_model_name": "Haircare model",
            "output_dir": args.output_dir,
            "n_iter": args.n_iter,
            "input_fname": args.input_fname,
            "entity_config_key": args.entity_key,
        }

        TrainSpacy(**dict_arg).train_spacymodel()
        print(f"Total training time = {time.time() - start_time} seconds")

    elif args.service == "evaluate_model":
        start_time = time.time()
        dict_arg = {
            "model_path": args.model_path,
            "validation_set": args.validation_set,
            "entity_config_key": args.entity_key,
        }
        eval_result = ModelEval(**dict_arg).model_evaluate()
        print(eval_result)
        print(f"validation executed in {time.time() - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--service", type=str, required=True, help="Service to be invoked"
    )
    parser.add_argument(
        "--model_name", type=str, required=False, help="Name of trained model"
    )
    parser.add_argument(
        "--output_fname",
        type=str,
        required=False,
        help="Name of annotated file to be saved",
    )
    parser.add_argument(
        "--input_fname",
        type=str,
        required=False,
        help="Name of doccano annotated jsonl file",
    )
    parser.add_argument(
        "--start_index", type=int, required=False, help="Start index of fold from csv"
    )
    parser.add_argument(
        "--end_index", type=int, required=False, help="Start index of fold from csv"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory name to save trained model",
    )
    parser.add_argument(
        "--n_iter", type=int, required=False, help="Number of training epochs"
    )
    parser.add_argument(
        "--entity_key",
        type=str,
        required=False,
        help="Key value for list of entity names from config file",
    )
    parser.add_argument(
        "--model_path", type=str, required=False, help="Name of trained model for validation"
    )
    parser.add_argument(
        "--validation_set",
        type=str,
        required=False,
        help="Name of validation set file",
    )
    args = parser.parse_args()

    main(args)


# SAMPLE COMMAND LINE ARGS
# python .\controller.py --service generate_data --model_name ORS_v3 --output_fname auto_annotated_fold1.jsonl --start_index 0 --end_index 0
# python .\controller.py --service train_model --output_dir ORS_v4 --n_iter 50 --input_fname doccano_f1_converted.json --entity_key haircare_entities
