from data.read_csv_folds import ReadCSV
from utility.doc_jsonl_to_json import ConvertToJson
from utility.json_to_traintuple import JsonToSpacy
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--service', type=str, required=True,
                        help="Service to be invoked")
    parser.add_argument('--model_name', type=str,
                        required=False, help="Relative path of model")
    parser.add_argument('--output_fname', type=str,
                        required=False, help="Relative path to save annotated file")
    parser.add_argument('--input_fname', type=str,
                        required=False, help="Relative path to input doccano annotated jsonl file")
    parser.add_argument('--start_index', type=int,
                        required=False, help="Start index of fold from csv")
    parser.add_argument('--end_index', type=int, required=False,
                        help="Start index of fold from csv")
    args = parser.parse_args()

    main(args)
