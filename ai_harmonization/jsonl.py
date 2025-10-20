import csv
import hashlib
import json
import logging
import os
import random


def split_randomize_jsonl_file(input_filename, output_filenames):
    """
    Opens a JSONL file, reads its contents, randomizes the order of the lines,
    and writes them out to multiple files with specified percentages.

    Args:
        input_filename (str): The path to the JSONL file to be randomized.
        output_filenames (list): A list of tuples containing the filename and percentage for each output file.

    Example usage:
        split_randomize_jsonl_file('input.jsonl', [('output1.jsonl', 40), ('validation.jsonl', 30), ('test_output.jsonl', 30)])
    """
    with open(input_filename, "r") as input_file:
        data = [line for line in input_file]

    if not data:
        for output_filename, _ in output_filenames:
            with open(output_filename, "w"):
                logging.warning(f"Writing nothing to {output_filename}.")
        return

    # Randomize the order of the lines
    logging.info(f"Shuffling {len(data)} lines from {input_filename}")
    random.shuffle(data)

    total_percentage = sum(percentage for _, percentage in output_filenames)
    assert round(total_percentage, 6) == 100.0

    i = 0
    percentage_to_get_to = 0
    for output_filename, percentage in output_filenames:
        lines_in_file = 0
        percentage_to_get_to += percentage
        with open(output_filename, "w") as output_file:
            # while the percentage of processed lines is less than the target percentage, continue
            # in case there's not an even split, ensure we don't go over the size of the overall data
            while ((float(i) / len(data)) * 100) <= percentage_to_get_to and i < len(
                data
            ):
                output_file.write(data[i])
                i += 1
                lines_in_file += 1

            if percentage_to_get_to == 100:
                # we can miss 1 line above due to rounding, so add it here
                i -= 1
                while i < len(data):
                    output_file.write(data[i])
                    i += 1
        logging.info(
            f"Wrote {lines_in_file} lines of total {len(data)} to {output_filename} to meet percentage {percentage}."
        )


def split_jsonl_into_separate_files(input_filepath, output_directory):
    """
    Reads a JSONL file and saves each line as a separate JSONL file
    in the specified output directory.

    Args:
        input_filepath (str): The path to the input JSONL file.
        output_directory (str): The directory where the output files will be saved.
    """
    import json
    import os

    with open(input_filepath, "r") as f:
        i = 0
        for item in f:
            output_filepath = os.path.join(output_directory, f"output_{i}.jsonl")
            with open(
                output_filepath,
                "w",
            ) as f2:
                f2.write(item)
                i += 1

    # Print a message to indicate completion
    print("JSONL data successfully converted and saved to separate files.")


def filter_jsonl_by_length(input_file, max_length, output_file=None):
    """
    Filters a JSONL file, keeping only lines below a specified character limit.

    Args:
        input_file: Path to the input JSONL file.
        max_length: The maximum character length for a line to be included.
        output_file: (Optional) Path to the output file.  If None,
                     a default filename is created based on the input filename.

    Returns:
        None.  Creates an output file with the filtered lines.  Raises
        FileNotFoundError if the input file doesn't exist.  Raises
        ValueError if max_length is not a positive integer.
    """

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer.")

    output_dir = "output"
    if output_file is None:
        # Create a default output filename in the same directory
        input_dir, input_filename = os.path.split(input_file)
        base_name, ext = os.path.splitext(input_filename)
        output_file = os.path.join(input_dir, f"{base_name}_filtered{ext}")
    else:
        # get path to output file:
        output_dir = os.path.dirname(output_file)

    # Check if the output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if (
                len(line.strip()) < max_length
            ):  # strip() removes leading/trailing whitespace
                outfile.write(line)
    logging.info(f"Filtered file written to: {output_file}")


def jsonl_to_csv(jsonl_path, csv_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Skip empty lines
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error on line {i+1}: {e}")
                print(line)
                continue

    if not records:
        print("No data found!")
        return

    headers = records[0].keys()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


def split_harmonization_jsonl_by_input_target_model(jsonl_path, output_dir):
    """
    Splits the JSONL into one file per unique input_target_model,
    using a SHA256 checksum of the input_target_model string as the filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map from input_target_model value to its checksum filename
    model_to_filename = {}
    grouped_rows = {}

    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            row = json.loads(line)
            key = str(row["input_target_model"])
            # Compute SHA256 checksum of the input_target_model string
            checksum = hashlib.sha256(key.encode("utf-8")).hexdigest()
            filename = f"target_model_{checksum}"
            if checksum not in model_to_filename:
                model_to_filename[checksum] = filename
                grouped_rows[filename] = []
            grouped_rows[filename].append(row)

    # Write out each group
    for filename, rows in grouped_rows.items():
        with open(
            os.path.join(output_dir, f"{filename}.jsonl"), "w", encoding="utf-8"
        ) as outfile:
            for row in rows:
                json.dump(row, outfile)
                outfile.write("\n")
    print(
        f"Wrote {len(model_to_filename)} JSONL files (one per unique input_target_model) to: {output_dir}"
    )
