import csv
import time
import argparse

def read_hyperparameters(task_id, file_path):
    """
    Reads the hyperparameters from a CSV file for the given task ID.
    """
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == task_id:
                return row
    return None


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--task_id', type=int, required=True,
                        help='SLURM task ID')
    args = parser.parse_args()

    # Read the hyperparameters for the given task ID
    hyperparameters = read_hyperparameters(args.task_id,
                                           'hyperparameters.csv')

    if hyperparameters is not None:
        # Write the hyperparameters to an output file
        output_file = f'output_{args.task_id}.txt'
        with open(output_file, 'w') as f:
            for key, value in hyperparameters.items():
                f.write(f'{key}: {value}\n')
        print(f"Hyperparameters for task {args.task_id} written to {output_file}")

        # Sleep for 10 minutes
        print("Sleeping for 5 minutes...")
        time.sleep(300)
        print("Done!")
    else:
        print(f"No hyperparameters found for task {args.task_id}")

if __name__ == "__main__":
    main()
