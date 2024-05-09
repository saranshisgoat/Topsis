import pandas as pd
import numpy as np
import sys


def get_ranks(result):
    """
    Assigns ranks to the 'Performance' column in the DataFrame in descending order.

    Parameters:
    - result (pd.DataFrame): Input DataFrame containing the 'Performance' column.

    Returns:
    - pd.DataFrame: DataFrame with an additional 'rank' column.
    """
    result['rank'] = result['Performance'].rank(ascending=False, method='min').astype('int')
    return result.iloc[:, 1:]


def calculate_performance_scores(euclidean_distance):
    """
    Calculates performance scores based on Euclidean distance.

    Parameters:
    - euclidean_distance (pd.DataFrame): DataFrame containing Euclidean distances.

    Returns:
    - pd.DataFrame: DataFrame with 'Performance' column.
    """
    nrows, _ = euclidean_distance.shape
    performance = []
    for i in range(nrows):
        x = euclidean_distance.iloc[i, 2] / (euclidean_distance.iloc[i, 2] + euclidean_distance.iloc[i, 1])
        performance.append(float(f'{x:.2f}'))
    performance = pd.DataFrame(performance)
    result = pd.concat([euclidean_distance.iloc[:, 0], performance], axis=1,)
    result.columns = [euclidean_distance.columns[0], "Performance"]
    return result


def calculate_euclidean_distances(data, row_names):
    """
    Calculates Euclidean distances between each row and the best and worst reference points.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the numerical data.
    - row_names (pd.Series): Series containing row names.

    Returns:
    - pd.DataFrame: DataFrame with Euclidean distances.
    """
    nrows, ncols = data.shape
    euclid_best = []
    euclid_worst = []
    for i in range(len(row_names)):
        dist_best = 0
        dist_worst = 0
        for j in range(ncols):
            dist_best += (data.iloc[i, j] - data.iloc[-2, j]) ** 2
            dist_worst += (data.iloc[i, j] - data.iloc[-1, j]) ** 2
        euclid_best.append(dist_best)
        euclid_worst.append(dist_worst)
    euclid_best = pd.DataFrame(np.sqrt(euclid_best))
    euclid_worst = pd.DataFrame(np.sqrt(euclid_worst))
    return pd.concat([row_names, euclid_best, euclid_worst], axis=1)


def get_ideal_reference_points(weighted_norm, impacts):
    """
    Generates ideal best and worst reference points based on impact values.

    Parameters:
    - weighted_norm (pd.DataFrame): DataFrame containing normalized and weighted data.
    - impacts (list): List of impact values (+ or -) for each column.

    Returns:
    - pd.DataFrame: DataFrame with ideal best and worst reference points.
    """
    ideal_best = []
    ideal_worst = []
    _, ncols = weighted_norm.shape
    if len(impacts) != ncols:
        raise ValueError("Number of impacts doesn't match the number of columns.")
    for _, col_data in weighted_norm.items():
        if impacts[i] == '-':
            ideal_best.append(min(col_data.values))
            ideal_worst.append(max(col_data.values))
        elif impacts[i] == '+':
            ideal_best.append(max(col_data.values))
            ideal_worst.append(min(col_data.values))
        else:
            raise ValueError("Impacts can only be + or -")
    ideal_best = pd.DataFrame([ideal_best], columns=list(weighted_norm.columns))
    ideal_worst = pd.DataFrame([ideal_worst], columns=list(weighted_norm.columns))
    weighted_norm = pd.concat([weighted_norm, ideal_best, ideal_worst], axis=0)
    return weighted_norm


def apply_weights(normalized_data, weights):
    """
    Applies weights to the normalized data.

    Parameters:
    - normalized_data (pd.DataFrame): DataFrame containing normalized data.
    - weights (list): List of weights for each column.

    Returns:
    - pd.DataFrame: DataFrame with weighted normalized data.
    """
    _, ncols = normalized_data.shape
    if len(weights) != ncols:
        raise ValueError("Number of weights doesn't match the number of columns.")
    weighted = normalized_data.copy().values
    weighted *= weights
    return pd.DataFrame(weighted, columns=normalized_data.columns)


def normalize_data(num_data, root_square_sum):
    """
    Normalizes numerical data.

    Parameters:
    - num_data (pd.DataFrame): DataFrame containing numerical data.
    - root_square_sum (np.ndarray): Array containing the root square sum for each column.

    Returns:
    - pd.DataFrame: DataFrame with normalized data.
    """
    normalized = num_data.copy().values
    normalized /= root_square_sum
    return pd.DataFrame(normalized, columns=num_data.columns)


def parse_args():
    """
    Parses command-line arguments.

    Returns:
    - tuple: Tuple containing input file path, weights, impacts, and output file path.
    """
    if len(sys.argv) != 5:
        raise ValueError("Wrong number of arguments. Expected 5.")
    input_file, weights, impacts, output_file = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    if ',' not in weights or ',' not in impacts:
        raise ValueError("Weights and impacts must be separated by commas.")
    try:
        weights = list(map(float, weights.split(",")))
    except ValueError:
        sys.exit('Invalid weights')
    impacts = impacts.split(",")
    return input_file, weights, impacts, output_file


def main():
    """
    Main function for executing the decision matrix analysis.

    Parses arguments, reads input file, performs analysis, and writes the result to the output file.
    """
    try:
        input_file, weights, impacts, output_file = parse_args()
    except ValueError as e:
        sys.exit(e)

    try:
        if input_file.split(".")[1] == "xlsx":
            data = pd.read_excel(input_file)
        elif input_file.split(".")[1] == "csv":
            data = pd.read_csv(input_file)
        else:
            raise ValueError("Invalid file extension. Supported formats are .xlsx and .csv.")
    except FileNotFoundError:
        sys.exit("File Not found. Kindly check the file path and provide the correct path.")
    except ValueError as e:
        sys.exit(e)

    try:
        if data.shape[1] < 3:
            raise ValueError('Number of columns in data less than 3')
        if not data.iloc[:, 1:].applymap(np.isreal).all().all():
            raise ValueError('Non-numeric values found in columns from 2nd to last.')
        num_data = data.loc[:, "P1":"P5"]
        root_square_sum = np.sqrt((num_data ** 2).sum()).values
        normalized_data = normalize_data(num_data, root_square_sum)
        del (num_data, root_square_sum)
        normalized_data = apply_weights(normalized_data, weights)
        normalized_data = get_ideal_reference_points(normalized_data, impacts)
        euclidean_distance = calculate_euclidean_distances(normalized_data, data.iloc[:, 0])
        del (normalized_data)
        result = calculate_performance
