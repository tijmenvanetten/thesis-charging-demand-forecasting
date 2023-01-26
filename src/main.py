from data import load_data
from models import train_model


def run(*args, **kwargs):
    results = dict()
    X, y = load_data(args.filename)

    model = load_model()
    model = train_model(model, X, y)

    predictions = predict_model()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-f', '--filename', help='Select Prediction Model', type=str, default='data/raw/uk_sessions_2022-01-01_2022-11-01.csv')
    parser.add_argument(
        '-m', '--model', help='Select Prediction Model', type=str, required=True)
    parser.add_argument(
        '-n', '--nstep', help='Select n-step prediction evaluation', type=int, required=True)
    parser.add_argument(
        '-t', '--time-window', help='Select time-window over which prediction is aggregated',
        type=int, required=True)

    args = vars(parser.parse_args())

    results = run(args)
    print(results)
