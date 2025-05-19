import preprocess
import compare_algorithms

def main():
    df_train = preprocess.preprocess_data("train.csv")

    preprocess.analyze_data(df_train, "train.csv", "train")

    df_test = preprocess.preprocess_data("test.csv")

    preprocess.analyze_data(df_test, "test.csv", "test")

    models = compare_algorithms.train_models(df_train)

    acc_arr, f1_arr, auc_arr = compare_algorithms.evaluate_models(models, df_test)

    compare_algorithms.comparative_table(models, acc_arr, f1_arr, auc_arr)

    compare_algorithms.plot_confusion_matrices(models, df_test)

if __name__ == "__main__":
    main()
