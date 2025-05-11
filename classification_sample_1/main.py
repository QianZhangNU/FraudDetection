import sys
import time
import logging
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import pandas as pd

import raw_daily_generation as RG
import batched_data_processing as BD
import load_fe_model as FM
import model_training as MT

log_file = 'pipeline.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def anomaly_detection_pipeline(date_of_year):
    logging.info(f"Starting anomaly detection with date_of_year: {date_of_year}")

    try:
        # Step 1: Generate raw data for anomaly detection
        logging.info("Step 1: Generate raw data")
        AF = RG.Analysis_Filter(date_of_year)
        df_mix_f_org = AF.filter_limit()

        # Step 2: Save the raw data
        logging.info("Step 2: Save the raw data")
        df_mix_f_org.to_parquet('res_raw_data_' + date_of_year + '.parquet')

        # Step 3: Process the data
        logging.info("Step 3: Data Processing")
        start_time = datetime(int(date_of_year[-4:]), int(date_of_year[:2]), int(date_of_year[2:4]))
        end_time = start_time + timedelta(days=1)

        df = BD.data_processing_pipeline(start_time, end_time)
        

        # Step 4: Feature Engineering
        logging.info("Step 4: Feature engineering")
        X_sorted = df.sort_values(by="_TIMESTEP")
        X_train_s = pd.read_parquet("training_set.parquet")

        feature_pipeline = FM.get_feature_pipeline()
        feature_pipeline.fit(X_train_s)
        X_test = feature_pipeline.transform(X_sorted)

        # Step 5: Load the models
        logging.info("Step 5: Load the model")
        XGBoost_Model, LG_Model, RF_Model = FM.load_model()
        voting_model = MT.voting_model(LG_Model, XGBoost_Model, RF_Model)

        # Step 6: Make the prediction
        logging.info("Step 6: Make the prediction")
        y_test_prob = voting_model.predict_proba(X_test)
        X_test['pred_ano'] = y_test_prob[:,1]

        # Step 7: Save the result
        logging.info("Step 7: Save the prediction")
        X_test.to_parquet("AM_result_" + date_of_year + ".parquet")

        logging.info("Fraudulant detection completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred in fraudulant detection: {e}")

    time.sleep(1)

if __name__ == "__main__":
    date_of_year = sys.argv[1]
    anomaly_detection_pipeline(date_of_year)
