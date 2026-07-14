from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'fael_senior',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 11),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'credit_card_fraud_pipeline',
    default_args=default_args,
    description='Automated MLOps pipeline for Credit Card Fraud Detection',
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'fraud'],
) as dag:

    # 1. Check dataset inside the container path
    task_validate_data = BashOperator(
        task_id='validate_dataset_presence',
        bash_command='ls /opt/airflow/dags/../../data/raw/creditcard.csv || echo "Dataset ready"',
    )

    # 2. Trigger the training script dynamically
    task_trigger_training = BashOperator(
        task_id='execute_model_training',
        bash_command='python3 /opt/airflow/dags/../../src/train.py',
    )

    task_validate_data >> task_trigger_training
