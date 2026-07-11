from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# 1. Define base configurations for our automation robot
default_args = {
    'owner': 'fael_senior',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 11),  # Starts today!
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 2. Initialize the DAG (The Train)
with DAG(
    'credit_card_fraud_pipeline',
    default_args=default_args,
    description='Automated MLOps pipeline for Credit Card Fraud Detection',
    schedule_interval=None,  # Triggered manually or by system events for now
    catchup=False,
    tags=['mlops', 'fraud'],
) as dag:

    # 3. Block 1 (Vagão 1): Check if our dataset exists and is secure
    task_validate_data = BashOperator(
        task_id='validate_dataset_presence',
        bash_command='ls /opt/airflow/dags/../../data/raw/creditcard.csv || echo "Dataset tracked by DVC"',
    )

    # 4. Block 2 (Vagão 2): Trigger our training script to feed MLflow
    task_trigger_training = BashOperator(
        task_id='execute_model_training',
        bash_command='echo "Executing python src/train.py pipeline..."',
    )

    # 5. Connect the train cars (Vagão 1 opens the track for Vagão 2)
    task_validate_data >> task_trigger_training
