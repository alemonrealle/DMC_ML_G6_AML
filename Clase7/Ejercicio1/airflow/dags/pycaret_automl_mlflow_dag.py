from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 5, 26),
}

dag = DAG(
    'pycaret_automl_mlflow',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Ejecuta AutoML con PyCaret y trackea en MLflow'
)

run_pycaret = BashOperator(
    task_id='run_pycaret_script',
    bash_command='cd /home/ubuntu/scripts && source ~/mlflow_project/venv-mlflow/bin/activate && python pycaret_automl_mlflow.py',
    dag=dag,
)

run_pycaret
