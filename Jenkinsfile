pipeline {
    agent any

    stages {
        stage('Install dependencies') {
            steps {
                bat 'py -3.10 -m venv .venv'
                bat '.venv\\Scripts\\python.exe -m pip install -r requirements.txt'
            }
        }
        stage('Lint') {
            steps {
                bat '.venv\\Scripts\\python.exe -m ruff check src tests'
            }
        }
        stage('Unit tests') {
            steps {
                bat '.venv\\Scripts\\python.exe -m pytest -q'
            }
        }
        stage('Train and track models') {
            steps {
                bat '.venv\\Scripts\\python.exe -m src.eda'
                bat '.venv\\Scripts\\python.exe -m src.train'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'artifacts/**, models/**, mlflow.db', allowEmptyArchive: true
        }
        success {
            echo 'Lint, tests, EDA, and model training completed.'
        }
        failure {
            echo 'A pipeline stage failed. Review the stage output above.'
        }
    }
}