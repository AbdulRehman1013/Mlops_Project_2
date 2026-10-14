pipeline {
    agent any
    
    environment {
        MLFLOW_TRACKING_URI = "sqlite:////mnt/c/Users/ASUS/OneDrive/Desktop/MLOPS/project2/mlflow.db"
        PYTHONPATH = "${WORKSPACE}/src"
        STUDENT_EMAIL = "abdulrehman5662423@gmail.com"
        TEACHER_EMAIL = "sikandar.hayat@is.uol.edu.pk"
    }
    
    stages {
        stage('Checkout & Setup') {
            steps {
                echo "Setting up virtual environment..."
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Data Ingestion') {
            steps {
                echo "Running data ingestion..."
                sh '. venv/bin/activate && python3 src/data_ingest.py'
            }
        }
        
        stage('Model Training & MLflow Registry') {
            steps {
                script {
                    echo "Using MLflow Tracking URI: ${env.MLFLOW_TRACKING_URI}"
                    if (env.BRANCH_NAME == 'dev') {
                        echo "Running Dev Pipeline: Saving as Challenger"
                        sh ". venv/bin/activate && export MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} && python3 src/train.py --stage dev"
                    } else if (env.BRANCH_NAME == 'main') {
                        echo "Running Pre-prod Pipeline: Validating Challenger"
                        sh ". venv/bin/activate && export MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} && python3 src/train.py --stage pre-prod"
                    } else {
                        echo "Running Prod Pipeline (Release Tag): Promoting to Champion"
                        sh ". venv/bin/activate && export MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} && python3 src/train.py --stage prod"
                    }
                }
            }
        }
        
        stage('Model Testing & Evaluation') {
            steps {
                script {
                    sh '. venv/bin/activate && pytest tests/test_model.py'
                    if (env.BRANCH_NAME == 'dev') {
                        sh ". venv/bin/activate && export MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} && python3 src/evaluate.py Challenger"
                    } else if (env.BRANCH_NAME == 'main') {
                        sh ". venv/bin/activate && export MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} && python3 src/evaluate.py Challenger-pre-test"
                    } else {
                        sh ". venv/bin/activate && export MLFLOW_TRACKING_URI=${env.MLFLOW_TRACKING_URI} && python3 src/evaluate.py Champion"
                    }
                }
            }
        }
    }
    
    post {
        success {
            script {
                def targetName = env.TAG_NAME ? env.TAG_NAME : env.BRANCH_NAME
                def projectName = "Abdul Rehman's MLOps Pipeline (${targetName})"
                echo "Pipeline completed successfully. Sending notification email..."
                emailext (
                    subject: "SUCCESS: ${projectName} Completed Successfully",
                    body: "Hello,\n\nThe pipeline task for ${projectName} has completed successfully.\n\nYou can track and view the model registry (Challenger/Champion) on MLflow UI here: http://localhost:5000\n\nBest regards,\nMlOps Automation",
                    to: "abdulrehman5662423@gmail.com, sikandar.hayat@is.uol.edu.pk",
                    recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'ListRecipientProvider']]
                )
            }
        }
        failure {
            script {
                def targetName = env.TAG_NAME ? env.TAG_NAME : env.BRANCH_NAME
                def projectName = "Abdul Rehman's MLOps Pipeline (${targetName})"
                echo "Pipeline failed. Sending alert email..."
                emailext (
                    subject: "FAILED: ${projectName} Encountered an Error",
                    body: "Hello,\n\nThe pipeline task for ${projectName} has failed during execution. Please check Jenkins logs.\n\nMLflow UI: http://localhost:5000\n\nBest regards,\nMLOps Automation",
                    to: "abdulrehman5662423@gmail.com, sikandar.hayat@is.uol.edu.pk",
                    recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'ListRecipientProvider']]
                )
            }
        }
    }
}