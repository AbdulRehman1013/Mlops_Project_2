def pipelineInfo() {
    if (env.TAG_NAME) {
        return [stage: 'prod', loadAlias: 'Champion', label: env.TAG_NAME]
    } else if (env.BRANCH_NAME == 'main') {
        return [stage: 'pre-prod', loadAlias: 'Challenger-pre-test', label: 'main']
    } else {
        return [stage: 'dev', loadAlias: 'Challenger', label: env.BRANCH_NAME]
    }
}

pipeline {
    agent any

    environment {
        MLFLOW_TRACKING_URI  = "sqlite:////var/lib/jenkins/mlflow/mlflow.db"
        MLFLOW_ARTIFACT_ROOT = "/var/lib/jenkins/mlflow/artifacts"
        PYTHONPATH           = "${WORKSPACE}/src"
        EMAILS               = "abdulrehman5662423@gmail.com, sikandar.hayat@is.uol.edu.pk"
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
                sh '. venv/bin/activate && python3 src/data_ingest.py'
            }
        }

        stage('Model Train & Register') {
            steps {
                script {
                    def info = pipelineInfo()
                    echo "Running ${info.stage} pipeline (${info.label})"
                    sh ". venv/bin/activate && python3 src/train.py --stage ${info.stage}"
                }
            }
        }

        stage('Model Deploy (MLflow)') {
            steps {
                script {
                    def info = pipelineInfo()
                    def alias = (info.stage == 'dev') ? 'Challenger' : (info.stage == 'pre-prod') ? 'Challenger-pre-test' : 'Champion'
                    sh ". venv/bin/activate && python3 src/deploy.py ${alias} ${info.stage}"
                }
            }
        }

        stage('Model Testing & Evaluation') {
            when { expression { return !env.TAG_NAME } }
            steps {
                script {
                    def info = pipelineInfo()
                    def alias = (info.stage == 'dev') ? 'Challenger' : 'Challenger-pre-test'
                    sh '. venv/bin/activate && pytest tests/test_model.py'
                    sh ". venv/bin/activate && python3 src/evaluate.py ${alias}"
                }
            }
        }
    }

    post {
        success {
            emailext(
                subject: "SUCCESS: MLOps Pipeline (${pipelineInfo().label})",
                body: "Pipeline ${pipelineInfo().stage} completed successfully.\nBuild: ${env.BUILD_URL}",
                to: "${EMAILS}"
            )
        }
        failure {
            emailext(
                subject: "FAILED: MLOps Pipeline (${pipelineInfo().label})",
                body: "Pipeline ${pipelineInfo().stage} failed.\nCheck logs: ${env.BUILD_URL}console",
                to: "${EMAILS}"
            )
        }
    }
}
