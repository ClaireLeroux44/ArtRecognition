# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* ArtRecognition/*.py

black:
	@black scripts/* ArtRecognition/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr ArtRecognition-*.dist-info
	@rm -fr ArtRecognition.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
	'{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
	'{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''


# ---------------------------------------
# - API -
# ---------------------------------------

#### LOCAL
run_api:
	@uvicorn api.fast:app --host "0.0.0.0" --port 8000 --reload

run_api_test:
	@uvicorn api.fast_test:app --host "0.0.0.0" --port 8000 --reload

#### BUILD AND DEPLOY
PROJECT_ID=artrecognition
build_api :
	@docker build -t eu.gcr.io/${PROJECT_ID}/artrecognition-api -f Dockerfile_API .
	#@gcloud docker -- push eu.gcr.io/$(PROJECT_ID)/artrecognition-api

docker_api_locally:
	@docker run -e PORT=8000 -p 8000:8000 eu.gcr.io/${PROJECT_ID}/artrecognition-api


deploy_api :
	@docker push eu.gcr.io/${PROJECT_ID}/artrecognition-api
	@gcloud run deploy --image eu.gcr.io/${PROJECT_ID}/artrecognition-api --platform managed --cpu 4 --memory 8Gi --region "europe-west1" --port 8000

##### Google Storage params
BUCKET_NAME=art-recognition-app
BUCKET_TRAINING_FOLDER=trainings

##### Machine configuration - - - - - - - - - - - - - - - -
REGION=europe-west1
PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=2.2

##### Package params  - - - - - - - - - - - - - - - - - - -
PACKAGE_NAME=ArtRecognition
FILENAME=trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -
JOB_NAME=ArtRecognition_model_training_$(shell date +'%Y%m%d_%H%M%S')

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
		--scale-tier=basic-gpu
