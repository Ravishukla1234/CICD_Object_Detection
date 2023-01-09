#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# There are 3 arguments in this script:
#    - image - required, this will be used as the image on the local machine and combined with the account and region to form the repository name for ECR;
#    - tag - optional, if provided, it will be used as ":tag" of your image; otherwise, ":latest" will be used;
#    - Dockerfile - optional, if provided, then docker will try to build image from provided dockerfile (e.g. "Dockerfile.serving"); otherwise, default "Dcokerfile" will be used.
# Usage examples:
#    1. "./build_and_push.sh d2-sm-coco-serving debug Dockerfile.serving"
#    2. "./build_and_push.sh d2-sm-coco v2"

image=$1
tag=$2
dockerfile=$3
region=$4

REGION="ap-south-1"
ACCOUNT="296512243111"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com
# loging to your private ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com



echo "image $image" 
echo "tag $tag" 
echo "dockerfile $dockerfile" 
echo "region $region"
echo "account id $ACCOUNT_ID"
if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-east-1 if none defined)


echo "Working in region $region"


if [ "$tag" == "" ]
then
    fullname="${ACCOUNT_ID}.dkr.ecr.$region.amazonaws.com/${image}:latest"
    echo "yes $fullname"
else
    fullname="${ACCOUNT_ID}.dkr.ecr.$region.amazonaws.com/${image}:${tag}"
    echo "yo $fullname"
fi


# If the repository doesn't exist in ECR, create it.
echo "check"
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1
echo "check done"
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi
echo "check done done"

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region $region --no-include-email)

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

if [ "$dockerfile" == "" ]
then
    docker build  -t ${image} . --build-arg REGION=$region
else
    docker build -t ${image} . -f ${dockerfile} --build-arg REGION=$region
fi

docker tag ${image} ${fullname}
docker push ${fullname}
