ROOT_DIR=$(dirname ${0})
cd $ROOT_DIR/..

VERSION="$(git rev-parse --abbrev-ref HEAD)"
AWS_REGISTRY=743499434080.dkr.ecr.eu-west-1.amazonaws.com
PROJECT_NAME=3dml-instant-ngp

cache_params=""
if [ "$#" -eq 1 ]; then
    if [ "${1}" = "--force" ]; then
        cache_params="--no-cache"
    fi
fi

aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $AWS_REGISTRY

docker build --build-arg APP_ENV=build $cache_params -t $PROJECT_NAME:$VERSION -f .devcontainer/Dockerfile .

if ! [ $? -eq 0 ]; then
    echo "Failed to build docker"
    exit $?
fi

docker tag $PROJECT_NAME:$VERSION $AWS_REGISTRY/$PROJECT_NAME:$VERSION

docker push $AWS_REGISTRY/$PROJECT_NAME:$VERSION
