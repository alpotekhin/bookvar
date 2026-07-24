---
title: "Containerized server"
type: external-resource
status: imported-source
language: original
source_kind: readme
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week07_application_deployment/02_docker/README.md`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week07_application_deployment/02_docker/README.md) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.

* Put `vgg16.pt` inside this directory

* Install Docker and Docker-compose

  * On Ubuntu
    ```bash
    curl https://get.docker.com -L | bash
    DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
    mkdir -p $DOCKER_CONFIG/cli-plugins
    curl -SL https://github.com/docker/compose/releases/download/v2.33.0/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
    chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
    ```

  * On other systems
    * [Docker](https://docs.docker.com/engine/install)
    * [Docker-compose](https://docs.docker.com/compose/install/)

* Build and run a simple server:

    If you are using MacOS
    ```bash
    export DOCKER_BUILDKIT=0
    export COMPOSE_DOCKER_CLI_BUILD=0
    ```

* Build and run a simple Python server:
    
    ```bash
    docker compose up --build
    ```

* Build and run a production-ready server:

    ```bash
    docker compose -f docker-compose.production.yaml up --build
    ```

## Further reading

* Docker - https://docs.docker.com/
* Gunicorn - https://docs.gunicorn.org/en/stable/index.html
* Nginx - https://nginx.org/ru/ 
* Why gunicorn & nginx - https://docs.gunicorn.org/en/stable/deploy.html
* Supervisord - http://supervisord.org/
* GIL - https://habr.com/ru/post/84629/
