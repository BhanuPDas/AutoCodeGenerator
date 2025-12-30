# AutoCodeGenerator
Repository contains React UI, Java service, AI Inference service and other configurations for code generator project.
Each Service has its own Dockerfile. Run the docker file to build and run the container service.

## Steps to build images and run the containers:

1. CodeLlama-13B inference service: Since this uses model from meta repository provided by hugging face, we must have access to the HF repository, else service won't be able to download the model.
    Pass the access token while building the image and running it inorder to download the model. 

    Build Image: **docker build --build-arg HF_TOKEN=xxxx -t llama-inference .**

    Run: **docker run --name llama-infer --gpus all -p 5050:5050 -e HF_TOKEN=xxxx -d llama-inference:latest**

    Note: CodeLlama-13B needs a minimum of 27.1 GB of GPU to run. Make sure the prerequisites are met.

2. Java Service: (Used for collecting various metrics)

    Build Image: **docker build -t code-gen-ser .**

    Run: **docker run --name code-gen-ser -p 4010:4010 -d code-gen-ser:latest**

3. React UI: UI to trigger request from browser (Currently, reads predefined set of prompts from a json file).

    Build Image: **docker build -t code-gen-ui .**

    Run: **docker run --name ui -p 5173:80 -d code-gen-ui:latest**

## Other Software:
Apart from the above services, the architecture has dependency on other software and are required.
1. Grafana
2. Redis
3. PostgreSQL
4. Prometheus
