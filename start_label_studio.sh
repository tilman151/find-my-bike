export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/tilman/Programming/find-my-bike/
(trap 'kill 0' SIGINT; label-studio start & label-studio-ml start label_studio_backend)