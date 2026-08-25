# Exact runtime image used by the accepted Iteration-92 measurement.
#
# Keep this Dockerfile layer-free.  Adding RUN, COPY, ENV, or LABEL changes the
# image ID and makes the frozen service launcher reject the result.
ARG ATOM_RUNTIME_IMAGE="rocm/atom-dev@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976"
FROM ${ATOM_RUNTIME_IMAGE}
