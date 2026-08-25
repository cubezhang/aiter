# Exact public runtime parent used by the accepted Iteration-92 measurement.
#
# Keep this Dockerfile layer-free. A Docker build export still creates a new
# manifest even with no new filesystem layer, so strict replay uses
# bin/build_images.sh to pull and retag the parent manifest without rebuilding.
# This file documents the only allowed service runtime parent.
ARG ATOM_RUNTIME_IMAGE="rocm/atom-dev:nightly_202608201458@sha256:a5cfa1ab503af6e0f55e0ed83cd7e999edbea6940a38dba44aaeee9a22758976"
FROM ${ATOM_RUNTIME_IMAGE}
