#!/bin/bash
# Test script for the PROTEUS Docker image
#
# Usage: ./tools/test_docker_image.sh

set -e

# Ensure DOCKER_HOST is set for Colima
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"

echo "=========================================="
echo "PROTEUS Docker Image Test Suite"
echo "=========================================="
echo ""

# Test 1: Check if image exists
echo "Test 1: Checking if image exists..."
if docker images | grep -q "proteus-test.*local"; then
    echo "✅ Image found: proteus-test:local"
else
    echo "❌ Image not found. Please build it first:"
    echo "   docker build -t proteus-test:local ."
    exit 1
fi
echo ""

# Test 2: Check image size
echo "Test 2: Checking image size..."
IMAGE_SIZE=$(docker images proteus-test:local --format "{{.Size}}")
echo "   Image size: $IMAGE_SIZE"
echo ""

# Test 3: Test container can start
echo "Test 3: Testing container startup..."
docker run --rm proteus-test:local bash -c "echo 'Container started successfully'" || {
    echo "❌ Container failed to start"
    exit 1
}
echo "✅ Container starts successfully"
echo ""

# Test 4: Check Python version
echo "Test 4: Checking Python version..."
PYTHON_VERSION=$(docker run --rm proteus-test:local python --version)
echo "   $PYTHON_VERSION"
echo ""

# Test 5: Check Julia version
echo "Test 5: Checking Julia version..."
JULIA_VERSION=$(docker run --rm proteus-test:local julia --version)
echo "   $JULIA_VERSION"
echo ""

# Test 6: Check if SOCRATES is built
echo "Test 6: Checking SOCRATES..."
docker run --rm proteus-test:local bash -c "ls -la /opt/proteus/socrates/bin/ | head -5" || {
    echo "⚠️  SOCRATES binaries not found"
}
echo ""

# Test 7: Check if SPIDER is built
echo "Test 7: Checking SPIDER..."
docker run --rm proteus-test:local bash -c "test -f /opt/proteus/SPIDER/spider && echo '✅ SPIDER binary exists' || echo '⚠️  SPIDER binary not found'"
echo ""

# Test 8: Check if PETSc is built
echo "Test 8: Checking PETSc..."
docker run --rm proteus-test:local bash -c "test -d /opt/proteus/petsc && echo '✅ PETSc directory exists' || echo '⚠️  PETSc directory not found'"
echo ""

# Test 9: Check Python packages
echo "Test 9: Checking Python packages..."
docker run --rm proteus-test:local pip list | grep -E "proteus|janus|mors|calliope" || {
    echo "⚠️  Some PROTEUS packages not found"
}
echo ""

# Test 10: Try importing proteus
echo "Test 10: Testing Python imports..."
docker run --rm proteus-test:local python -c "import proteus; print('✅ proteus imports successfully')" || {
    echo "❌ Failed to import proteus"
    exit 1
}
echo ""

# Test 11: Run pytest collection
echo "Test 11: Testing pytest collection..."
docker run --rm -w /opt/proteus proteus-test:local pytest --collect-only tests/examples/test_marker_usage.py | head -20
echo ""

# Test 12: Check environment variables
echo "Test 12: Checking environment variables..."
docker run --rm proteus-test:local bash -c 'echo "FWL_DATA=$FWL_DATA"; echo "RAD_DIR=$RAD_DIR"; echo "PETSC_DIR=$PETSC_DIR"'
echo ""

echo "=========================================="
echo "Test Suite Complete!"
echo "=========================================="
echo ""
echo "To run the container interactively:"
echo "  docker run -it --rm proteus-test:local bash"
echo ""
echo "To run tests inside the container:"
echo "  docker run --rm -w /opt/proteus proteus-test:local pytest -m unit"
echo ""
