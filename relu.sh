#!/bin/bash
# relu.sh - Generate and compile ReLU shaders for all data types

set -e  # Exit on any error

# Output directories
SHADER_DIR="./shader"
SPV_DIR="./spv"
mkdir -p "$SHADER_DIR" "$SPV_DIR"

# Data type mapping: suffix : GLSL scalar type : zero literal : required extensions
declare -A TYPES=(
    ["int8"]="int8_t:int8_t(0):GL_EXT_shader_explicit_arithmetic_types GL_EXT_shader_8bit_storage"
    ["int16"]="int16_t:int16_t(0):GL_EXT_shader_explicit_arithmetic_types GL_EXT_shader_16bit_storage"
    ["int32"]="int:0:"
    ["int64"]="int64_t:int64_t(0):GL_EXT_shader_explicit_arithmetic_types"
    ["float32"]="float:0.0:"
    ["float64"]="double:0.0:"
)

echo "Generating and compiling ReLU shaders..."

for suffix in "${!TYPES[@]}"; do
    IFS=':' read -r scalar zero extensions <<< "${TYPES[$suffix]}"
    
    comp_file="$SHADER_DIR/relu_${suffix}.comp"
    spv_file="$SPV_DIR/relu_${suffix}.spv"

    # Generate .comp file
    cat > "$comp_file" <<EOF
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
$(for ext in $extensions; do
    if [ -n "$ext" ]; then echo "#extension $ext : require"; fi
done)

layout(push_constant) uniform Params {
    int64_t numel;
} params;

layout(std430, binding = 0) buffer Data {
    ${scalar} values[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (int64_t(idx) >= params.numel) return;
    values[idx] = max(values[idx], ${zero});
}
EOF

    # Compile to SPIR-V
    glslangValidator -V --target-env vulkan1.4 "$comp_file" -o "$spv_file"
    echo "✅ Compiled: $spv_file"
done

echo "All ReLU shaders compiled successfully!"