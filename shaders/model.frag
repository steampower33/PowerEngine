#version 450

layout(set=0, binding=2) uniform sampler2D tex;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(tex, vUV);
}
