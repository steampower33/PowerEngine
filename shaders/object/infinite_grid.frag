#version 450

layout(location = 0) in vec2 fragCoords;

layout(location = 0) out vec4 outColor;

const float grid_size = 1.0;

void main()
{
    vec2 xy = fragCoords / grid_size + vec2(0.5);
    outColor = vec4(xy, 0.0, 1.0);
}