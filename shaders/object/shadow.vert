#version 460

layout(set=0, binding=0) uniform ShadowUBO {
    mat4 lightVP;
} uShadow;

layout(set=1, binding=0) uniform ObjectUBO {
    mat4 model;
} uObj;

layout(location=0) in vec3 inPos;

void main()
{
    gl_Position = uShadow.lightVP * uObj.model * vec4(inPos, 1.0);
}
