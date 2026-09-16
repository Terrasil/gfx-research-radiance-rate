#version 460 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uMvp;

out VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} v;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    mat3 normalMatrix = transpose(inverse(mat3(uModel)));
    v.worldPosition = world.xyz;
    v.worldNormal = normalize(normalMatrix * aNormal);
    v.uv = aUv;
    gl_Position = uMvp * vec4(aPosition, 1.0);
}
