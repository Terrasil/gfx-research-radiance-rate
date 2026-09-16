#version 460 core

in VS_OUT {
    vec3 worldPosition;
    vec3 worldNormal;
    vec2 uv;
} v;

layout(location = 0) out vec4 outAlbedoRoughness;
layout(location = 1) out vec4 outNormalValid;
layout(location = 2) out vec4 outPositionValid;

layout(binding = 0) uniform sampler2D uBaseTexture;
uniform vec3 uBaseColor;
uniform float uRoughness;
uniform int uUseBaseTexture;
uniform float uRoughnessScale;

void main() {
    vec3 tex = uUseBaseTexture != 0 ? texture(uBaseTexture, v.uv).rgb : vec3(1.0);
    vec3 albedo = max(tex * uBaseColor, vec3(0.0));
    float roughness = clamp(uRoughness * uRoughnessScale, 0.04, 1.0);
    outAlbedoRoughness = vec4(albedo, roughness);
    outNormalValid = vec4(normalize(v.worldNormal) * 0.5 + 0.5, 1.0);
    outPositionValid = vec4(v.worldPosition, 1.0);
}
