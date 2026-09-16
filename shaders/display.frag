#version 460 core
in vec2 vUv;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D uHdr;
layout(binding = 1) uniform sampler2D uRate;
uniform int uShowRate;
uniform float uExposure;
vec3 tonemap(vec3 x) {
    x *= uExposure;
    x = x / (1.0 + x);
    return pow(max(x, vec3(0.0)), vec3(1.0 / 2.2));
}
void main() {
    if (uShowRate != 0) {
        float r = texture(uRate, vUv).r;
        vec3 c = r < 1.5 ? vec3(0.95, 0.23, 0.18) : (r < 3.0 ? vec3(0.95, 0.72, 0.15) : vec3(0.18, 0.78, 0.34));
        outColor = vec4(c, 1.0);
        return;
    }
    outColor = vec4(tonemap(texture(uHdr, vUv).rgb), 1.0);
}
