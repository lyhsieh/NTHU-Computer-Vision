#version 410
#define pi 3.14159265359

in vec3 color;
layout(location = 0) out vec4 fragColor;

void main(){
    fragColor = vec4(color, 1.0);
}
