#version 410
layout(location=0) in vec3 iv3vertex;
layout(location=1) in vec3 iv3color;
out vec3 color;

uniform mat4 um4p;
uniform mat4 um4v;
uniform mat4 um4m;

void main(){
    gl_Position = um4p * um4v * um4m * vec4(iv3vertex[0], iv3vertex[1], iv3vertex[2], 1.0);
    //gl_Position = vec4(iv3vertex, 1.0);
    color = iv3color;
}
