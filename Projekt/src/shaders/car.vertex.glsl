#version 120

uniform mat4 MVP;

attribute vec2 vertexPosition;

void main()
{
    vec4 vertex = vec4(vertexPosition, 0.001f, 1.0f);
    gl_Position = MVP * vertex;
}
