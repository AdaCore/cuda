#version 330 core

// License https://creativecommons.org/licenses/by-nc/4.0/
// Author Joey de Vries
// URL https://learnopengl.com/
// Twitter https://twitter.com/JoeyDeVriez

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoords;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aAlbedo;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;
out vec3 Albedo;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

void main()
{
    TexCoords = aTexCoords;
    WorldPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(model) * aNormal;   
    Albedo = aAlbedo;

    gl_Position =  projection * view * vec4(WorldPos, 1.0);
}
