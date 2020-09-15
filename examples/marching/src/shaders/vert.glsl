#version 410

in vec4 position;

out VS_OUT {
    vec4 colour;
} vs_out;

uniform mat4 m_viewModel;
uniform mat4 m_pvm;

void main() {
    gl_Position = m_pvm * m_viewModel * position;
    vs_out.colour = 2.0 * position + vec4 (0.5, 0.5, 0.5, 0.0);
}
