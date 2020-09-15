#version 410

 in VS_OUT
{
    vec4 colour;
} fs_in;

 out vec4 fragment_colour;

void main()
{
     fragment_colour = fs_in.colour;
}

#version 150

uniform mat4 model;
uniform sampler2D tex;

uniform vec3 light_position;
uniform vec3 light_intensities; //a.k.a the color of the light

in vec2 fragTexCoord;
in vec3 fragNormal;
in vec3 fragVert;

out vec4 finalColor;

void main() {
    //calculate normal in world coordinates
    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 normal = normalize(normalMatrix * fragNormal);

    //calculate the location of this fragment (pixel) in world coordinates
    vec3 fragPosition = vec3(model * vec4(fragVert, 1));

    //calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = light_position - fragPosition;

    //calculate the cosine of the angle of incidence
    float brightness = dot(normal, surfaceToLight) / (length(surfaceToLight) * length(normal));
    brightness = clamp(brightness, 0, 1);

    //calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)
    vec4 surfaceColor = texture(tex, fragTexCoord);
    finalColor = fragVert; //vec4(1.0, 1.0, 1.0, 1.0); // vec4(brightness * light_intensities * surfaceColor.rgb, surfaceColor.a);
}


////////////////// vert

#version 150

uniform mat4 camera;
uniform mat4 model;

in vec3 vert;
in vec2 vertTexCoord;
in vec3 vertNormal;

out vec3 fragVert;
out vec2 fragTexCoord;
out vec3 fragNormal;

void main() {

    // Pass some variables to the fragment shader
    fragTexCoord = vertTexCoord;
    fragNormal = vertNormal;
    fragVert = vert;

    // Apply all matrix transformations to vert
    gl_Position = camera * model * vec4(vert, 1);
}

// // layout (std140) uniform Materials {
// uniform vec4 diffuse;
// uniform vec4 ambient;
// uniform vec4 specular;
// uniform float shininess;
// // };

// // layout (std140) uniform Lights {
// uniform vec3 l_dir;    // camera space
// // };

// in vec4 position;   // local space
// //in vec3 normal;     // local space

// // the data to be sent to the fragment shader
// out Data {
//     vec4 color;
// } DataOut;

// void main () {

//     // // set the specular term initially to black
//     vec4 spec = vec4(0.0);

//     vec3 n = m_normal; //normalize(m_normal * normal);

//     float intensity = max(dot(n, l_dir), 0.0);

//     // // if the vertex is lit compute the specular term
//     if (intensity > 0.0) {
//     //     // compute position in camera space
//          vec3 pos = vec3(m_viewModel * position);
//     //     // compute eye vector and normalize it
//          vec3 eye = normalize(-pos);
//     //     // compute the half vector
//          vec3 h = normalize(l_dir + eye);

//     //     // compute the specular term into spec
//          float intSpec = max(dot(h,n), 0.0);
//          spec = specular * pow(intSpec, shininess);
//     }
//     // // add the specular term
//     DataOut.color = ambient; //max(intensity *  diffuse + spec, ambient);

//     //DataOut.color =  2.0 * position + vec4 (0.5, 0.5, 0.5, 0.0);
//     gl_Position = m_pvm * m_viewModel * position;
// }
// ////////////////////////////////////////////////////
//#version 410

//in vec4 position;

//out VS_OUT {
//    vec4 colour;
//} vs_out;

//uniform mat4 mv_matrix;
//uniform mat4 projection_matrix;

//void main() {
//    gl_Position = projection_matrix * mv_matrix * position;
//    vs_out.colour = 2.0 * position + vec4 (0.5, 0.5, 0.5, 0.0);
//}
